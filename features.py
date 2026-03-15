"""
features.py — Complete feature engineering for MAG Energy Solutions Challenge.

ONE file, all features. Replaces both the old features.py and price_predictor.py.

What we are predicting:
    profit = |PR_{M+1}| - C_{M+1} > 0   →   is_profitable (binary label)

    BOTH PR and C for M+1 are unknown at decision time (day 7 of month M).
    We build features that estimate them using only allowed data:
      - sim_monthly PSM     → proxy for future |PR|   (allowed: produced before day 7)
      - historical prices   → persistence signal for PR
      - historical costs    → estimate future C
      - sim ACTIVATIONLEVEL → predicts whether EID will activate at all

Key findings from EDA that shape feature design:
    - 3.29% base profitability rate (not 5% — we confirmed 3.29%)
    - 98.4% of candidates have C=0 → task = predict activation (PR > 0)
    - Seasonal lift: P(profitable | profitable LY) = 27.6% vs 1.9% (+25.7%)
    - 50:1 profit asymmetry → optimize recall, always select 100
    - ~247 EIDs have ≥ 50% profit rate → persistence is concentrated

Anti-leakage:
    For target_month M+1, all features use only data from months < M+1.
    Specifically:
      - prices:       up to and including day 7 of month M (we use full months < M+1)
      - costs:        up to and including month M
      - sim_monthly:  for month M+1 (explicitly allowed — produced before day 7 of M)
      - sim_daily:    not used in features (used for calibration only, optional)
"""

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent if SCRIPT_DIR.name in ("analysis", "src") else SCRIPT_DIR
sys.path.insert(0, str(ROOT_DIR))

from data_loader import add_months


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    candidates:   pd.DataFrame,
    target_month: str,
    truth_past:   pd.DataFrame,
    sim_agg:      Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the complete feature matrix for all candidates in one target month.

    Parameters
    ----------
    candidates   : DataFrame (EID, PEAKID) — the universe of candidates to score
                   Typically: sim_candidates filtered to target_month
    target_month : str e.g. '2022-06' — the month we predict (M+1)
    truth_past   : full truth table filtered STRICTLY to months < target_month
                   Columns: EID, MONTH, PEAKID, PR, C, profit, is_profitable
                   Anti-leakage: caller must enforce this filter
    sim_agg      : aggregated sim_monthly for target_month
                   Output of aggregate_sim_across_scenarios()
                   Pass None if sim data unavailable

    Returns
    -------
    DataFrame: one row per (EID, PEAKID) with all features + TARGET_MONTH
    """
    t0 = time.time()

    # Reference months for seasonal lookups
    same_month_ly  = add_months(target_month, -12)
    same_month_2ly = add_months(target_month, -24)
    same_month_3ly = add_months(target_month, -36)

    # ── Pre-filter truth to lookback window once ──────────────────────────────
    # 36 months back for 3-year seasonal features
    # This keeps the truth table small before the groupby
    lookback_start = add_months(target_month, -36)
    truth_window   = truth_past[truth_past["MONTH"] >= lookback_start].copy()

    # ── Pre-group by (EID, PEAKID) once — O(n) not O(n²) ────────────────────
    # Without this, filtering a 750k-row truth table 12,000 times per month
    # would take hours. With pre-grouping, each lookup is a dict key check.
    grouped = {
        key: grp.sort_values("MONTH")
        for key, grp in truth_window.groupby(["EID", "PEAKID"])
    }
    empty = pd.DataFrame(columns=truth_window.columns)

    # ── Index sim_agg by (EID, PEAKID) for O(1) lookup ───────────────────────
    sim_index = {}
    if sim_agg is not None and len(sim_agg) > 0:
        for _, row in sim_agg.iterrows():
            sim_index[(int(row["EID"]), int(row["PEAKID"]))] = row

    # ── Build one feature row per candidate ──────────────────────────────────
    rows = []
    for _, cand in candidates.iterrows():
        eid    = int(cand["EID"])
        peakid = int(cand["PEAKID"])

        g    = grouped.get((eid, peakid), empty)
        sim  = sim_index.get((eid, peakid), None)
        feat = {"EID": eid, "PEAKID": peakid}

        feat.update(_seasonal(g, same_month_ly, same_month_2ly, same_month_3ly))
        feat.update(_persistence(g))
        feat.update(_pr_features(g))
        feat.update(_cost_features(g, same_month_ly, same_month_2ly))
        feat.update(_sim_features(sim))
        feat.update(_combined(feat))
        feat.update(_temporal(target_month))
        feat["n_months_history"] = len(g)
        feat["has_any_history"]  = int(len(g) > 0)

        rows.append(feat)

    result = pd.DataFrame(rows).fillna(0)
    result["TARGET_MONTH"] = target_month

    elapsed = time.time() - t0
    print(f"    features: {len(result):,} candidates | "
          f"{len(result.columns)-3} features | {elapsed:.1f}s")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 1 — SEASONAL PERSISTENCE
# The single strongest signal: +25.7% lift from EDA
# ─────────────────────────────────────────────────────────────────────────────

def _seasonal(
    g: pd.DataFrame,
    same_month_ly: str,
    same_month_2ly: str,
    same_month_3ly: str,
) -> dict:
    feat = {}

    for suffix, month in [
        ("ly",  same_month_ly),
        ("2ly", same_month_2ly),
        ("3ly", same_month_3ly),
    ]:
        row = g[g["MONTH"] == month]
        if len(row) > 0:
            r = row.iloc[0]
            feat[f"profitable_{suffix}"]  = int(r["is_profitable"])
            feat[f"pr_{suffix}"]          = float(r["PR"])
            feat[f"profit_{suffix}"]      = float(r["profit"])
            feat[f"c_{suffix}"]           = float(r["C"])
        else:
            feat[f"profitable_{suffix}"]  = 0
            feat[f"pr_{suffix}"]          = 0.0
            feat[f"profit_{suffix}"]      = 0.0
            feat[f"c_{suffix}"]           = 0.0

    # How many of the last 3 same-month-LY observations were profitable?
    feat["seasonal_consistency"] = (
        feat["profitable_ly"] +
        feat["profitable_2ly"] +
        feat["profitable_3ly"]
    )  # 0-3 scale
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 2 — HISTORICAL PROFITABILITY PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def _persistence(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return {
            "profit_rate_3m":       0.0,
            "profit_rate_6m":       0.0,
            "profit_rate_12m":      0.0,
            "profit_rate_24m":      0.0,
            "profit_last_month":    0,
            "profit_streak":        0,
            "profitable_last_3_all":0,
            "profit_avg_3m":        0.0,
            "profit_avg_12m":       0.0,
            "profit_max_12m":       0.0,
        }

    feat = {
        "profit_rate_3m":    float(g["is_profitable"].tail(3).mean()),
        "profit_rate_6m":    float(g["is_profitable"].tail(6).mean()),
        "profit_rate_12m":   float(g["is_profitable"].tail(12).mean()),
        "profit_rate_24m":   float(g["is_profitable"].tail(24).mean()),
        "profit_last_month": int(g["is_profitable"].iloc[-1]),
        "profit_avg_3m":     float(g["profit"].tail(3).mean()),
        "profit_avg_12m":    float(g["profit"].tail(12).mean()),
        "profit_max_12m":    float(g["profit"].tail(12).max()),
    }

    # Consecutive profitable months ending at cutoff
    streak = 0
    for val in reversed(g["is_profitable"].values.tolist()):
        if val == 1:
            streak += 1
        else:
            break
    feat["profit_streak"]         = streak
    feat["profitable_last_3_all"] = int(
        len(g) >= 3 and g["is_profitable"].tail(3).sum() == 3
    )
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 3 — REALIZED PRICE (PR) FEATURES
# Key insight: 98.4% of candidates have C=0, so profitability ≈ PR > 0
# Main task: predict ACTIVATION, not price magnitude
# ─────────────────────────────────────────────────────────────────────────────

def _pr_features(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return {
            "pr_avg_3m":           0.0,
            "pr_avg_12m":          0.0,
            "pr_last":             0.0,
            "pr_std_12m":          0.0,
            "pr_max_12m":          0.0,
            "pr_nonzero_rate_12m": 0.0,
            "pr_nonzero_rate_24m": 0.0,
            "pr_ever_nonzero":     0,
            "pr_avg_when_active":  0.0,
            "pr_count_active_12m": 0,
        }

    active = g[g["PR"] > 0]
    feat = {
        "pr_avg_3m":           float(g["PR"].tail(3).mean()),
        "pr_avg_12m":          float(g["PR"].tail(12).mean()),
        "pr_last":             float(g["PR"].iloc[-1]),
        "pr_std_12m":          float(g["PR"].tail(12).std()) if len(g) >= 2 else 0.0,
        "pr_max_12m":          float(g["PR"].tail(12).max()),
        # Activation rate — most important PR feature
        "pr_nonzero_rate_12m": float((g["PR"] > 0).tail(12).mean()),
        "pr_nonzero_rate_24m": float((g["PR"] > 0).tail(24).mean()),
        "pr_ever_nonzero":     int(g["PR"].sum() > 0),
        "pr_count_active_12m": int((g["PR"] > 0).tail(12).sum()),
        # Average PR when actually active — not diluted by zero months
        # An EID earning $5000 every 3 months has pr_avg_12m=$1667 (misleading)
        # but pr_avg_when_active=$5000 (the true signal when it fires)
        "pr_avg_when_active":  float(active["PR"].mean()) if len(active) > 0 else 0.0,
    }
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 4 — COST (C) FEATURES
# C for M+1 is unknown at decision time — must estimate from history
# ─────────────────────────────────────────────────────────────────────────────

def _cost_features(
    g: pd.DataFrame,
    same_month_ly: str,
    same_month_2ly: str,
) -> dict:
    if len(g) == 0:
        return {
            "c_last":            0.0,
            "c_avg_3m":          0.0,
            "c_avg_12m":         0.0,
            "c_same_month_ly":   0.0,
            "c_same_month_2ly":  0.0,
            "c_nonzero_rate_12m":0.0,
            "c_avg_when_nonzero":0.0,
            "c_std_12m":         0.0,
            "c_trend":           0.0,
            "is_zero_cost_hist": 1,
        }

    c_ly_row  = g[g["MONTH"] == same_month_ly]
    c_2ly_row = g[g["MONTH"] == same_month_2ly]
    c_12m     = g["C"].tail(12)

    nonzero_c = g[g["C"] > 0]

    feat = {
        "c_last":            float(g["C"].iloc[-1]),
        "c_avg_3m":          float(g["C"].tail(3).mean()),
        "c_avg_12m":         float(c_12m.mean()),
        "c_same_month_ly":   float(c_ly_row["C"].iloc[0])  if len(c_ly_row) > 0  else 0.0,
        "c_same_month_2ly":  float(c_2ly_row["C"].iloc[0]) if len(c_2ly_row) > 0 else 0.0,
        "c_nonzero_rate_12m":float((c_12m > 0).mean()),
        "c_avg_when_nonzero":float(nonzero_c["C"].mean()) if len(nonzero_c) > 0 else 0.0,
        "c_std_12m":         float(c_12m.std()) if len(c_12m) >= 2 else 0.0,
        "is_zero_cost_hist": int(g["C"].sum() == 0),
    }

    # Cost trend: is cost increasing or decreasing?
    c6 = g["C"].tail(6).values
    if len(c6) >= 3:
        feat["c_trend"] = float(np.polyfit(range(len(c6)), c6, 1)[0])
    else:
        feat["c_trend"] = 0.0

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 5 — SIMULATION (PSM / ACTIVATIONLEVEL)
# Primary forward-looking signal — allowed because sim for M+1 is
# produced BEFORE day 7 of month M (explicitly stated in case doc)
# ─────────────────────────────────────────────────────────────────────────────

def _sim_features(sim_row: Optional[pd.Series]) -> dict:
    zeros = {
        "psm_abs_mean":       0.0,
        "psm_abs_min":        0.0,
        "psm_abs_max":        0.0,
        "psm_abs_std":        0.0,
        "psm_scenario_agree": 0,
        "psm_s1":             0.0,
        "psm_s2":             0.0,
        "psm_s3":             0.0,
        "act_mean":           0.0,
        "act_max":            0.0,
        "act_pos_frac":       0.0,
        "wind_mean":          0.0,
        "solar_mean":         0.0,
        "hydro_mean":         0.0,
        "nonrenew_mean":      0.0,
        "external_mean":      0.0,
        "load_mean":          0.0,
        "transmission_mean":  0.0,
        "source_impact_sum":  0.0,
        "sim_in_universe":    0,
        "sim_predicts_active":0,
    }
    if sim_row is None:
        return zeros

    return {
        "psm_abs_mean":       float(sim_row.get("psm_abs_mean", 0)),
        "psm_abs_min":        float(sim_row.get("psm_abs_min", 0)),
        "psm_abs_max":        float(sim_row.get("psm_abs_max", 0)),
        "psm_abs_std":        float(sim_row.get("psm_abs_std", 0)),
        "psm_scenario_agree": int(sim_row.get("scenario_agree", 0)),
        "psm_s1":             float(sim_row.get("psm_s1", 0)),
        "psm_s2":             float(sim_row.get("psm_s2", 0)),
        "psm_s3":             float(sim_row.get("psm_s3", 0)),
        "act_mean":           float(sim_row.get("act_mean", 0)),
        "act_max":            float(sim_row.get("act_max", 0)),
        "act_pos_frac":       float(sim_row.get("act_pos_frac", 0)),
        "wind_mean":          float(sim_row.get("wind_mean", 0)),
        "solar_mean":         float(sim_row.get("solar_mean", 0)),
        "hydro_mean":         float(sim_row.get("hydro_mean", 0)),
        "nonrenew_mean":      float(sim_row.get("nonrenew_mean", 0)),
        "external_mean":      float(sim_row.get("external_mean", 0)),
        "load_mean":          float(sim_row.get("load_mean", 0)),
        "transmission_mean":  float(sim_row.get("transmission_mean", 0)),
        "source_impact_sum":  float(sim_row.get("source_impact_sum", 0)),
        "sim_in_universe":    1,
        "sim_predicts_active":int(float(sim_row.get("act_mean", 0)) > 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 6 — COMBINED SIGNALS
# Interaction features — often the most predictive for LightGBM
# ─────────────────────────────────────────────────────────────────────────────

def _combined(feat: dict) -> dict:
    psm        = feat.get("psm_abs_mean", 0)
    pr_12m     = feat.get("pr_avg_12m", 0)
    pr_active  = feat.get("pr_avg_when_active", 0)
    act_rate   = feat.get("pr_nonzero_rate_12m", 0)
    c_avg      = feat.get("c_avg_12m", 0)
    c_ly       = feat.get("c_same_month_ly", 0)
    profitable_ly  = feat.get("profitable_ly", 0)
    profit_12m     = feat.get("profit_rate_12m", 0)
    streak         = feat.get("profit_streak", 0)
    sim_active     = feat.get("sim_predicts_active", 0)
    sim_agree      = feat.get("psm_scenario_agree", 0)

    # ── Simulated profit proxy: PSM as price estimate vs cost estimate ─────
    # This is the closest we can get to |PR_{M+1}| - C_{M+1} at decision time
    # Use conservative cost estimate (same month LY or recent avg)
    c_estimate = c_ly if c_ly > 0 else c_avg
    combined = {
        "sim_profit_est":          psm - c_estimate,
        "sim_profit_est_positive": int(psm > c_estimate),
        "sim_profit_conservative": float(feat.get("psm_abs_min", 0)) - c_estimate,
    }

    # ── PSM vs historical PR ratios ────────────────────────────────────────
    combined["psm_vs_pr_ratio_12m"]    = psm / (pr_12m   + 1e-9)
    combined["psm_vs_pr_ratio_active"] = psm / (pr_active + 1e-9)

    # ── Signal agreement features ─────────────────────────────────────────
    # Both sim AND history agree → strongest confirmation
    combined["sim_and_history_agree"] = int(sim_active == 1 and act_rate > 0.1)
    combined["sim_confirms_seasonal"] = int(profitable_ly == 1 and sim_active == 1)

    # Count of bullish signals (0-5 scale)
    n_bullish = (
        int(profitable_ly == 1) +
        int(profit_12m > 0.5) +
        int(sim_active == 1) +
        int(sim_agree == 1) +
        int(streak >= 2)
    )
    combined["n_bullish_signals"] = n_bullish
    combined["high_confidence"]   = int(n_bullish >= 3)

    # Bearish flag: all signals negative → definitely don't select
    combined["all_signals_negative"] = int(
        profitable_ly == 0 and
        profit_12m < 0.1 and
        sim_active == 0
    )

    # PSM uncertainty: high std across scenarios = unreliable forecast
    psm_std = feat.get("psm_abs_std", 0)
    combined["psm_cv"] = psm_std / (psm + 1e-9)  # coefficient of variation

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GROUP 7 — TEMPORAL / CALENDAR
# ─────────────────────────────────────────────────────────────────────────────

def _temporal(target_month: str) -> dict:
    period    = pd.Period(target_month, freq="M")
    month_num = period.month
    return {
        "month_of_year": month_num,
        "is_winter":     int(month_num in [11, 12, 1, 2, 3]),
        "is_summer":     int(month_num in [6, 7, 8, 9]),
        "is_shoulder":   int(month_num in [4, 5, 10]),
        "quarter":       (month_num - 1) // 3 + 1,
        "month_sin":     float(np.sin(2 * np.pi * month_num / 12)),
        "month_cos":     float(np.cos(2 * np.pi * month_num / 12)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMN LISTS (for model.py to reference)
# ─────────────────────────────────────────────────────────────────────────────

ID_COLS = ["EID", "PEAKID", "TARGET_MONTH"]
LABEL_COLS = ["is_profitable", "profit"]

ALL_FEATURE_COLS = [
    # Seasonal
    "profitable_ly", "profitable_2ly", "profitable_3ly",
    "pr_ly", "profit_ly", "c_ly",
    "seasonal_consistency",
    # Persistence
    "profit_rate_3m", "profit_rate_6m", "profit_rate_12m", "profit_rate_24m",
    "profit_last_month", "profit_streak", "profitable_last_3_all",
    "profit_avg_3m", "profit_avg_12m", "profit_max_12m",
    # PR
    "pr_avg_3m", "pr_avg_12m", "pr_last", "pr_std_12m", "pr_max_12m",
    "pr_nonzero_rate_12m", "pr_nonzero_rate_24m",
    "pr_ever_nonzero", "pr_avg_when_active", "pr_count_active_12m",
    # Cost
    "c_last", "c_avg_3m", "c_avg_12m", "c_same_month_ly", "c_same_month_2ly",
    "c_nonzero_rate_12m", "c_avg_when_nonzero", "c_std_12m",
    "c_trend", "is_zero_cost_hist",
    # Sim
    "psm_abs_mean", "psm_abs_min", "psm_abs_max", "psm_abs_std",
    "psm_scenario_agree", "psm_s1", "psm_s2", "psm_s3",
    "act_mean", "act_max", "act_pos_frac",
    "wind_mean", "solar_mean", "hydro_mean", "nonrenew_mean", "external_mean",
    "load_mean", "transmission_mean", "source_impact_sum",
    "sim_in_universe", "sim_predicts_active",
    # Combined
    "sim_profit_est", "sim_profit_est_positive", "sim_profit_conservative",
    "psm_vs_pr_ratio_12m", "psm_vs_pr_ratio_active",
    "sim_and_history_agree", "sim_confirms_seasonal",
    "n_bullish_signals", "high_confidence", "all_signals_negative", "psm_cv",
    # Temporal
    "month_of_year", "is_winter", "is_summer", "is_shoulder",
    "quarter", "month_sin", "month_cos",
    # Meta
    "n_months_history", "has_any_history",
]


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return only columns that are actual features (not ID or label cols)."""
    exclude = set(ID_COLS + LABEL_COLS + ["MONTH", "PR", "C", "SCENARIOID"])
    return [c for c in df.columns if c not in exclude]