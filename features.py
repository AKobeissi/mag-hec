"""
features.py — Feature engineering for MAG Energy Solutions Challenge.

All features respect the anti-leakage rule:
  For target month M+1, only data available on the 7th of month M is used.

Feature groups:
  1. Cost features     — estimated M+1 cost from history
  2. PR features       — historical realized price patterns
  3. Profitability     — historical profit/loss patterns
  4. Sim monthly       — forward-looking PSM signal (primary signal)
  5. Sim calibration   — how well are sims tracking reality this month
  6. Temporal          — month-of-year seasonality
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

from config import (
    LOOKBACK_SHORT, LOOKBACK_MEDIUM, LOOKBACK_LONG, LOOKBACK_LONG2
)
from data_loader import (
    add_months,
    load_sim_monthly_for_target,
    aggregate_sim_across_scenarios,
)


# ─────────────────────────────────────────────────────────────────────────────
# CORE FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_features_for_month(
    target_month: str,
    truth: pd.DataFrame,
    sim_agg: Optional[pd.DataFrame] = None,
    calibration_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix for all (EID, PEAKID) candidates
    for a given target_month (M+1).

    Parameters
    ----------
    target_month    : str   e.g. '2022-03'  (this is M+1)
    truth           : DataFrame  full ground truth (EID, MONTH, PEAKID,
                                 PR, C, profit, is_profitable)
                      *** Must already be filtered to months < target_month ***
    sim_agg         : DataFrame  aggregated sim_monthly output for target_month
                      (from aggregate_sim_monthly_across_scenarios)
    calibration_df  : DataFrame  sim calibration accuracy for cutoff month

    Returns
    -------
    DataFrame: one row per (EID, PEAKID) with all features + TARGET_MONTH
    """
    cutoff_month = add_months(target_month, -1)   # month M

    # ── Candidate universe ────────────────────────────────────────────────────
    # All (EID, PEAKID) seen in any historical data
    # If sim_agg available, use those as the primary candidates
    if sim_agg is not None and len(sim_agg) > 0:
        candidates = sim_agg[["EID", "PEAKID"]].drop_duplicates().copy()
        print(f"    Candidates from sim_monthly: {len(candidates):,}")
    else:
        candidates = (truth[["EID", "PEAKID"]]
                      .drop_duplicates()
                      .copy())
        print(f"    Candidates from history: {len(candidates):,}")

    # ── Historical features (per EID, PEAKID) ────────────────────────────────
    hist_feats = _build_historical_features(
        candidates, truth, target_month, cutoff_month
    )

    # ── Merge sim_monthly features ────────────────────────────────────────────
    if sim_agg is not None and len(sim_agg) > 0:
        sim_feats = _build_sim_features(sim_agg, hist_feats)
        features = hist_feats.merge(
            sim_feats.drop(columns=["EID","PEAKID"], errors="ignore"),
            left_index=True, right_index=True, how="left"
        )
        # Re-merge properly
        features = hist_feats.merge(sim_agg, on=["EID","PEAKID"], how="left")
        features = _build_sim_features_merged(features, hist_feats)
    else:
        features = hist_feats.copy()

    # ── Calibration features ──────────────────────────────────────────────────
    if calibration_df is not None and len(calibration_df) > 0:
        features = features.merge(
            calibration_df, on=["EID","PEAKID"], how="left"
        )

    # ── Temporal features ─────────────────────────────────────────────────────
    features = _add_temporal_features(features, target_month)

    features["TARGET_MONTH"] = target_month
    features = features.fillna(0)

    return features


def _build_historical_features(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    target_month: str,
    cutoff_month: str,
) -> pd.DataFrame:
    """
    Build cost, PR, and profitability features from historical truth data.
    """
    same_month_ly = add_months(target_month, -12)
    same_month_2y = add_months(target_month, -24)

    rows = []

    for _, row in candidates.iterrows():
        eid    = row["EID"]
        peakid = row["PEAKID"]

        # Filter to this EID/PEAKID history
        g = truth[
            (truth["EID"]    == eid) &
            (truth["PEAKID"] == peakid) &
            (truth["MONTH"]  <= cutoff_month)
        ].sort_values("MONTH")

        feat = {"EID": eid, "PEAKID": peakid}

        if len(g) == 0:
            # New EID — no history, fill zeros
            rows.append(feat)
            continue

        # ── Cost features ─────────────────────────────────────────────────────
        feat["cost_last"]          = g["C"].iloc[-1]
        feat["cost_avg_3m"]        = g["C"].tail(LOOKBACK_SHORT).mean()
        feat["cost_avg_6m"]        = g["C"].tail(LOOKBACK_MEDIUM).mean()
        feat["cost_avg_12m"]       = g["C"].tail(LOOKBACK_LONG).mean()
        feat["cost_std_12m"]       = g["C"].tail(LOOKBACK_LONG).std()
        feat["cost_cv_12m"]        = (feat["cost_std_12m"] /
                                      (feat["cost_avg_12m"] + 1e-9))

        # Same-month last year cost (strong seasonal signal)
        ly = g[g["MONTH"] == same_month_ly]["C"]
        feat["cost_same_month_ly"] = ly.iloc[0] if len(ly) > 0 else feat["cost_avg_12m"]

        ly2 = g[g["MONTH"] == same_month_2y]["C"]
        feat["cost_same_month_2y"] = ly2.iloc[0] if len(ly2) > 0 else feat["cost_avg_12m"]

        # Cost trend (positive = cost rising)
        if len(g) >= 3:
            recent_costs = g["C"].tail(6).values
            feat["cost_trend"] = np.polyfit(
                range(len(recent_costs)), recent_costs, 1
            )[0]
        else:
            feat["cost_trend"] = 0.0

        feat["is_zero_cost_hist"] = int(feat["cost_avg_12m"] == 0)

        # ── PR (realized price) features ──────────────────────────────────────
        feat["pr_avg_3m"]          = g["PR"].tail(LOOKBACK_SHORT).mean()
        feat["pr_avg_6m"]          = g["PR"].tail(LOOKBACK_MEDIUM).mean()
        feat["pr_avg_12m"]         = g["PR"].tail(LOOKBACK_LONG).mean()
        feat["pr_std_12m"]         = g["PR"].tail(LOOKBACK_LONG).std()
        feat["pr_last"]            = g["PR"].iloc[-1]

        ly_pr = g[g["MONTH"] == same_month_ly]["PR"]
        feat["pr_same_month_ly"]   = ly_pr.iloc[0] if len(ly_pr) > 0 else feat["pr_avg_12m"]

        ly2_pr = g[g["MONTH"] == same_month_2y]["PR"]
        feat["pr_same_month_2y"]   = ly2_pr.iloc[0] if len(ly2_pr) > 0 else feat["pr_avg_12m"]

        # ── Historical profit features ────────────────────────────────────────
        feat["profit_avg_3m"]       = g["profit"].tail(LOOKBACK_SHORT).mean()
        feat["profit_avg_12m"]      = g["profit"].tail(LOOKBACK_LONG).mean()
        feat["profit_last"]         = g["profit"].iloc[-1]

        feat["profit_rate_3m"]      = g["is_profitable"].tail(LOOKBACK_SHORT).mean()
        feat["profit_rate_6m"]      = g["is_profitable"].tail(LOOKBACK_MEDIUM).mean()
        feat["profit_rate_12m"]     = g["is_profitable"].tail(LOOKBACK_LONG).mean()
        feat["profit_rate_24m"]     = g["is_profitable"].tail(LOOKBACK_LONG2).mean()
        feat["profit_last_month"]   = g["is_profitable"].iloc[-1]

        # Consecutive profitable months streak
        streak = 0
        for val in reversed(g["is_profitable"].values):
            if val == 1:
                streak += 1
            else:
                break
        feat["profit_streak"]       = streak

        # Ever profitable in last 12m
        feat["ever_profitable_12m"] = int(g["is_profitable"].tail(LOOKBACK_LONG).sum() > 0)

        # Same month last year profitability
        ly_p = g[g["MONTH"] == same_month_ly]["is_profitable"]
        feat["profitable_same_month_ly"] = ly_p.iloc[0] if len(ly_p) > 0 else 0

        ly_profit = g[g["MONTH"] == same_month_ly]["profit"]
        feat["profit_same_month_ly"] = ly_profit.iloc[0] if len(ly_profit) > 0 else 0

        # ── Estimated profit proxy (using history, no sim) ────────────────────
        feat["est_profit_ly"]        = (feat["pr_same_month_ly"] -
                                         feat["cost_same_month_ly"])
        feat["est_profit_3m"]        = feat["pr_avg_3m"] - feat["cost_avg_3m"]
        feat["est_profit_12m"]       = feat["pr_avg_12m"] - feat["cost_avg_12m"]

        feat["n_months_history"]     = len(g)

        rows.append(feat)

    result = pd.DataFrame(rows).fillna(0)
    return result


def _build_sim_features_merged(
    merged: pd.DataFrame,
    hist_feats: pd.DataFrame,
) -> pd.DataFrame:
    """
    After merging sim_agg into historical features, compute derived sim features.
    """
    # Primary signal: simulated profit proxy
    # Use estimated cost from history vs simulated price
    if "psm_abs_mean" in merged.columns:
        merged["sim_profit_mean"]   = merged["psm_abs_mean"] - merged["cost_avg_3m"]
        merged["sim_profit_ly"]     = merged["psm_abs_mean"] - merged["cost_same_month_ly"]
        merged["sim_profit_min"]    = merged.get("psm_abs_min", merged["psm_abs_mean"]) - merged["cost_avg_3m"]
        merged["sim_profit_pos"]    = (merged["sim_profit_mean"] > 0).astype(int)

        # Ratio of simulated to historical price
        merged["psm_to_pr_ratio_3m"] = merged["psm_abs_mean"] / (merged["pr_avg_3m"] + 1e-9)
        merged["psm_to_pr_ratio_ly"] = merged["psm_abs_mean"] / (merged["pr_same_month_ly"] + 1e-9)

        # Uncertainty across scenarios
        if "psm_abs_std" in merged.columns:
            merged["psm_uncertainty"] = merged["psm_abs_std"] / (merged["psm_abs_mean"] + 1e-9)

    return merged


def _build_sim_features(
    sim_agg: pd.DataFrame,
    hist_feats: pd.DataFrame,
) -> pd.DataFrame:
    """Derive sim-based features as a standalone step."""
    # Placeholder — actual work done in _build_sim_features_merged
    return sim_agg


def _add_temporal_features(
    features: pd.DataFrame,
    target_month: str,
) -> pd.DataFrame:
    """Add calendar-based features for seasonality."""
    period = pd.Period(target_month, freq="M")
    month_num = period.month

    features["month_of_year"]  = month_num
    features["is_winter"]      = int(month_num in [11, 12, 1, 2, 3])
    features["is_summer"]      = int(month_num in [6, 7, 8, 9])
    features["is_shoulder"]    = int(month_num in [4, 5, 10])
    features["quarter"]        = (month_num - 1) // 3 + 1

    # Cyclical encoding (better than raw month number for ML)
    features["month_sin"]      = np.sin(2 * np.pi * month_num / 12)
    features["month_cos"]      = np.cos(2 * np.pi * month_num / 12)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE MATRIX for multiple months (full walk-forward)
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    truth: pd.DataFrame,
    target_months: list,
    sim_loader_fn=None,
) -> pd.DataFrame:
    """
    Build feature matrix for a list of target months.
    Respects walk-forward: each month only sees past data.

    Parameters
    ----------
    truth          : full ground truth DataFrame (all months)
    target_months  : list of YYYY-MM strings to build features for
    sim_loader_fn  : optional callable(target_month) → sim_agg DataFrame
                     (pass None to skip sim features)

    Returns
    -------
    DataFrame: all features stacked, with TARGET_MONTH column
    """
    all_features = []

    for i, target_month in enumerate(target_months):
        cutoff_month = add_months(target_month, -1)

        print(f"\n  [{i+1}/{len(target_months)}] Building features for "
              f"{target_month} (cutoff: {cutoff_month})")

        # ANTI-LEAKAGE: only use truth up to cutoff month
        truth_past = truth[truth["MONTH"] < target_month].copy()

        if len(truth_past) == 0:
            print(f"    SKIP: no historical data before {target_month}")
            continue

        # Load sim_monthly if loader provided
        sim_agg = None
        if sim_loader_fn is not None:
            try:
                sim_raw = load_sim_monthly_for_target(target_month)
                if sim_raw is not None:
                    sim_agg = aggregate_sim_across_scenarios(sim_raw)
            except Exception as e:
                print(f"    WARNING: sim loading failed: {e}")

        feats = build_features_for_month(
            target_month=target_month,
            truth=truth_past,
            sim_agg=sim_agg,
        )
        all_features.append(feats)

    if not all_features:
        raise ValueError("No features were built. Check date ranges.")

    result = pd.concat(all_features, ignore_index=True)
    print(f"\n  ✓ Feature matrix: {result.shape} "
          f"({result['TARGET_MONTH'].nunique()} months)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature columns (exclude ID and target columns)."""
    exclude = {
        "EID", "PEAKID", "TARGET_MONTH", "MONTH",
        "PR", "C", "profit", "is_profitable",
        "SCENARIOID", "n_hours",
    }
    return [c for c in df.columns if c not in exclude]


def get_id_columns() -> list:
    return ["EID", "PEAKID", "TARGET_MONTH"]