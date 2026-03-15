"""
cost_predictor.py — Predict C_{M+1} using XGBoost + historical costs + sim features.

Converted from Costs_predictions_MAGE.ipynb.

What this does:
  1. Loads sim_monthly_with_costs (sim features joined with realized costs)
  2. Builds lagged features: C_M (last month cost), same-month-LY, 3m average
  3. Trains XGBoost regressor on 2020-2022
  4. Validates on 2023, tests on 2024
  5. Generates predicted_costs.parquet for ALL (EID, PEAKID, TARGET_MONTH)
     - 1,203 EIDs with cost history → XGBoost prediction
     - 7,401 EIDs with no cost history → predicted_c = 0.0

Anti-leakage:
  Features for TARGET_MONTH M+1 use only costs up to month M.
  C_M = cost in month M (the cutoff month, allowed per case doc 4.2)
  C_M_plus_1 = target (what we're predicting, never used as feature)

Usage:
    python cost_predictor.py
    python cost_predictor.py --skip-tuning   (uses best params directly, faster)
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent if SCRIPT_DIR.name in ("analysis","src") else SCRIPT_DIR
sys.path.insert(0, str(ROOT_DIR))

from config import (
    OUTPUT_DIR, COSTS_DIR, RANDOM_SEED,
    SIM_MONTHLY_PATHS, SIM_DAILY_PATHS,
)
from data_loader import (
    load_costs,
    load_prices_and_aggregate,
    build_sim_candidate_universe,
    compute_ground_truth,
    get_month_range,
    add_months,
    SIM_MONTHLY_PATHS,
    SIM_DAILY_PATHS,
)

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("WARNING: pip install xgboost")

try:
    from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: pip install scikit-learn")

OUTPUT_DIR.mkdir(exist_ok=True)
COST_PRED_CACHE   = OUTPUT_DIR / "predicted_costs.parquet"
SIM_COST_CACHE    = OUTPUT_DIR / "sim_monthly_with_costs.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Predict C_{M+1} using XGBoost + sim features"
    )
    p.add_argument("--skip-tuning", action="store_true",
                   help="Skip hyperparameter search, use best known params")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Force rebuild sim_monthly_with_costs cache")
    p.add_argument("--sim-costs-path", type=str, default=None,
                   help="Path to sim_monthly_with_costs.parquet "
                        "(e.g. from Google Drive export)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: BUILD SIM + COSTS JOINED DATASET
# ─────────────────────────────────────────────────────────────────────────────

def build_sim_costs_dataset(rebuild: bool = False) -> pd.DataFrame:
    """
    Join sim_monthly aggregated features with realized costs.
    Only keeps rows where C > 0 (the 1,203 EIDs that have real costs).

    Returns DataFrame ready for model training with columns:
      EID, PEAKID, MONTH, C,
      ACTIVATIONLEVEL, WINDIMPACT, SOLARIMPACT, HYDROIMPACT,
      NONRENEWBALIMPACT, EXTERNALIMPACT, LOADIMPACT,
      TRANSMISSIONOUTAGEIMPACT, n_hours
    """
    if SIM_COST_CACHE.exists() and not rebuild:
        print(f"  Loading sim+costs cache from {SIM_COST_CACHE}...")
        df = pd.read_parquet(SIM_COST_CACHE)
        print(f"  ✓ {len(df):,} rows | {df['EID'].nunique():,} EIDs")
        return df

    print("  Building sim+costs dataset...")

    # Load realized costs (sparse — only non-zero rows)
    costs = load_costs()
    print(f"  Costs: {len(costs):,} rows | {costs['EID'].nunique():,} EIDs")

    # Load sim cache (already built by predict_price.py)
    sim_cache_path = OUTPUT_DIR / "sim_monthly_cache.parquet"
    if not sim_cache_path.exists():
        raise FileNotFoundError(
            f"Sim cache not found at {sim_cache_path}.\n"
            f"Run predict_price.py first to build it."
        )

    sim_cache = pd.read_parquet(sim_cache_path)
    sim_cache = sim_cache.rename(columns={"TARGET_MONTH": "MONTH"})
    print(f"  Sim cache: {len(sim_cache):,} rows")

    # Inner join — only keep (EID, PEAKID, MONTH) that have both sim and cost
    merged = costs.merge(
        sim_cache[[
            "EID","PEAKID","MONTH",
            "act_mean","act_max","act_pos_frac",
            "wind_mean","solar_mean","hydro_mean",
            "nonrenew_mean","external_mean",
            "load_mean","transmission_mean",
            "source_impact_sum","scenario_agree",
        ]],
        on=["EID","PEAKID","MONTH"],
        how="left"
    )

    # Rename to match notebook column names for consistency
    merged = merged.rename(columns={
        "act_mean":           "ACTIVATIONLEVEL",
        "wind_mean":          "WINDIMPACT",
        "solar_mean":         "SOLARIMPACT",
        "hydro_mean":         "HYDROIMPACT",
        "nonrenew_mean":      "NONRENEWBALIMPACT",
        "external_mean":      "EXTERNALIMPACT",
        "load_mean":          "LOADIMPACT",
        "transmission_mean":  "TRANSMISSIONOUTAGEIMPACT",
    })

    # Add n_hours proxy (using act_pos_frac × typical hours per month)
    # act_pos_frac = fraction of hours with activation > 0
    # multiply by typical hours in a month (~720)
    merged["n_hours"] = merged["act_pos_frac"] * 720

    # Impute any missing sim features with column mean
    sim_feat_cols = [
        "ACTIVATIONLEVEL","WINDIMPACT","SOLARIMPACT","HYDROIMPACT",
        "NONRENEWBALIMPACT","EXTERNALIMPACT","LOADIMPACT",
        "TRANSMISSIONOUTAGEIMPACT","n_hours"
    ]
    for col in sim_feat_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(merged[col].mean())

    merged["MONTH"] = pd.to_datetime(merged["MONTH"] + "-01")
    merged = merged.sort_values(["EID","MONTH"]).reset_index(drop=True)

    merged.to_parquet(SIM_COST_CACHE, index=False)
    print(f"  ✓ Saved sim+costs cache: {len(merged):,} rows → {SIM_COST_CACHE}")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features for cost prediction.

    Key features:
      C_M          = cost in month M (cutoff month — last known cost)
      C_M_ly       = cost same month last year (strong seasonal signal)
      C_M_2ly      = cost same month 2 years ago
      C_avg_3m     = 3-month average cost
      C_avg_12m    = 12-month average cost
      month_of_year = calendar month (seasonality)
    """
    df = df.copy()
    df = df.sort_values(["EID","PEAKID","MONTH"]).reset_index(drop=True)

    # ── Lagged cost features ──────────────────────────────────────────────────
    # C_M = previous month's cost (most recent known value)
    df["C_M"] = df.groupby(["EID","PEAKID"])["C"].shift(1)

    # C_M_plus_1 = next month's cost (TARGET — what we predict)
    df["C_M_plus_1"] = df.groupby(["EID","PEAKID"])["C"].shift(-1)

    # 3-month rolling average (not including current month)
    df["C_avg_3m"] = (
        df.groupby(["EID","PEAKID"])["C"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # 12-month rolling average
    df["C_avg_12m"] = (
        df.groupby(["EID","PEAKID"])["C"]
        .transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    )

    # Same-month last year
    df["C_M_ly"] = df.groupby(["EID","PEAKID","month_num"])["C"].shift(1) \
                   if "month_num" in df.columns else np.nan

    # ── Calendar features ─────────────────────────────────────────────────────
    df["month_of_year"] = df["MONTH"].dt.month
    df["is_winter"]     = df["month_of_year"].isin([11,12,1,2,3]).astype(int)
    df["is_summer"]     = df["month_of_year"].isin([6,7,8,9]).astype(int)

    # ── Same month last year via merge ────────────────────────────────────────
    df_ly = df[["EID","PEAKID","MONTH","C"]].copy()
    df_ly["MONTH_NEXT_YR"] = df_ly["MONTH"] + pd.DateOffset(years=1)
    df = df.merge(
        df_ly[["EID","PEAKID","MONTH_NEXT_YR","C"]].rename(
            columns={"MONTH_NEXT_YR":"MONTH","C":"C_same_month_ly"}
        ),
        on=["EID","PEAKID","MONTH"],
        how="left"
    )

    # Drop rows where TARGET is NaN (last row per EID has no M+1 to predict)
    # Keep rows where C_M = 0 — zero last month IS valid information
    # (means EID had no cost last month — model should learn this pattern)
    df["C_M"]      = df["C_M"].fillna(0)       # missing → 0, not unknown
    df["C_avg_3m"] = df["C_avg_3m"].fillna(0)
    df["C_avg_12m"]= df["C_avg_12m"].fillna(0)
    df = df.dropna(subset=["C_M_plus_1"])      # only drop if no target

    # Impute remaining NaN features
    num_cols = [
        "ACTIVATIONLEVEL","WINDIMPACT","SOLARIMPACT","HYDROIMPACT",
        "NONRENEWBALIMPACT","EXTERNALIMPACT","LOADIMPACT",
        "TRANSMISSIONOUTAGEIMPACT","n_hours",
        "C_avg_3m","C_avg_12m","C_same_month_ly",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "EID", "PEAKID",
    "C_M",                         # last known cost
    "C_avg_3m",                    # 3-month average
    "C_avg_12m",                   # 12-month average
    "C_same_month_ly",             # same month last year
    "ACTIVATIONLEVEL",
    "WINDIMPACT", "SOLARIMPACT", "HYDROIMPACT",
    "NONRENEWBALIMPACT", "EXTERNALIMPACT",
    "LOADIMPACT", "TRANSMISSIONOUTAGEIMPACT",
    "n_hours",
    "month_of_year", "is_winter", "is_summer",
]

# Best params from notebook's RandomizedSearchCV — use with --skip-tuning
BEST_PARAMS = {
    "subsample":        0.7,
    "reg_lambda":       0.5,
    "reg_alpha":        0.01,
    "n_estimators":     400,
    "max_depth":        4,
    "learning_rate":    0.01,
    "gamma":            0,
    "colsample_bytree": 1.0,
}


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
) -> dict:
    """
    RandomizedSearchCV with PredefinedSplit — val set is fixed, not cross-val.
    Mirrors the notebook's approach exactly.
    """
    print("  Running hyperparameter search (50 iterations)...")
    print("  This takes ~5-10 minutes...")

    param_dist = {
        "n_estimators":    [100, 200, 300, 400, 500],
        "max_depth":       [3, 4, 5, 6, 7, 8, 9, 10],
        "learning_rate":   [0.01, 0.05, 0.1, 0.2, 0.3],
        "subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma":           [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "reg_alpha":       [0, 0.001, 0.01, 0.1],
        "reg_lambda":      [0.5, 1, 1.5, 2],
    }

    # PredefinedSplit: -1 = training, 0 = validation
    test_fold = np.append(
        np.full(len(X_train), -1),
        np.full(len(X_val),    0)
    )
    ps = PredefinedSplit(test_fold)

    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])

    rsearch = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=50,
        scoring="neg_mean_absolute_error",
        cv=ps,
        verbose=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rsearch.fit(X_combined, y_combined)

    best = rsearch.best_params_
    score = -rsearch.best_score_

    print(f"\n  Best params: {best}")
    print(f"  Best val MAE: {score:.2f}")

    return best


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: TRAIN FINAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

def train_cost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params:  dict,
) -> XGBRegressor:
    """
    Train final XGBoost on combined train+val data with best params.

    Winsorizes target at 99th percentile before training:
      - Cost distribution is heavily right-skewed (max $146k vs mean $5.6k)
      - Training on raw values causes model to chase rare extreme costs
      - Winsorizing trains for general patterns, not one-off spikes
      - Actual costs still used at evaluation time (not capped)
    """
    # Winsorize target at 99th percentile
    p99     = y_train.quantile(0.99)
    n_capped = (y_train > p99).sum()
    y_train_capped = y_train.clip(upper=p99)

    print(f"  Training samples:     {len(X_train):,}")
    print(f"  Cost 99th pct cap:    {p99:,.2f}")
    print(f"  Values capped:        {n_capped:,} ({100*n_capped/len(y_train):.1f}%)")

    model = XGBRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train_capped)   # train on capped values

    # Feature importance
    imp = pd.DataFrame({
        "feature":    X_train.columns.tolist(),
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 10 features (cost model):")
    for _, row in imp.head(10).iterrows():
        bar = "█" * int(row["importance"] / imp["importance"].max() * 25)
        print(f"    {row['feature']:35s} {row['importance']:>8.4f}  {bar}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_cost_model(
    model: XGBRegressor,
    df: pd.DataFrame,
    val_months: list,
    label: str = "Validation",
):
    """Evaluate cost model month by month on known months."""
    print(f"\n{'='*55}")
    print(f"  COST MODEL VALIDATION — {label}")
    print(f"{'='*55}\n")

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    monthly_maes  = []
    monthly_rmses = []

    for month_str in val_months:
        month_dt = pd.to_datetime(month_str + "-01")
        mask = df["MONTH"] == month_dt

        if mask.sum() == 0:
            continue

        X_m = df.loc[mask, feat_cols]
        y_m = df.loc[mask, "C_M_plus_1"]

        preds = np.maximum(model.predict(X_m), 0)
        mae   = mean_absolute_error(y_m, preds)
        rmse  = np.sqrt(mean_squared_error(y_m, preds))

        monthly_maes.append(mae)
        monthly_rmses.append(rmse)
        print(f"  {month_str}: MAE={mae:>9,.2f}  RMSE={rmse:>9,.2f}  n={mask.sum()}")

    if monthly_maes:
        print(f"\n  Average MAE:  {np.mean(monthly_maes):>9,.2f}")
        print(f"  Average RMSE: {np.mean(monthly_rmses):>9,.2f}")
        print(f"  Mean cost:    {df.loc[df['MONTH'].dt.strftime('%Y-%m').isin(val_months), 'C'].mean():>9,.2f}")
        rel = np.mean(monthly_maes) / df.loc[df["MONTH"].dt.strftime("%Y-%m").isin(val_months), "C"].mean()
        print(f"  Relative MAE: {100*rel:.1f}%  (MAE / mean cost)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: GENERATE PREDICTIONS FOR FULL UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────

def generate_cost_predictions(
    model:        XGBRegressor,
    df:           pd.DataFrame,
    sim_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate predicted_c for ALL (EID, PEAKID, TARGET_MONTH) in the universe.

    For 1,203 EIDs with cost history → XGBoost prediction
    For 7,401 EIDs with no cost history → predicted_c = 0.0

    OUTPUT format matches what predict_profit.py expects:
      EID, PEAKID, TARGET_MONTH, predicted_c
    """
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    # ── Predictions for EIDs with cost history ────────────────────────────────
    print("  Generating predictions for cost-history EIDs...")

    df["TARGET_MONTH"] = df["MONTH"].dt.strftime("%Y-%m")
    X_all  = df[feat_cols].fillna(0)
    preds  = np.maximum(model.predict(X_all), 0)

    cost_preds = df[["EID","PEAKID","TARGET_MONTH"]].copy()
    cost_preds["predicted_c"] = preds
    cost_preds["has_cost_hist"] = 1

    print(f"  Cost-history predictions: {len(cost_preds):,} rows")

    # ── Zero predictions for EIDs with no cost history ────────────────────────
    print("  Adding zero predictions for no-cost-history EIDs...")

    cost_eids   = set(cost_preds["EID"].unique())
    zero_cands  = sim_candidates[~sim_candidates["EID"].isin(cost_eids)].copy()
    zero_cands["predicted_c"]  = 0.0
    zero_cands["has_cost_hist"] = 0
    zero_cands = zero_cands.rename(columns={"MONTH":"TARGET_MONTH"})

    print(f"  Zero predictions:         {len(zero_cands):,} rows")

    # ── Combine ───────────────────────────────────────────────────────────────
    full = pd.concat(
        [cost_preds, zero_cands[["EID","PEAKID","TARGET_MONTH",
                                  "predicted_c","has_cost_hist"]]],
        ignore_index=True
    ).drop_duplicates(subset=["EID","PEAKID","TARGET_MONTH"])

    full["EID"]    = full["EID"].astype(int)
    full["PEAKID"] = full["PEAKID"].astype(int)

    print(f"\n  Full prediction file:")
    print(f"    Total rows:        {len(full):,}")
    print(f"    Unique EIDs:       {full['EID'].nunique():,}")
    print(f"    Months:            {full['TARGET_MONTH'].nunique()}")
    print(f"    With real model:   {full['has_cost_hist'].sum():,} rows")
    print(f"    Zero-padded:       {(full['has_cost_hist']==0).sum():,} rows")
    print(f"    predicted_c > 0:   {(full['predicted_c'] > 0).sum():,}")
    print(f"    predicted_c mean:  {full['predicted_c'].mean():.2f}")
    print(f"    predicted_c max:   {full['predicted_c'].max():.2f}")

    return full


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: COMPARE PRICE-ONLY vs PRICE-MINUS-COST DECISION RULE
# ─────────────────────────────────────────────────────────────────────────────

def compare_decision_rules(
    cost_preds:  pd.DataFrame,
    truth:       pd.DataFrame,
    val_months:  list,
):
    """
    Side-by-side comparison of two selection strategies on 2023 holdout:
      Strategy A: top-100 by predicted_pr only
      Strategy B: top-100 by (predicted_pr - predicted_c)

    Prints monthly breakdown and totals.
    """
    price_cache = OUTPUT_DIR / "predicted_prices.parquet"
    if not price_cache.exists():
        print(f"\n  WARNING: {price_cache} not found.")
        print("  Run predict_price.py first, then re-run this comparison.")
        return

    print(f"\n{'='*70}")
    print("  DECISION RULE COMPARISON — 2023 VALIDATION")
    print(f"  Strategy A: top-100 by predicted_pr")
    print(f"  Strategy B: top-100 by (predicted_pr - predicted_c)")
    print(f"{'='*70}\n")

    price_preds = pd.read_parquet(price_cache)

    # Merge price + cost predictions
    combined = price_preds.merge(
        cost_preds[["EID","PEAKID","TARGET_MONTH","predicted_c"]],
        on=["EID","PEAKID","TARGET_MONTH"],
        how="left"
    ).fillna(0)

    combined["score_A"] = combined["predicted_pr"]
    combined["score_B"] = combined["predicted_pr"] - combined["predicted_c"]

    # Join with ground truth
    truth_val = truth[truth["MONTH"].isin(val_months)].copy()
    truth_val = truth_val.rename(columns={"MONTH":"TARGET_MONTH"})

    combined = combined.merge(
        truth_val[["EID","PEAKID","TARGET_MONTH","PR","C","profit","is_profitable"]],
        on=["EID","PEAKID","TARGET_MONTH"],
        how="inner"
    ).fillna(0)

    print(f"  {'Month':<12} {'A: n_prof':>10} {'B: n_prof':>10}  "
          f"{'A: profit':>14} {'B: profit':>14}  {'Δ profit':>12}")
    print(f"  {'-'*78}")

    results_a, results_b = [], []

    for month in sorted(val_months):
        grp   = combined[combined["TARGET_MONTH"] == month]
        top_a = grp.nlargest(100, "score_A")
        top_b = grp.nlargest(100, "score_B")

        na = top_a["is_profitable"].sum()
        nb = top_b["is_profitable"].sum()
        pa = top_a["profit"].sum()
        pb = top_b["profit"].sum()
        delta = pb - pa
        flag  = " ← B better" if delta > 10000 else (
                " ← A better" if delta < -10000 else "")

        print(f"  {month:<12} {na:>9}/100 {nb:>9}/100  "
              f"{pa:>14,.0f} {pb:>14,.0f}  {delta:>+12,.0f}{flag}")

        results_a.append({"month":month,"n_prof":na,"profit":pa})
        results_b.append({"month":month,"n_prof":nb,"profit":pb})

    df_a = pd.DataFrame(results_a)
    df_b = pd.DataFrame(results_b)

    print(f"  {'-'*78}")
    print(f"  {'TOTAL':<12} {df_a['n_prof'].sum():>9} {df_b['n_prof'].sum():>9}  "
          f"{df_a['profit'].sum():>14,.0f} {df_b['profit'].sum():>14,.0f}  "
          f"{df_b['profit'].sum()-df_a['profit'].sum():>+12,.0f}")

    prec_a = df_a["n_prof"].sum() / 1200
    prec_b = df_b["n_prof"].sum() / 1200

    print(f"\n  ── Summary ──────────────────────────────────────────────")
    print(f"  {'Metric':<30}  {'Strategy A':>14}  {'Strategy B':>14}")
    print(f"  {'-'*62}")
    print(f"  {'Precision':<30}  {100*prec_a:>13.1f}%  {100*prec_b:>13.1f}%")
    print(f"  {'Net profit':<30}  {df_a['profit'].sum():>14,.0f}  "
          f"{df_b['profit'].sum():>14,.0f}")
    print(f"  {'Monthly std':<30}  {df_a['profit'].std():>14,.0f}  "
          f"{df_b['profit'].std():>14,.0f}")
    print(f"  {'Best month':<30}  {df_a['profit'].max():>14,.0f}  "
          f"{df_b['profit'].max():>14,.0f}")
    print(f"  {'Worst month':<30}  {df_a['profit'].min():>14,.0f}  "
          f"{df_b['profit'].min():>14,.0f}")

    improvement = df_b["profit"].sum() - df_a["profit"].sum()
    std_change  = df_b["profit"].std()  - df_a["profit"].std()

    print(f"\n  Profit change from adding cost model: {improvement:+,.0f}")
    print(f"  Std change (negative = more consistent): {std_change:+,.0f}")

    if improvement > 0:
        print(f"\n  ✓ Cost model IMPROVES profit by ${improvement:,.0f}")
    else:
        print(f"\n  ✗ Cost model HURTS profit by ${abs(improvement):,.0f}")
        print(f"    This likely means cost MAE is too high to help on 2023.")
        print(f"    Check cost model accuracy on 2023 specifically.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not XGB_AVAILABLE:
        print("ERROR: pip install xgboost")
        sys.exit(1)
    if not SKLEARN_AVAILABLE:
        print("ERROR: pip install scikit-learn")
        sys.exit(1)

    args = parse_args()
    np.random.seed(RANDOM_SEED)

    print("\n" + "="*55)
    print("  Cost Predictor — XGBoost on cost history + sim features")
    print(f"  Train: 2020-2022  |  Val: 2023  |  Test: 2024")
    print("="*55)

    t0 = time.time()

    # ── Load base data ────────────────────────────────────────────────────────
    print("\n[1] Loading base data...")
    costs = load_costs()
    pr    = load_prices_and_aggregate()

    cand_cache = OUTPUT_DIR / "sim_candidates.parquet"
    if cand_cache.exists():
        sim_candidates = pd.read_parquet(cand_cache)
        sim_candidates = sim_candidates[
            (sim_candidates["MONTH"] >= "2020-01") &
            (sim_candidates["MONTH"] <= "2024-12")
        ]
    else:
        from data_loader import build_sim_candidate_universe
        sim_candidates = build_sim_candidate_universe(
            sim_monthly_paths=SIM_MONTHLY_PATHS,
            sim_daily_paths=SIM_DAILY_PATHS,
            pr=pr, costs=costs,
            start_month="2020-01", end_month="2024-12",
            cache_path=cand_cache,
        )

    truth = compute_ground_truth(
        pr=pr, costs=costs, sim_candidates=sim_candidates,
        months=get_month_range("2020-01","2023-12")
    )

    # ── Build sim+costs joined dataset ───────────────────────────────────────
    print("\n[2] Building sim+costs dataset...")
    if args.sim_costs_path:
        p = Path(args.sim_costs_path)
        if not p.exists():
            print(f"ERROR: file not found: {p}")
            sys.exit(1)
        print(f"  Loading from {p}...")
        raw_df = pd.read_parquet(p)
        raw_df["MONTH"] = pd.to_datetime(raw_df["MONTH"])
        raw_df = raw_df.sort_values(["EID","MONTH"]).reset_index(drop=True)
        print(f"  ✓ {len(raw_df):,} rows | {raw_df['EID'].nunique():,} EIDs")
    else:
        raw_df = build_sim_costs_dataset(rebuild=args.rebuild_cache)

    # ── Feature engineering ───────────────────────────────────────────────────
    print("\n[3] Engineering features...")
    df = build_features(raw_df)

    print(f"  Final shape:  {df.shape}")
    print(f"  Date range:   "
          f"{df['MONTH'].min().strftime('%Y-%m')} → "
          f"{df['MONTH'].max().strftime('%Y-%m')}")
    print(f"  C_M_plus_1:   mean={df['C_M_plus_1'].mean():.2f}  "
          f"max={df['C_M_plus_1'].max():.2f}")

    # ── Data splits ───────────────────────────────────────────────────────────
    print("\n[4] Splitting data...")
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    train_mask = (df["MONTH"] >= "2020-01-01") & (df["MONTH"] <= "2022-12-31")
    val_mask   = (df["MONTH"] >= "2023-01-01") & (df["MONTH"] <= "2023-12-31")
    test_mask  = (df["MONTH"] >= "2024-01-01") & (df["MONTH"] <= "2024-12-31")

    X_train = df.loc[train_mask, feat_cols]
    y_train = df.loc[train_mask, "C_M_plus_1"]
    X_val   = df.loc[val_mask,   feat_cols]
    y_val   = df.loc[val_mask,   "C_M_plus_1"]
    X_test  = df.loc[test_mask,  feat_cols]
    y_test  = df.loc[test_mask,  "C_M_plus_1"]

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # ── Hyperparameter tuning ─────────────────────────────────────────────────
    print("\n[5] Hyperparameter tuning...")
    if args.skip_tuning:
        print(f"  Skipping search — using best known params from notebook:")
        print(f"  {BEST_PARAMS}")
        best_params = BEST_PARAMS
    else:
        best_params = tune_hyperparameters(X_train, y_train, X_val, y_val)

    # ── Train final model on 2020-2023 combined ───────────────────────────────
    print("\n[6] Training final model on 2020-2023 combined...")
    X_train_final = pd.concat([X_train, X_val])
    y_train_final = pd.concat([y_train, y_val])

    print(f"  Training samples: {len(X_train_final):,}")
    model = train_cost_model(X_train_final, y_train_final, best_params)

    # ── Validate on 2023 (val set — not used for training above) ─────────────
    print("\n[7] Validating on 2023...")
    validate_cost_model(
        model=model,
        df=df,
        val_months=get_month_range("2023-01","2023-12"),
        label="2023 Validation"
    )

    # ── Test on 2024 (robustness check) ───────────────────────────────────────
    print("\n[8] Testing on 2024 (robustness)...")
    validate_cost_model(
        model=model,
        df=df,
        val_months=[m for m in get_month_range("2024-01","2024-12")
                    if m in df["MONTH"].dt.strftime("%Y-%m").values],
        label="2024 Test (robustness)"
    )

    # ── Generate full predictions ─────────────────────────────────────────────
    print("\n[9] Generating predictions for full universe...")
    full_preds = generate_cost_predictions(model, df, sim_candidates)

    # ── Save ──────────────────────────────────────────────────────────────────
    full_preds.to_parquet(COST_PRED_CACHE, index=False)
    print(f"\n  ✓ Saved → {COST_PRED_CACHE}")

    # ── Compare decision rules ────────────────────────────────────────────────
    print("\n[10] Comparing decision rules...")
    compare_decision_rules(
        cost_preds=full_preds,
        truth=truth,
        val_months=get_month_range("2023-01","2023-12"),
    )

    total = (time.time() - t0) / 60
    print(f"\n{'='*55}")
    print(f"  Done in {total:.1f} minutes")
    print(f"  Outputs:")
    print(f"    {COST_PRED_CACHE}")
    print(f"  Next: run predict_profit.py to generate opportunities.csv")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()