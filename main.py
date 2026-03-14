"""
main.py — Entry point for MAG Energy Solutions Data Challenge.

Usage:
    python main.py --start-month 2020-01 --end-month 2023-12
    python main.py --start-month 2024-01 --end-month 2024-12
    python main.py --start-month 2020-01 --end-month 2023-12 --validate
    python main.py --start-month 2020-01 --end-month 2023-12 --no-ml
"""

import argparse
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    COSTS_DIR, PRICES_DIR, SIM_MONTHLY_DIR,
    DEFAULT_SELECT, MIN_SELECT_PER_MONTH, MAX_SELECT_PER_MONTH,
    OPPORTUNITIES_PATH, OUTPUT_DIR, RANDOM_SEED,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END,
)
from data_loader import (
    load_costs,
    load_prices_and_aggregate,
    compute_ground_truth,
    load_sim_monthly_for_target,
    aggregate_sim_across_scenarios,
    precompute_sim_monthly_all,
    get_month_range,
    add_months,
    summarize_dataframe,
    SIM_MONTHLY_PATHS,
)
from features import (
    build_features_for_month,
    get_feature_columns,
)
from model import (
    ProfitabilityModel,
    baseline_score,
    sim_score,
    select_opportunities,
    evaluate_selections,
    print_evaluation,
    walk_forward_validate,
    tune_k,
    _get_labels,
    LGBM_AVAILABLE,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENTS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="MAG Energy Solutions — FTR Opportunity Selection"
    )
    parser.add_argument("--start-month", required=True,
                        help="Start of target period (YYYY-MM)")
    parser.add_argument("--end-month", required=True,
                        help="End of target period (YYYY-MM)")
    parser.add_argument("--n-select", type=int, default=DEFAULT_SELECT,
                        help=f"Selections per month (default: {DEFAULT_SELECT})")
    parser.add_argument("--validate", action="store_true",
                        help="Run walk-forward validation and print metrics")
    parser.add_argument("--no-ml", action="store_true",
                        help="Use heuristic scoring only (no LightGBM)")
    parser.add_argument("--tune-k", action="store_true",
                        help="Tune number of selections on validation set")
    parser.add_argument("--use-sim", action="store_true",
                        help="Load sim_monthly data via DuckDB (slower, needs duckdb)")
    parser.add_argument("--output", type=str, default=str(OPPORTUNITIES_PATH),
                        help="Output CSV path")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    t0     = time.time()
    np.random.seed(RANDOM_SEED)

    print("\n" + "="*65)
    print("  MAG Energy Solutions — FTR Portfolio Selection")
    print(f"  Period:     {args.start_month} → {args.end_month}")
    print(f"  N select:   {args.n_select} per month")
    print(f"  ML:         {'disabled' if args.no_ml else 'enabled'}")
    print(f"  Sim data:   {'enabled' if args.use_sim else 'disabled (use --use-sim to enable)'}")
    print("="*65 + "\n")

    use_ml  = not args.no_ml and LGBM_AVAILABLE
    use_sim = args.use_sim

    if args.no_ml and not LGBM_AVAILABLE:
        print("INFO: LightGBM not available, using heuristic scoring.")

    # ── 1. LOAD BASE DATA ────────────────────────────────────────────────────
    print("\n[1/5] Loading costs and prices...")

    costs = load_costs(COSTS_DIR)
    summarize_dataframe(costs, "Costs")

    pr_monthly = load_prices_and_aggregate(PRICES_DIR)
    summarize_dataframe(pr_monthly, "Monthly PR")

    # ── 2. COMPUTE GROUND TRUTH ──────────────────────────────────────────────
    print("\n[2/5] Computing ground truth...")

    truth = compute_ground_truth(pr_monthly, costs)
    summarize_dataframe(truth, "Ground Truth")

    # Print profitability by peak type
    print("\n  Profitability by PEAKID:")
    print(truth.groupby("PEAKID")["is_profitable"].agg(["sum","mean","count"]))

    # ── 3. WALK-FORWARD VALIDATION (optional) ────────────────────────────────
    if args.validate:
        print("\n[3/5] Running walk-forward validation (2020-2022 train → 2023 val)...")
        _run_validation(truth, use_ml, use_sim, args.n_select)
    else:
        print("\n[3/5] Skipping validation (pass --validate to enable)")

    # ── 4. GENERATE SELECTIONS ───────────────────────────────────────────────
    print(f"\n[4/5] Generating selections for {args.start_month} → {args.end_month}...")

    target_months = get_month_range(args.start_month, args.end_month)
    all_selections = []
    n_select  = args.n_select

    # ── Pre-compute sim_monthly features for all target months (once) ─────────
    sim_cache = {}
    if use_sim and SIM_MONTHLY_PATHS:
        print(f"\n  Pre-computing sim_monthly features for "
              f"{len(target_months)} months...")
        cache_path = OUTPUT_DIR / "sim_monthly_cache.parquet"
        sim_all = precompute_sim_monthly_all(
            target_months,
            cache_path=cache_path,
        )
        # Index by TARGET_MONTH for fast lookup
        for tm, grp in sim_all.groupby("TARGET_MONTH"):
            sim_cache[tm] = grp.drop(columns=["TARGET_MONTH"])
        print(f"  ✓ Sim features cached for {len(sim_cache)} months")

    for i, target_month in enumerate(target_months):
        cutoff_month = add_months(target_month, -1)
        print(f"\n  [{i+1}/{len(target_months)}] Selecting for {target_month} "
              f"(cutoff: day 7 of {cutoff_month})")

        # ── Features for this target month ────────────────────────────────────
        truth_past = truth[truth["MONTH"] < target_month].copy()

        if len(truth_past) == 0:
            print(f"    SKIP: no history before {target_month}")
            continue

        # ── Load sim features for this month (from cache) ────────────────────
        sim_agg = sim_cache.get(target_month, None)
        if use_sim and sim_agg is None:
            print(f"    No sim features in cache for {target_month}")

        feats = build_features_for_month(
            target_month=target_month,
            truth=truth_past,
            sim_agg=sim_agg,
        )

        if len(feats) == 0:
            print(f"    SKIP: no features for {target_month}")
            continue

        # ── Score candidates ──────────────────────────────────────────────────
        if use_ml and LGBM_AVAILABLE:
            scores = _ml_score(model, feats, truth, target_month)
        elif use_sim and sim_agg is not None and "sim_profit_mean" in feats.columns:
            scores = sim_score(feats)
        else:
            scores = baseline_score(feats)

        # ── Select opportunities ──────────────────────────────────────────────
        selections = select_opportunities(
            scores, target_month, n_select=n_select
        )
        all_selections.append(selections)
        print(f"    → {len(selections)} opportunities selected")

    # ── 5. SAVE OUTPUT ───────────────────────────────────────────────────────
    print(f"\n[5/5] Saving output...")

    if not all_selections:
        print("ERROR: No selections generated. Check date range and data.")
        sys.exit(1)

    final = pd.concat(all_selections, ignore_index=True)

    # Final validation of format
    assert "TARGET_MONTH" in final.columns
    assert "PEAK_TYPE"    in final.columns
    assert "EID"          in final.columns

    # Deduplicate
    n_before = len(final)
    final = final.drop_duplicates(subset=["TARGET_MONTH","PEAK_TYPE","EID"])
    if len(final) < n_before:
        print(f"  Removed {n_before - len(final)} duplicates")

    # Check monthly constraints
    monthly_counts = final.groupby("TARGET_MONTH").size()
    violations = monthly_counts[
        (monthly_counts < MIN_SELECT_PER_MONTH) |
        (monthly_counts > MAX_SELECT_PER_MONTH)
    ]
    if len(violations) > 0:
        print(f"  WARNING: {len(violations)} months violate selection constraints:")
        print(violations)

    # Save
    output_path = Path(args.output)
    final.to_csv(output_path, index=False)

    print(f"\n{'='*65}")
    print(f"  ✓ Saved {len(final):,} selections → {output_path}")
    print(f"  Months covered: {final['TARGET_MONTH'].nunique()}")
    print(f"  Avg per month:  {monthly_counts.mean():.1f}")
    print(f"  PEAK_TYPE dist: {final['PEAK_TYPE'].value_counts().to_dict()}")
    print(f"  Runtime:        {(time.time()-t0)/60:.1f} minutes")
    print(f"{'='*65}\n")

    return final


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _run_validation(
    truth: pd.DataFrame,
    use_ml: bool,
    use_sim: bool,
    n_select: int,
):
    """Run walk-forward validation on 2020-2022 train → 2023 validate."""
    train_months = get_month_range(TRAIN_START, TRAIN_END)
    val_months   = get_month_range(VALID_START, VALID_END)

    print(f"  Train: {TRAIN_START} → {TRAIN_END} ({len(train_months)} months)")
    print(f"  Val:   {VALID_START} → {VALID_END} ({len(val_months)} months)")

    # Build feature matrix for all months (train + val)
    all_months = train_months + val_months
    all_features = []

    for target_month in all_months:
        truth_past = truth[truth["MONTH"] < target_month]
        if len(truth_past) < 3:
            continue

        sim_agg = None
        if use_sim:
            sim_raw = load_sim_monthly_for_target(target_month)
            if sim_raw is not None:
                sim_agg = aggregate_sim_monthly_across_scenarios(sim_raw)

        feats = build_features_for_month(target_month, truth_past, sim_agg)
        all_features.append(feats)

    if not all_features:
        print("  ERROR: could not build validation features")
        return

    feature_matrix = pd.concat(all_features, ignore_index=True)

    # Run walk-forward
    all_sel, eval_result = walk_forward_validate(
        truth=truth,
        feature_matrix=feature_matrix,
        val_months=val_months,
        train_end_month=TRAIN_END,
        n_select=n_select,
        use_ml=use_ml,
    )

    # Also evaluate on training period (in-sample, just for reference)
    # Build training period selections
    train_sel = []
    for target_month in train_months[-12:]:  # last 12 months of train
        feats = feature_matrix[feature_matrix["TARGET_MONTH"] == target_month]
        if len(feats) == 0:
            continue
        if "sim_profit_mean" in feats.columns:
            scores = sim_score(feats)
        else:
            scores = baseline_score(feats)
        sel = select_opportunities(scores, target_month, n_select)
        train_sel.append(sel)

    if train_sel:
        train_sel_df = pd.concat(train_sel, ignore_index=True)
        eval_train = evaluate_selections(
            train_sel_df, truth, train_months[-12:]
        )
        print_evaluation(eval_train, "Training Period (last 12m, in-sample)")

    # Save validation selections
    if len(all_sel) > 0:
        val_path = OUTPUT_DIR / "validation_selections.csv"
        all_sel.to_csv(val_path, index=False)
        print(f"\n  Validation selections saved → {val_path}")


def _ml_score(
    model: "ProfitabilityModel",
    feats: pd.DataFrame,
    truth: pd.DataFrame,
    target_month: str,
) -> pd.DataFrame:
    """Train model on past and score current features."""
    # Training data: all features for months before target
    past_months = [m for m in truth["MONTH"].unique() if m < target_month]
    if len(past_months) < 3:
        # Fall back to heuristic
        return baseline_score(feats)

    # Build minimal training features from truth
    # (use a simple set of historical features as proxy)
    train_rows = []
    for month in past_months:
        truth_m = truth[truth["MONTH"] == month]
        for _, row in truth_m.iterrows():
            train_rows.append({
                "EID":             row["EID"],
                "PEAKID":          row["PEAKID"],
                "TARGET_MONTH":    month,
                "pr_last":         row["PR"],
                "cost_last":       row["C"],
                "est_profit_ly":   row["PR"] - row["C"],
                "is_profitable":   row["is_profitable"],
                "profit":          row["profit"],
            })

    if not train_rows:
        return baseline_score(feats)

    train_df = pd.DataFrame(train_rows)
    feat_cols = [c for c in feats.columns
                 if c in train_df.columns and
                 c not in ["EID","PEAKID","TARGET_MONTH",
                            "is_profitable","profit"]]

    if len(feat_cols) < 2:
        return baseline_score(feats)

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        X_tr = train_df[feat_cols].fillna(0)
        y_tr = train_df["is_profitable"]

        if y_tr.sum() < 3:
            return baseline_score(feats)

        # Use available feature columns in feats
        shared_cols = [c for c in feat_cols if c in feats.columns]
        X_val = feats[shared_cols].fillna(0)

        model.fit(
            train_df[feat_cols], y_tr, train_df["profit"]
        )
        scores_s = model.score(feats[feat_cols] if all(c in feats.columns for c in feat_cols)
                               else feats[shared_cols])

        scores = feats[["EID","PEAKID","TARGET_MONTH"]].copy()
        scores["score"] = scores_s.values
        return scores

    except Exception as e:
        print(f"    WARNING: ML scoring failed ({e}), using heuristic")
        return baseline_score(feats)


if __name__ == "__main__":
    main()