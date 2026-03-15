"""
correlation_analysis.py — Validate sim_monthly as a predictor of realized prices.

This is Person B's most important analysis. It answers:
  1. How well does PSM predict realized PR?          (primary signal strength)
  2. Does ACTIVATIONLEVEL predict whether PR > 0?    (activation prediction)
  3. Which scenario (1/2/3) is most accurate?        (scenario weighting)
  4. Which sim features correlate with profitability? (feature priority)
  5. What F1 can a naive sim-only rule achieve?       (baseline to beat)

Run AFTER eda.py has completed (uses its cached sim_candidates.parquet).
Results saved to output/correlation_analysis_2022.parquet for Person C.

Usage:
    python analysis/correlation_analysis.py
    python analysis/correlation_analysis.py --year 2021
    python analysis/correlation_analysis.py --year 2022 --months 6
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

# ── Path setup — works whether script is in root or analysis/ subfolder ───────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent if SCRIPT_DIR.name == "analysis" else SCRIPT_DIR
sys.path.insert(0, str(ROOT_DIR))

from config import SIM_MONTHLY_PATHS, OUTPUT_DIR
from data_loader import (
    load_costs,
    load_prices_and_aggregate,
    build_sim_candidate_universe,
    compute_ground_truth,
    load_sim_monthly_for_target,
    aggregate_sim_across_scenarios,
    get_month_range,
    SIM_MONTHLY_PATHS,
    SIM_DAILY_PATHS,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Correlation analysis: sim vs realized prices"
    )
    p.add_argument("--year",   type=int, default=2022,
                   help="Year to analyse (default: 2022)")
    p.add_argument("--months", type=int, default=12,
                   help="How many months of that year to use (default: 12)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_base_data():
    """Load costs, prices, candidates, truth — reuses EDA cache."""
    print("Loading base data (reusing EDA cache where possible)...")

    costs = load_costs()
    pr    = load_prices_and_aggregate()

    cache = OUTPUT_DIR / "sim_candidates.parquet"
    if cache.exists():
        print(f"  Loading sim_candidates from cache: {cache}")
        sim_candidates = pd.read_parquet(cache)
    else:
        print("  Cache not found — rebuilding sim_candidates (slow)...")
        sim_candidates = build_sim_candidate_universe(
            sim_monthly_paths=SIM_MONTHLY_PATHS,
            sim_daily_paths=SIM_DAILY_PATHS,
            pr=pr,
            costs=costs,
            cache_path=cache,
        )

    # Use ONLY 2020-2023 truth for analysis — never touch 2024
    truth = compute_ground_truth(
        pr, costs, sim_candidates,
        months=get_month_range("2020-01", "2023-12")
    )

    return costs, pr, sim_candidates, truth


# ─────────────────────────────────────────────────────────────────────────────
# SIM FEATURE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_sim_for_months(months: list) -> pd.DataFrame:
    """
    Load and aggregate sim_monthly for a list of target months.
    Returns one row per (EID, PEAKID, TARGET_MONTH).
    """
    results = []
    t0 = time.time()

    for i, tm in enumerate(months):
        print(f"  [{i+1}/{len(months)}] Loading sim for {tm}...", end=" ")
        t1 = time.time()

        sim_raw = load_sim_monthly_for_target(tm, SIM_MONTHLY_PATHS)
        if sim_raw is None or len(sim_raw) == 0:
            print("no data")
            continue

        sim_agg = aggregate_sim_across_scenarios(sim_raw)
        sim_agg["TARGET_MONTH"] = tm
        results.append(sim_agg)

        elapsed = time.time() - t1
        remaining = (len(months) - i - 1) * elapsed
        print(f"{len(sim_agg):,} EIDs | {elapsed:.1f}s | "
              f"~{remaining/60:.0f}min remaining")

    if not results:
        raise ValueError("No sim data loaded for requested months.")

    sim_all = pd.concat(results, ignore_index=True)
    print(f"\n  ✓ Sim loaded: {len(sim_all):,} rows | "
          f"{sim_all['TARGET_MONTH'].nunique()} months | "
          f"{(time.time()-t0)/60:.1f} min total\n")
    return sim_all


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 1: PSM vs Realized PR Correlation
# ─────────────────────────────────────────────────────────────────────────────

def analysis_psm_vs_pr(df: pd.DataFrame):
    """
    How well does the simulated price (PSM) predict realized price (PR)?
    This validates whether sim_monthly is a useful forward-looking signal.
    """
    print("\n" + "="*60)
    print("  ANALYSIS 1 — PSM vs Realized PR Correlation")
    print("="*60)

    # ── Overall ───────────────────────────────────────────────────────────────
    c_all = df[["PR", "psm_abs_mean"]].corr().iloc[0, 1]
    print(f"\n  PSM vs PR — all candidates:       r = {c_all:.3f}")

    # Only on EIDs that actually activated (PR > 0)
    active = df[df["PR"] > 0].copy()
    c_act  = active[["PR", "psm_abs_mean"]].corr().iloc[0, 1]
    print(f"  PSM vs PR — active only (PR > 0): r = {c_act:.3f}  "
          f"({len(active):,} rows, "
          f"{100*len(active)/len(df):.1f}% of candidates)")

    # ── By scenario ──────────────────────────────────────────────────────────
    print(f"\n  Per-scenario accuracy:")
    print(f"  {'Scenario':>10}  {'corr (all)':>12}  {'corr (active)':>14}  "
          f"{'mean PSM':>10}  {'mean PR':>10}")

    for s in [1, 2, 3]:
        col = f"psm_s{s}"
        if col not in df.columns:
            continue
        c_s_all = df[["PR", col]].corr().iloc[0, 1]
        c_s_act = active[["PR", col]].corr().iloc[0, 1] if len(active) > 0 else 0
        mean_psm = df[col].mean()
        mean_pr  = df["PR"].mean()
        print(f"  {'Scenario ' + str(s):>10}  {c_s_all:>12.3f}  "
              f"{c_s_act:>14.3f}  {mean_psm:>10.1f}  {mean_pr:>10.1f}")

    # ── By PEAKID ────────────────────────────────────────────────────────────
    print(f"\n  By PEAKID:")
    for pid, name in [(0, "OFF"), (1, "ON")]:
        sub = df[df["PEAKID"] == pid]
        c   = sub[["PR", "psm_abs_mean"]].corr().iloc[0, 1]
        sub_act = sub[sub["PR"] > 0]
        c_a = sub_act[["PR", "psm_abs_mean"]].corr().iloc[0, 1] if len(sub_act) > 0 else 0
        print(f"    {name}-peak: r={c:.3f} (all)  r={c_a:.3f} (active)  "
              f"n={len(sub):,}")

    # ── By month ─────────────────────────────────────────────────────────────
    print(f"\n  Monthly correlations (PSM vs PR, active EIDs only):")
    print(f"  {'Month':>10}  {'r':>8}  {'n_active':>10}  {'n_total':>10}")
    for tm, grp in df.groupby("TARGET_MONTH"):
        act_g = grp[grp["PR"] > 0]
        if len(act_g) < 10:
            continue
        c_m = act_g[["PR", "psm_abs_mean"]].corr().iloc[0, 1]
        print(f"  {tm:>10}  {c_m:>8.3f}  {len(act_g):>10,}  {len(grp):>10,}")

    # ── Interpretation ────────────────────────────────────────────────────────
    print(f"\n  ── Interpretation ──")
    if c_act > 0.6:
        print(f"  ✓ STRONG signal (r={c_act:.3f}): PSM is a reliable predictor of PR.")
        print(f"    psm_abs_mean should be your primary feature.")
    elif c_act > 0.3:
        print(f"  ~ MODERATE signal (r={c_act:.3f}): PSM helps but is noisy.")
        print(f"    Combine with historical PR persistence features.")
    else:
        print(f"  ✗ WEAK signal (r={c_act:.3f}): PSM has low predictive power for PR.")
        print(f"    Focus on: activation prediction + historical persistence.")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2: Does sim predict ACTIVATION (PR > 0 vs PR = 0)?
# ─────────────────────────────────────────────────────────────────────────────

def analysis_activation_prediction(df: pd.DataFrame):
    """
    Can sim_monthly tell us which EIDs will have PR > 0 at all?
    This is critical because 96.7% of candidates have PR = 0.
    """
    print("\n" + "="*60)
    print("  ANALYSIS 2 — Does sim predict activation (PR > 0)?")
    print("="*60)

    df = df.copy()
    df["was_active"]          = (df["PR"] > 0).astype(int)
    df["sim_says_active"]     = (df["act_mean"] > 0).astype(int)
    df["sim_says_active_psm"] = (df["psm_abs_mean"] > 0).astype(int)

    # ── Activation level by actual activation ────────────────────────────────
    act_stats = df.groupby("was_active")["act_mean"].agg(["mean","median","std","count"])
    print(f"\n  ACTIVATIONLEVEL statistics by actual activation:")
    print(f"  {'was_active':>12}  {'mean':>8}  {'median':>8}  "
          f"{'std':>8}  {'count':>10}")
    for idx, row in act_stats.iterrows():
        label = "PR > 0" if idx == 1 else "PR = 0"
        print(f"  {label:>12}  {row['mean']:>8.3f}  {row['median']:>8.3f}  "
              f"{row['std']:>8.3f}  {int(row['count']):>10,}")

    # ── Confusion matrix: sim_active vs actual_active ─────────────────────────
    print(f"\n  Confusion matrix (sim predicts active vs actually active):")
    cm = (df.groupby(["sim_says_active", "was_active"])
            .size()
            .unstack(fill_value=0))
    print(f"  {cm.to_string()}")

    if 1 in cm.index and 1 in cm.columns:
        tp = cm.loc[1, 1]
        fp = cm.loc[1, 0] if 0 in cm.columns else 0
        fn = cm.loc[0, 1] if 0 in cm.index else 0
        tn = cm.loc[0, 0] if (0 in cm.index and 0 in cm.columns) else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
        accuracy  = (tp + tn) / len(df)

        print(f"\n  When sim says active (act_mean > 0):")
        print(f"    Precision  (actually active):  {100*precision:.1f}%")
        print(f"    Recall     (catches all active): {100*recall:.1f}%")
        print(f"    F1:                             {f1:.4f}")
        print(f"    Accuracy:                       {100*accuracy:.1f}%")

    # ── Threshold analysis: best ACTIVATIONLEVEL cutoff ──────────────────────
    print(f"\n  ACTIVATIONLEVEL threshold analysis:")
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  "
          f"{'F1':>8}  {'n_selected':>12}")

    thresholds = [0, 1, 5, 10, 20, 30, 50]
    best_f1 = 0
    best_thresh = 0

    for thresh in thresholds:
        pred = (df["act_mean"] >= thresh).astype(int)
        tp = ((pred == 1) & (df["was_active"] == 1)).sum()
        fp = ((pred == 1) & (df["was_active"] == 0)).sum()
        fn = ((pred == 0) & (df["was_active"] == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        n_sel = pred.sum()
        marker = " ← best" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
        print(f"  {thresh:>10}  {100*prec:>9.1f}%  {100*rec:>7.1f}%  "
              f"{f1:>8.4f}  {n_sel:>12,}{marker}")

    print(f"\n  Best activation threshold: {best_thresh} "
          f"(F1={best_f1:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 3: Naive sim-only profitability prediction
# ─────────────────────────────────────────────────────────────────────────────

def analysis_naive_baseline(df: pd.DataFrame):
    """
    If we just use |PSM| > C_estimate as our selection rule, what F1 do we get?
    This is the baseline that the ML model needs to beat.
    """
    print("\n" + "="*60)
    print("  ANALYSIS 3 — Naive sim-only baseline performance")
    print("="*60)

    df = df.copy()

    # Naive rule: select top-100 EIDs by psm_abs_mean per month
    # (simulating what selecting 100/month based on PSM alone would give)
    selections = []
    for tm, grp in df.groupby("TARGET_MONTH"):
        top100 = grp.nlargest(100, "psm_abs_mean")
        top100 = top100.copy()
        top100["selected"] = 1
        selections.append(top100)

    sel_df = pd.concat(selections, ignore_index=True)
    merged = df.merge(
        sel_df[["EID","PEAKID","TARGET_MONTH","selected"]],
        on=["EID","PEAKID","TARGET_MONTH"],
        how="left"
    )
    merged["selected"] = merged["selected"].fillna(0).astype(int)

    # Compute F1
    for peak_id, peak_name in [(0, "OFF"), (1, "ON")]:
        sub = merged[merged["PEAKID"] == peak_id]
        tp = ((sub["selected"] == 1) & (sub["is_profitable"] == 1)).sum()
        fp = ((sub["selected"] == 1) & (sub["is_profitable"] == 0)).sum()
        fn = ((sub["selected"] == 0) & (sub["is_profitable"] == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        print(f"\n  {peak_name}-Peak (top-100 by PSM):")
        print(f"    TP={tp:,}  FP={fp:,}  FN={fn:,}")
        print(f"    Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    # Also compute profit
    sel_only = merged[merged["selected"] == 1]
    total_profit = sel_only["profit"].sum()
    n_prof = (sel_only["profit"] > 0).sum()
    n_lose = (sel_only["profit"] <= 0).sum()

    print(f"\n  Profit from naive top-100 PSM selection:")
    print(f"    Total net profit: {total_profit:,.2f}")
    print(f"    Profitable picks: {n_prof:,}")
    print(f"    Losing picks:     {n_lose:,}")
    print(f"\n  ← This is your baseline. ML model must beat this F1 and profit.")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 4: Feature correlation with profitability
# ─────────────────────────────────────────────────────────────────────────────

def analysis_feature_importance(df: pd.DataFrame):
    """
    Which sim features are most correlated with is_profitable?
    Tells Person B which features to prioritize in price_predictor.py.
    """
    print("\n" + "="*60)
    print("  ANALYSIS 4 — Sim feature correlations with profitability")
    print("="*60)

    sim_cols = [
        "psm_abs_mean", "psm_abs_min", "psm_abs_max", "psm_abs_std",
        "act_mean", "act_max", "act_pos_frac",
        "wind_mean", "solar_mean", "hydro_mean",
        "nonrenew_mean", "external_mean",
        "load_mean", "transmission_mean",
        "source_impact_sum", "scenario_agree",
    ]
    sim_cols = [c for c in sim_cols if c in df.columns]

    corrs = (df[sim_cols + ["is_profitable"]]
             .corr()["is_profitable"]
             .drop("is_profitable")
             .sort_values(ascending=False))

    print(f"\n  Feature correlations with is_profitable (|r| sorted):")
    print(f"  {'Feature':35s}  {'r':>8}  {'direction'}")
    for feat, val in corrs.items():
        bar      = "█" * int(abs(val) * 40)
        sign     = "↑ positive" if val > 0 else "↓ negative"
        print(f"  {feat:35s}  {val:>8.4f}  {bar} {sign}")

    # ── Mean feature values profitable vs not ─────────────────────────────────
    print(f"\n  Mean feature values by profitability:")
    compare_cols = [c for c in [
        "psm_abs_mean", "act_mean", "act_pos_frac",
        "source_impact_sum", "scenario_agree", "transmission_mean"
    ] if c in df.columns]

    comparison = df.groupby("is_profitable")[compare_cols].mean()
    print(comparison.round(3).to_string())

    # ── Scenario agreement vs profitability ───────────────────────────────────
    if "scenario_agree" in df.columns:
        agree_stats = df.groupby("scenario_agree")["is_profitable"].agg(
            ["mean","count"]
        )
        print(f"\n  Scenario agreement vs profit rate:")
        print(agree_stats.round(4).to_string())
        print(f"  (1 = all 3 scenarios agree on direction)")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 5: PSM accuracy by EID type
# ─────────────────────────────────────────────────────────────────────────────

def analysis_by_eid_type(df: pd.DataFrame):
    """
    Is PSM more accurate for some types of EIDs than others?
    Helps identify when to trust vs distrust the sim signal.
    """
    print("\n" + "="*60)
    print("  ANALYSIS 5 — PSM accuracy by EID characteristics")
    print("="*60)

    df = df.copy()
    df["psm_error"]      = (df["psm_abs_mean"] - df["PR"]).abs()
    df["psm_error_rel"]  = df["psm_error"] / (df["PR"] + 1e-9)
    df["psm_overestimates"] = (df["psm_abs_mean"] > df["PR"]).astype(int)

    # Overall error stats
    print(f"\n  PSM prediction error (vs realized PR):")
    print(f"    Mean absolute error:     {df['psm_error'].mean():.2f}")
    print(f"    Median absolute error:    {df['psm_error'].median():.2f}")
    print(f"    PSM overestimates PR:     {100*df['psm_overestimates'].mean():.1f}% of the time")

    # Error for active vs inactive EIDs
    print(f"\n  PSM error by activation status:")
    err_by_act = df.groupby("was_active" if "was_active" in df.columns
                            else (df["PR"] > 0).astype(int))[
        ["psm_error","psm_overestimates"]
    ].mean()
    print(err_by_act.round(3).to_string())

    # Does high PSM reliability predict profitable?
    df["psm_nonzero"] = (df["psm_abs_mean"] > 0).astype(int)
    cross = df.groupby(["psm_nonzero"])["is_profitable"].agg(["mean","count"])
    cross.index = ["PSM=0", "PSM>0"]
    print(f"\n  Profit rate when PSM=0 vs PSM>0:")
    print(cross.round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    n_months    = min(args.months, 12)
    year        = args.year
    start_month = f"{year}-01"
    end_month   = f"{year}-{n_months:02d}"
    test_months = get_month_range(start_month, end_month)

    print("\n" + "="*60)
    print("  MAG Energy Solutions — Correlation Analysis")
    print(f"  Period: {start_month} → {end_month} ({len(test_months)} months)")
    print("="*60 + "\n")

    # ── Load base data ────────────────────────────────────────────────────────
    costs, pr, sim_candidates, truth = load_base_data()

    # ── Load sim features for test months ────────────────────────────────────
    print(f"\nLoading sim_monthly for {len(test_months)} months...")
    print("(Each month takes ~30-60 seconds)\n")
    sim_all = load_sim_for_months(test_months)

    # ── Join sim features with realized truth ─────────────────────────────────
    print("Joining sim features with realized truth...")

    truth_test = truth[truth["MONTH"].isin(test_months)][
        ["EID","PEAKID","MONTH","PR","C","profit","is_profitable"]
    ].copy()
    truth_test = truth_test.rename(columns={"MONTH": "TARGET_MONTH"})

    df = sim_all.merge(
        truth_test,
        on=["EID","PEAKID","TARGET_MONTH"],
        how="inner"
    )

    # Add was_active flag here so all analyses can use it
    df["was_active"] = (df["PR"] > 0).astype(int)

    print(f"\n  Joined dataset: {len(df):,} rows")
    print(f"  Coverage: {100*len(df)/len(sim_all):.1f}% of sim candidates "
          f"matched to truth")
    print(f"  Active EIDs (PR > 0): {df['was_active'].sum():,} "
          f"({100*df['was_active'].mean():.2f}%)")
    print(f"  Profitable:           {df['is_profitable'].sum():,} "
          f"({100*df['is_profitable'].mean():.2f}%)\n")

    # ── Run all analyses ──────────────────────────────────────────────────────
    analysis_psm_vs_pr(df)
    analysis_activation_prediction(df)
    analysis_naive_baseline(df)
    analysis_feature_importance(df)
    analysis_by_eid_type(df)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY FOR PERSON B")
    print("="*60)

    c_act = df[df["PR"] > 0][["PR","psm_abs_mean"]].corr().iloc[0,1]
    best_sim_feat = (df[["psm_abs_mean","act_mean","act_pos_frac",
                          "source_impact_sum","scenario_agree","is_profitable"]]
                     .corr()["is_profitable"]
                     .drop("is_profitable")
                     .abs()
                     .idxmax())

    print(f"""
  PSM vs PR correlation (active EIDs): {c_act:.3f}
  Best sim feature for profitability:  {best_sim_feat}
  
  Key decisions for price_predictor.py:
  
  {'✓' if c_act > 0.5 else '✗'} Use psm_abs_mean as primary sim feature
    {'→ Strong signal, trust it' if c_act > 0.5 else '→ Weak signal, use as secondary only'}
  
  ✓ Always include:
    - profitable_same_month_ly  (EDA: +25.7% lift)
    - profit_rate_12m           (EDA: strong persistence)
    - pr_nonzero_rate_12m       (predicts activation)
    - act_mean                  (sim activation signal)
    - psm_abs_mean              (forward price signal)
  
  Handoff to Person C:
    - price_predictor.py (feature builder)
    - output/correlation_analysis_{year}.parquet (for feature validation)
    - This summary (for model design decisions)
    """)

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / f"correlation_analysis_{year}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  ✓ Results saved → {out_path}")
    print(f"    Pass this to Person C for feature validation.\n")


if __name__ == "__main__":
    main()