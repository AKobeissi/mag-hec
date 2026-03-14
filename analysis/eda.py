"""
eda.py — Exploratory Data Analysis for MAG Energy Solutions Challenge.

Run this FIRST to understand your data before training any model.

Usage:
    python eda.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is on sys.path when running this script directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from config import COSTS_DIR, PRICES_DIR, OUTPUT_DIR
from data_loader import (
    load_costs,
    load_prices_and_aggregate,
    build_sim_candidate_universe,
    compute_ground_truth,
    summarize_dataframe,
    SIM_MONTHLY_PATHS,
    SIM_DAILY_PATHS,
)


def run_eda():
    print("\n" + "="*65)
    print("  MAG Energy Solutions — Exploratory Data Analysis")
    print("="*65)
 
    # ── 1. Load costs and prices ──────────────────────────────────────────────
    print("\n[1] Loading costs...")
    costs = load_costs(COSTS_DIR)
    summarize_dataframe(costs, "Costs")
 
    print("\n[2] Loading + aggregating prices...")
    pr = load_prices_and_aggregate(PRICES_DIR)
    summarize_dataframe(pr, "Monthly PR")
 
    # ── 2. Build full candidate universe (ALL 4 sources) ─────────────────────
    print("\n[3] Building full candidate universe (union of all 4 data sources)...")
    print("    Per organizer: universe = prices ∪ costs ∪ sim_daily ∪ sim_monthly")
    print("    First run scans ~15 GB — result cached to output/sim_candidates.parquet\n")
 
    cache_path = OUTPUT_DIR / "sim_candidates.parquet"
    sim_candidates = build_sim_candidate_universe(
        sim_monthly_paths=SIM_MONTHLY_PATHS,
        sim_daily_paths=SIM_DAILY_PATHS,
        pr=pr,
        costs=costs,
        start_month="2020-01",
        end_month="2024-12",
        cache_path=cache_path,
    )
    summarize_dataframe(sim_candidates, "Full Candidate Universe")
 
    # ── 3. Correct ground truth ───────────────────────────────────────────────
    print("\n[4] Computing CORRECT ground truth (sim universe as denominator)...")
    truth = compute_ground_truth(pr, costs, sim_candidates)
 
    # Also show the OLD (wrong) calculation for comparison
    print("\n  [COMPARISON] Old wrong calculation (outer join of prices ∪ costs):")
    truth_wrong = pr.merge(costs, on=["EID","MONTH","PEAKID"], how="outer")
    truth_wrong["PR"] = truth_wrong["PR"].fillna(0)
    truth_wrong["C"]  = truth_wrong["C"].fillna(0)
    truth_wrong["profit"] = truth_wrong["PR"] - truth_wrong["C"]
    truth_wrong["is_profitable"] = (truth_wrong["profit"] > 0).astype(int)
    n_w = len(truth_wrong)
    p_w = truth_wrong["is_profitable"].sum()
    print(f"    {n_w:,} rows | {p_w:,} profitable | {100*p_w/n_w:.1f}% ← WRONG (survivorship bias)")
    print(f"  [CORRECT] Using sim universe: {100*truth['is_profitable'].mean():.1f}% ← matches case doc ~5%\n")
 
    # ── 4. Key EDA questions ──────────────────────────────────────────────────
 
    print("\n" + "─"*50)
    print("  DATA OVERVIEW")
    print("─"*50)
    print(f"  Unique EIDs in costs:         {costs['EID'].nunique():,}")
    print(f"  Unique EIDs in prices:         {pr['EID'].nunique():,}")
    print(f"  Unique EIDs in sim universe:   {sim_candidates['EID'].nunique():,}")
    print(f"  Unique EIDs in truth:          {truth['EID'].nunique():,}")
    print(f"  Total opportunities (correct): {len(truth):,}")
    print(f"  Months covered:                {truth['MONTH'].nunique()}")
 
    print("\n" + "─"*50)
    print("  PROFITABILITY OVERVIEW (corrected)")
    print("─"*50)
    print(f"  Overall profit rate:      {100*truth['is_profitable'].mean():.2f}%")
    print(f"  Total profitable:         {truth['is_profitable'].sum():,}")
    print(f"  Total unprofitable:       {(~truth['is_profitable'].astype(bool)).sum():,}")
    print(f"  Mean profit (all):        {truth['profit'].mean():.2f}")
    print(f"  Mean profit (profitable): {truth[truth['is_profitable']==1]['profit'].mean():.2f}")
    print(f"  Mean profit (losing):     {truth[truth['is_profitable']==0]['profit'].mean():.2f}")
 
    print("\n  By PEAKID:")
    by_peak = truth.groupby("PEAKID").agg(
        n_total   = ("is_profitable", "count"),
        n_profit  = ("is_profitable", "sum"),
        rate      = ("is_profitable", "mean"),
        pr_mean   = ("PR", "mean"),
        c_mean    = ("C", "mean"),
        profit_m  = ("profit", "mean"),
    )
    by_peak["rate_pct"] = (by_peak["rate"] * 100).round(2)
    print(by_peak.to_string())
 
    print("\n  By Month (sample — first 12):")
    by_month = truth.groupby("MONTH").agg(
        n_sim_candidates = ("EID", "count"),
        n_profitable     = ("is_profitable", "sum"),
        rate             = ("is_profitable", "mean"),
        pr_mean          = ("PR", "mean"),
        c_mean           = ("C", "mean"),
    ).head(12)
    by_month["rate_pct"] = (by_month["rate"] * 100).round(2)
    print(by_month.to_string())
 
    print("\n" + "─"*50)
    print("  COST DISTRIBUTION")
    print("─"*50)
    zero_cost_frac = (costs["C"] == 0).mean() if len(costs) > 0 else 0
    print(f"  Zero-cost records:       {100*zero_cost_frac:.1f}%")
    print(f"  Cost stats:")
    print(costs["C"].describe().round(4).to_string())
 
    print("\n  How often is C missing (zero by implicit rule)?")
    all_eids_months = truth[["EID","MONTH","PEAKID"]].drop_duplicates()
    merged_check = all_eids_months.merge(
        costs, on=["EID","MONTH","PEAKID"], how="left"
    )
    missing_cost_rate = merged_check["C"].isna().mean()
    print(f"  Missing cost rate: {100*missing_cost_rate:.1f}% (treated as 0)")
 
    print("\n" + "─"*50)
    print("  PRICE (PR) DISTRIBUTION")
    print("─"*50)
    zero_pr_frac = (pr["PR"] == 0).mean()
    print(f"  Zero PR records:         {100*zero_pr_frac:.1f}%")
    print(f"\n  PR stats (|PR|):")
    print(pr["PR"].describe().round(4).to_string())
 
    print("\n" + "─"*50)
    print("  EID PERSISTENCE ANALYSIS")
    print("─"*50)
    # How persistent is profitability for each EID?
    eid_stats = truth.groupby("EID").agg(
        n_months       = ("MONTH", "nunique"),
        profit_rate    = ("is_profitable", "mean"),
        profit_mean    = ("profit", "mean"),
    ).reset_index()
 
    print(f"  EIDs with ≥ 50% profit rate: "
          f"{(eid_stats['profit_rate'] >= 0.5).sum():,}")
    print(f"  EIDs with ≥ 70% profit rate: "
          f"{(eid_stats['profit_rate'] >= 0.7).sum():,}")
    print(f"  EIDs with 100% profit rate:  "
          f"{(eid_stats['profit_rate'] == 1.0).sum():,}")
    print(f"\n  Top 10 most consistently profitable EIDs:")
    top_eids = (eid_stats[eid_stats["n_months"] >= 6]
                .sort_values("profit_rate", ascending=False)
                .head(10))
    print(top_eids.to_string(index=False))
 
    print("\n" + "─"*50)
    print("  SAME-MONTH LAST YEAR SIGNAL STRENGTH")
    print("─"*50)
    _analyze_seasonal_signal(truth)
 
    print("\n" + "─"*50)
    print("  COST vs PROFITABILITY")
    print("─"*50)
    truth["zero_cost"] = (truth["C"] == 0).astype(int)
    print(truth.groupby("zero_cost")["is_profitable"].agg(["mean","count"]))
    print("  (zero_cost=1 means C=0 → any PR>0 is profitable)")
 
    print("\n  ✓ EDA complete.\n")
    return truth, costs, pr
 
 
def _analyze_seasonal_signal(truth: pd.DataFrame):
    """
    Measure how well 'same month last year' predicts profitability.
    If EID was profitable in Jan-2021, was it also profitable in Jan-2022?
    """
    truth = truth.copy()
 
    # For each row, compute what "same month next year" looks like
    # Then join: this year's row with next year's row on EID+PEAKID+month
    truth["MONTH_NEXT_YR"] = truth["MONTH"].apply(
        lambda m: str(pd.Period(m, freq="M") + 12)
    )
 
    # left side:  (EID, PEAKID, MONTH_NEXT_YR, is_profitable_this_yr)
    # right side: (EID, PEAKID, MONTH,          is_profitable_next_yr)
    # join key:   MONTH_NEXT_YR == MONTH
    left  = truth[["EID","PEAKID","MONTH_NEXT_YR","is_profitable"]].copy()
    right = truth[["EID","PEAKID","MONTH","is_profitable"]].rename(
        columns={
            "MONTH":          "MONTH_NEXT_YR",
            "is_profitable":  "profitable_next_yr"
        }
    )
 
    merged = left.merge(right, on=["EID","PEAKID","MONTH_NEXT_YR"], how="inner")
 
    if len(merged) == 0:
        print("  Not enough overlapping years for seasonal analysis")
        return
 
    cond_prob = merged.groupby("is_profitable")["profitable_next_yr"].mean()
    p_if_yes = cond_prob.get(1, 0)
    p_if_no  = cond_prob.get(0, 0)
    lift     = p_if_yes - p_if_no
 
    print(f"  P(profitable next yr | was profitable this yr):     {p_if_yes:.3f}")
    print(f"  P(profitable next yr | was NOT profitable this yr): {p_if_no:.3f}")
    print(f"  → Lift from knowing last year: +{lift:.3f}")
    print(f"  → Seasonal persistence is {'STRONG ✓' if lift > 0.05 else 'WEAK'}")
 
 
if __name__ == "__main__":
    run_eda()
 