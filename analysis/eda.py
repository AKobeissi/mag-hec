"""
eda.py — Exploratory Data Analysis for MAG Energy Solutions Challenge.

Run this FIRST to understand your data before training any model.

Usage:
    python eda.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

from config import COSTS_DIR, PRICES_DIR
from data_loader import (
    load_costs,
    load_prices_and_aggregate,
    compute_ground_truth,
    summarize_dataframe,
)


def run_eda():
    print("\n" + "="*65)
    print("  MAG Energy Solutions — Exploratory Data Analysis")
    print("="*65)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading costs...")
    costs = load_costs(COSTS_DIR)
    summarize_dataframe(costs, "Costs")

    print("\n[2] Loading + aggregating prices...")
    pr = load_prices_and_aggregate(PRICES_DIR)
    summarize_dataframe(pr, "Monthly PR")

    # ── 2. Ground truth ───────────────────────────────────────────────────────
    print("\n[3] Computing ground truth...")
    truth = compute_ground_truth(pr, costs)

    # ── 3. Key EDA questions ──────────────────────────────────────────────────

    print("\n" + "─"*50)
    print("  DATA OVERVIEW")
    print("─"*50)
    print(f"  Unique EIDs in costs:    {costs['EID'].nunique():,}")
    print(f"  Unique EIDs in prices:   {pr['EID'].nunique():,}")
    print(f"  Unique EIDs in truth:    {truth['EID'].nunique():,}")
    print(f"  Total opportunities:     {len(truth):,}")
    print(f"  Months covered:          {truth['MONTH'].nunique()}")

    print("\n" + "─"*50)
    print("  PROFITABILITY OVERVIEW")
    print("─"*50)
    print(f"  Overall profit rate:     {100*truth['is_profitable'].mean():.2f}%")
    print(f"  Total profitable:        {truth['is_profitable'].sum():,}")
    print(f"  Mean profit (all):       {truth['profit'].mean():.2f}")
    print(f"  Mean profit (profitable):{truth[truth['is_profitable']==1]['profit'].mean():.2f}")
    print(f"  Mean profit (losing):    {truth[truth['is_profitable']==0]['profit'].mean():.2f}")

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

    print("\n  By Month (aggregated across EIDs):")
    by_month = truth.groupby("MONTH").agg(
        n_total  = ("is_profitable", "count"),
        n_profit = ("is_profitable", "sum"),
        rate     = ("is_profitable", "mean"),
        pr_mean  = ("PR", "mean"),
        c_mean   = ("C", "mean"),
    )
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
    truth["MONTH_DT"] = pd.to_datetime(truth["MONTH"] + "-01")
    truth["MONTH_PREV_YR"] = (
        truth["MONTH_DT"] - pd.DateOffset(years=1)
    ).dt.to_period("M").astype(str)

    merged = truth[["EID","PEAKID","MONTH","is_profitable"]].merge(
        truth[["EID","PEAKID","MONTH","is_profitable"]].rename(
            columns={"MONTH": "MONTH_PREV_YR",
                     "is_profitable": "profitable_ly"}
        ),
        on=["EID","PEAKID","MONTH_PREV_YR"],
        how="inner"
    )

    if len(merged) == 0:
        print("  Not enough data for same-month-LY analysis")
        return

    # Accuracy: if profitable_ly=1, what fraction also profitable this year?
    cond_prob = merged.groupby("profitable_ly")["is_profitable"].mean()
    print(f"  P(profitable | was profitable LY):      "
          f"{cond_prob.get(1, 0):.3f}")
    print(f"  P(profitable | was NOT profitable LY):  "
          f"{cond_prob.get(0, 0):.3f}")
    print(f"  → Seasonal persistence is "
          f"{'STRONG' if abs(cond_prob.get(1,0) - cond_prob.get(0,0)) > 0.05 else 'WEAK'}")


if __name__ == "__main__":
    run_eda()