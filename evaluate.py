"""
Evaluation script for the CSD Data Challenge.

Computes:
  - Axis 1: F1-score (quality of selection)
  - Axis 2: Profit total net (economic value)

Usage:
  python evaluate.py opportunities.csv --start-month 2024-01 --end-month 2024-12
  python evaluate.py opportunities.csv --start-month 2020-01 --end-month 2023-12
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"


def parse_args():
    parser = argparse.ArgumentParser(description="CSD Data Challenge Evaluation")
    parser.add_argument("csv_path", type=str, help="Path to the opportunities.csv file")
    parser.add_argument("--start-month", required=True, help="Start month (YYYY-MM)")
    parser.add_argument("--end-month", required=True, help="End month (YYYY-MM)")
    return parser.parse_args()


def load_prices(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    folder = data_dir / "prices"
    parts = sorted(folder.glob("prices*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def load_costs(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    folder = data_dir / "costs"
    parts = sorted(folder.glob("costs*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def get_evaluation_months(start_month: str, end_month: str) -> list[str]:
    months = pd.date_range(start_month + "-01", end_month + "-01", freq="MS")
    return [m.strftime("%Y-%m") for m in months]


def compute_monthly_pr(prices: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """Compute PR_o = sum of PRICEREALIZED per (EID, MONTH, PEAKID)."""
    prices = prices.copy()
    prices["MONTH"] = prices["DATETIME"].dt.to_period("M").astype(str)
    prices = prices[prices["MONTH"].isin(months)]
    pr = prices.groupby(["EID", "MONTH", "PEAKID"])["PRICEREALIZED"].sum().reset_index()
    pr.rename(columns={"PRICEREALIZED": "PR"}, inplace=True)
    return pr


def compute_ground_truth(pr: pd.DataFrame, costs: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """Determine all profitable opportunities: |PR_o| - C_o > 0."""
    costs_filtered = costs[costs["MONTH"].isin(months)].copy()
    costs_filtered.rename(columns={"C": "COST"}, inplace=True)

    truth = pr.merge(costs_filtered, on=["EID", "MONTH", "PEAKID"], how="outer")
    truth["PR"] = truth["PR"].fillna(0)
    truth["COST"] = truth["COST"].fillna(0)
    truth["PROFIT"] = truth["PR"].abs() - truth["COST"]
    truth["PROFITABLE"] = truth["PROFIT"] > 0

    return truth


def load_selections(csv_path: str, months: list[str]) -> pd.DataFrame:
    """Load and validate the participant's CSV."""
    df = pd.read_csv(csv_path)

    required_cols = {"TARGET_MONTH", "PEAK_TYPE", "EID"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.drop_duplicates(subset=["TARGET_MONTH", "PEAK_TYPE", "EID"])

    df["PEAKID"] = df["PEAK_TYPE"].map({"OFF": 0, "ON": 1})
    invalid_peak = df["PEAKID"].isna().sum()
    if invalid_peak > 0:
        print(f"  Warning: {invalid_peak} rows with invalid PEAK_TYPE (not ON/OFF), dropped.")
        df = df.dropna(subset=["PEAKID"])
    df["PEAKID"] = df["PEAKID"].astype(int)

    df = df.rename(columns={"TARGET_MONTH": "MONTH"})
    df = df[df["MONTH"].isin(months)]

    return df[["MONTH", "PEAKID", "EID"]]


def enforce_max_selections(selections: pd.DataFrame, max_per_month: int = 100) -> pd.DataFrame:
    """Keep only the first max_per_month selections per month."""
    kept = []
    for month, group in selections.groupby("MONTH"):
        if len(group) > max_per_month:
            print(f"  Warning: {len(group)} selections for {month}, keeping first {max_per_month}.")
            group = group.head(max_per_month)
        kept.append(group)
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=selections.columns)


def compute_f1(selections: pd.DataFrame, truth: pd.DataFrame, months: list[str]):
    """Compute F1-score per PEAKID, then average."""
    profitable = truth[truth["PROFITABLE"]][["EID", "MONTH", "PEAKID"]].copy()
    profitable["IS_PROFITABLE"] = True

    selected = selections.copy()
    selected["IS_SELECTED"] = True

    merged = profitable.merge(selected, on=["EID", "MONTH", "PEAKID"], how="outer")
    merged["IS_PROFITABLE"] = merged["IS_PROFITABLE"].fillna(False)
    merged["IS_SELECTED"] = merged["IS_SELECTED"].fillna(False)

    results = {}
    for peak_id, peak_name in [(0, "OFF"), (1, "ON")]:
        sub = merged[merged["PEAKID"] == peak_id]
        tp = ((sub["IS_SELECTED"]) & (sub["IS_PROFITABLE"])).sum()
        fp = ((sub["IS_SELECTED"]) & (~sub["IS_PROFITABLE"])).sum()
        fn = ((~sub["IS_SELECTED"]) & (sub["IS_PROFITABLE"])).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[peak_name] = {
            "TP": int(tp), "FP": int(fp), "FN": int(fn),
            "Precision": precision, "Recall": recall, "F1": f1
        }

    return results


def compute_profit(selections: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Compute total net profit for selected opportunities."""
    selected_truth = selections.merge(truth, on=["EID", "MONTH", "PEAKID"], how="left")
    selected_truth["PR"] = selected_truth["PR"].fillna(0)
    selected_truth["COST"] = selected_truth["COST"].fillna(0)
    selected_truth["PROFIT"] = selected_truth["PR"].abs() - selected_truth["COST"]

    total_profit = selected_truth["PROFIT"].sum()
    n_profitable = (selected_truth["PROFIT"] > 0).sum()
    n_losing = (selected_truth["PROFIT"] <= 0).sum()

    return {
        "total_profit": total_profit,
        "n_selected": len(selected_truth),
        "n_profitable": int(n_profitable),
        "n_losing": int(n_losing),
    }


def main():
    args = parse_args()
    months = get_evaluation_months(args.start_month, args.end_month)
    print(f"Evaluation period: {args.start_month} to {args.end_month} ({len(months)} months)")

    print("\nLoading data...")
    prices = load_prices()
    costs = load_costs()
    print(f"  Prices: {len(prices):,} rows")
    print(f"  Costs:  {len(costs):,} rows")

    print("\nComputing ground truth (profitable opportunities)...")
    pr = compute_monthly_pr(prices, months)
    truth = compute_ground_truth(pr, costs, months)
    n_profitable = truth["PROFITABLE"].sum()
    n_total = len(truth)
    print(f"  Total opportunities with non-zero PR or C: {n_total:,}")
    print(f"  Profitable (|PR| - C > 0): {n_profitable:,} ({100*n_profitable/n_total:.1f}%)")

    print(f"\nLoading selections from: {args.csv_path}")
    selections = load_selections(args.csv_path, months)
    selections = enforce_max_selections(selections)
    print(f"  Valid selections: {len(selections):,}")

    sel_per_month = selections.groupby("MONTH").size()
    print(f"  Selections per month: min={sel_per_month.min()}, max={sel_per_month.max()}, "
          f"mean={sel_per_month.mean():.0f}")

    print("\n" + "=" * 60)
    print("AXE 1 — F1-score (25%)")
    print("=" * 60)
    f1_results = compute_f1(selections, truth, months)
    for peak_name, metrics in f1_results.items():
        print(f"\n  {peak_name}-Peak:")
        print(f"    TP={metrics['TP']:,}  FP={metrics['FP']:,}  FN={metrics['FN']:,}")
        print(f"    Precision={metrics['Precision']:.4f}  Recall={metrics['Recall']:.4f}  F1={metrics['F1']:.4f}")

    avg_f1 = (f1_results["OFF"]["F1"] + f1_results["ON"]["F1"]) / 2
    print(f"\n  Average F1-score: {avg_f1:.4f}")

    print("\n" + "=" * 60)
    print("AXE 2 — Profit total net (25%)")
    print("=" * 60)
    profit_results = compute_profit(selections, truth)
    print(f"\n  Total selections: {profit_results['n_selected']:,}")
    print(f"  Profitable: {profit_results['n_profitable']:,}")
    print(f"  Losing:     {profit_results['n_losing']:,}")
    print(f"  Total net profit: {profit_results['total_profit']:,.2f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  F1-score (avg ON/OFF): {avg_f1:.4f}")
    print(f"  Profit total net:      {profit_results['total_profit']:,.2f}")

    # Monthly breakdown
    print("\n  Monthly breakdown:")
    print(f"  {'MONTH':>10}  {'Sel':>5}  {'TP':>5}  {'FP':>5}  {'Profit':>12}")
    for month in months:
        sel_m = selections[selections["MONTH"] == month]
        truth_m = truth[truth["MONTH"] == month]
        profitable_m = set(truth_m[truth_m["PROFITABLE"]][["EID","PEAKID"]].apply(tuple, axis=1))

        n_sel = len(sel_m)
        tp = sum(1 for _, r in sel_m.iterrows() if (r["EID"], r["PEAKID"]) in profitable_m)
        fp = n_sel - tp

        sel_truth = sel_m.merge(truth_m, on=["EID", "MONTH", "PEAKID"], how="left")
        sel_truth["PR"] = sel_truth["PR"].fillna(0)
        sel_truth["COST"] = sel_truth["COST"].fillna(0)
        profit_m = (sel_truth["PR"].abs() - sel_truth["COST"]).sum()

        print(f"  {month:>10}  {n_sel:>5}  {tp:>5}  {fp:>5}  {profit_m:>12,.2f}")


if __name__ == "__main__":
    main()
