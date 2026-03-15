"""
predict_profit.py — Combine price + cost predictions, select top-100, generate opportunities.csv

Pipeline:
  predict_price.py   →  output/predicted_prices.parquet
  cost_predictor.py  →  output/predicted_costs.parquet
                              ↓
                  predict_profit.py  (this script)
                              ↓
                  opportunities.csv  (final submission)

Decision rule — soft exponential cost penalty (α=0.5):
  score = predicted_pr × exp(-0.5 × predicted_c / (predicted_pr + ε))

  Why soft penalty (not hard subtraction):
    Hard subtraction: drops profitable high-PR/high-C EIDs
    Soft penalty α=0.5: scales confidence down, doesn't eliminate
    Tuned on 2023 holdout:
      Price only (A):   $303k profit   std=$298k
      Hard sub   (B):   $130k profit   std=$128k
      Soft α=0.5 (D):   $662k profit   std=$205k  ← best

Usage:
    python predict_profit.py                              # 2023 validation
    python predict_profit.py --start-month 2023-01 --end-month 2023-12
    python predict_profit.py --start-month 2024-01 --end-month 2024-12
    python predict_profit.py --alpha 0.3                  # different penalty
    python predict_profit.py --no-validate                # skip evaluation
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
ROOT_DIR   = SCRIPT_DIR.parent if SCRIPT_DIR.name in ("analysis", "src") else SCRIPT_DIR
sys.path.insert(0, str(ROOT_DIR))

from config import (
    OUTPUT_DIR, OPPORTUNITIES_PATH,
    MIN_SELECT_PER_MONTH, MAX_SELECT_PER_MONTH,
    PEAKID_TO_NAME,
)
from data_loader import (
    load_costs, load_prices_and_aggregate,
    build_sim_candidate_universe, compute_ground_truth,
    get_month_range, SIM_MONTHLY_PATHS, SIM_DAILY_PATHS,
)

OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Combine price + cost predictions → opportunities.csv"
    )
    p.add_argument("--start-month", default="2023-01")
    p.add_argument("--end-month",   default="2023-12")
    p.add_argument("--alpha",       type=float, default=0.5,
                   help="Soft penalty strength (default 0.5, tuned on 2023 val)")
    p.add_argument("--n-select",    type=int,   default=100)
    p.add_argument("--no-validate", action="store_true",
                   help="Skip evaluation against ground truth")
    p.add_argument("--output",      type=str,
                   default=str(OPPORTUNITIES_PATH))
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────

def compute_scores(combined: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Soft exponential cost penalty:
      score = predicted_pr × exp(-α × predicted_c / predicted_pr)

      α=0:   pure price ranking
      α=0.5: soft penalty (tuned on 2023 validation)
      α→∞:   approaches hard subtraction
    """
    pr    = combined["predicted_pr"].fillna(0).clip(lower=0)
    c     = combined["predicted_c"].fillna(0).clip(lower=0)
    score = pr * np.exp(-alpha * c / (pr + 1e-9))

    combined = combined.copy()
    combined["score"]        = score
    combined["cost_ratio"]   = c / (pr + 1e-9)
    combined["pr_exceeds_c"] = (pr > c).astype(int)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_top_n(
    scores: pd.DataFrame,
    target_month: str,
    n: int = 100,
) -> pd.DataFrame:
    """
    Select top-n by score for one month.
    Always exactly n — F1 ceiling math:
      K=100 → F1 max 0.48  |  K=50 → F1 max 0.27
    """
    n        = int(np.clip(n, MIN_SELECT_PER_MONTH, MAX_SELECT_PER_MONTH))
    month_df = scores[scores["TARGET_MONTH"] == target_month].copy()

    if len(month_df) == 0:
        print(f"  WARNING: no candidates for {target_month}")
        return pd.DataFrame(columns=["TARGET_MONTH","PEAK_TYPE","EID"])

    selected = month_df.nlargest(n, "score")
    out      = selected[["EID","PEAKID","TARGET_MONTH"]].copy()
    out["PEAK_TYPE"] = out["PEAKID"].map(PEAKID_TO_NAME)
    return out[["TARGET_MONTH","PEAK_TYPE","EID"]].drop_duplicates()


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    selections: pd.DataFrame,
    truth: pd.DataFrame,
    months: list,
    label: str = "Evaluation",
    verbose: bool = True,
) -> dict:
    """Official F1 + profit evaluation. Mirrors evaluate.py logic."""
    sel = selections.copy()
    sel["PEAKID"] = sel["PEAK_TYPE"].map({"OFF": 0, "ON": 1})
    # Handle both TARGET_MONTH and MONTH column names
    if "TARGET_MONTH" in sel.columns:
        sel = sel.rename(columns={"TARGET_MONTH": "MONTH"})
    sel = sel[sel["MONTH"].isin(months)].drop_duplicates(
        subset=["MONTH","PEAKID","EID"]
    )
    sel["IS_SELECTED"] = True

    truth_m = truth[truth["MONTH"].isin(months)].copy()
    truth_m["IS_PROFITABLE"] = truth_m["is_profitable"].astype(bool)

    merged = truth_m.merge(
        sel[["EID","MONTH","PEAKID","IS_SELECTED"]],
        on=["EID","MONTH","PEAKID"], how="outer"
    )
    merged["IS_PROFITABLE"] = merged["IS_PROFITABLE"].fillna(False)
    merged["IS_SELECTED"]   = merged["IS_SELECTED"].fillna(False)
    merged["PR"]     = merged["PR"].fillna(0)
    merged["C"]      = merged["C"].fillna(0)
    merged["profit"] = merged["PR"] - merged["C"]

    results = {}
    for pid, pname in [(0,"OFF"),(1,"ON")]:
        sub = merged[merged["PEAKID"] == pid]
        tp  = ((sub["IS_SELECTED"]) &  (sub["IS_PROFITABLE"])).sum()
        fp  = ((sub["IS_SELECTED"]) & ~(sub["IS_PROFITABLE"])).sum()
        fn  = (~sub["IS_SELECTED"]  &   sub["IS_PROFITABLE"]).sum()
        pr  = tp/(tp+fp) if (tp+fp) > 0 else 0
        rec = tp/(tp+fn) if (tp+fn) > 0 else 0
        f1  = 2*pr*rec/(pr+rec) if (pr+rec) > 0 else 0
        results[pname] = {
            "TP":int(tp),"FP":int(fp),"FN":int(fn),
            "Precision":round(pr,4),"Recall":round(rec,4),"F1":round(f1,4),
        }

    f1_avg     = (results["OFF"]["F1"] + results["ON"]["F1"]) / 2
    sel_rows   = merged[merged["IS_SELECTED"]]
    tot_profit = sel_rows["profit"].sum()
    n_prof     = int((sel_rows["profit"] > 0).sum())
    n_lose     = int((sel_rows["profit"] <= 0).sum())
    n_sel      = int(merged["IS_SELECTED"].sum())
    base_rate  = truth_m["is_profitable"].mean()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        for pname, m in results.items():
            print(f"  {pname:3s}: TP={m['TP']:,}  FP={m['FP']:,}  FN={m['FN']:,}"
                  f"  P={m['Precision']:.4f}  R={m['Recall']:.4f}"
                  f"  F1={m['F1']:.4f}")
        print(f"  F1 avg:      {f1_avg:.4f}")
        print(f"  Net profit:  {tot_profit:>12,.2f}")
        print(f"  Selections:  {n_sel:,}  Profitable: {n_prof:,}  "
              f"Losing: {n_lose:,}")
        if n_sel > 0:
            prec = n_prof / n_sel
            print(f"  Precision:   {100*prec:.1f}%  "
                  f"(base rate: {100*base_rate:.1f}%  "
                  f"lift: {prec/base_rate:.1f}x)")

    return {
        "f1_off": results["OFF"]["F1"],
        "f1_on":  results["ON"]["F1"],
        "f1_avg": round(f1_avg, 4),
        "profit": round(tot_profit, 2),
        "n_prof": n_prof, "n_lose": n_lose,
        "detail": results,
    }


def evaluate_monthly(
    selections: pd.DataFrame,
    truth: pd.DataFrame,
    months: list,
):
    """Month-by-month profit breakdown."""
    print(f"\n  {'Month':>10}  {'n_prof':>8}  {'Net profit':>14}")
    print(f"  {'─'*38}")

    profits = []
    n_profs = []
    for month in sorted(months):
        col   = "MONTH" if "MONTH" in selections.columns else "TARGET_MONTH"
        sel_m = selections[selections[col] == month].copy()
        truth_m = truth[truth["MONTH"] == month].copy()
        if len(sel_m) == 0:
            continue
        r = evaluate(sel_m, truth_m, [month], verbose=False)
        flag = "  ✓" if r["profit"] > 100000 else (
               "  ✗" if r["profit"] < -100000 else "")
        print(f"  {month:>10}  {r['n_prof']:>7}/100  "
              f"{r['profit']:>14,.0f}{flag}")
        profits.append(r["profit"])
        n_profs.append(r["n_prof"])

    if profits:
        print(f"  {'─'*38}")
        print(f"  {'TOTAL':>10}  {sum(n_profs):>8}  "
              f"{sum(profits):>14,.0f}")
        print(f"  {'Std':>10}  {'':>8}  "
              f"{pd.Series(profits).std():>14,.0f}")
        print(f"  {'Best':>10}  {'':>8}  {max(profits):>14,.0f}")
        print(f"  {'Worst':>10}  {'':>8}  {min(profits):>14,.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  predict_profit.py — Final Selection Pipeline")
    print(f"  Period:  {args.start_month} → {args.end_month}")
    print(f"  Alpha:   {args.alpha}  (soft cost penalty)")
    print(f"  Select:  {args.n_select} per month")
    print("="*60)

    t0 = time.time()

    # ── Verify input files ────────────────────────────────────────────────────
    price_path = OUTPUT_DIR / "predicted_prices.parquet"
    cost_path  = OUTPUT_DIR / "predicted_costs.parquet"

    for path, name, script in [
        (price_path, "predicted_prices.parquet", "predict_price.py"),
        (cost_path,  "predicted_costs.parquet",  "cost_predictor.py"),
    ]:
        if not path.exists():
            print(f"\nERROR: {name} not found.")
            print(f"  Run: python {script} first")
            sys.exit(1)

    # ── Load predictions ──────────────────────────────────────────────────────
    print("\n[1] Loading predictions...")

    price_preds = pd.read_parquet(price_path)
    cost_preds  = pd.read_parquet(cost_path)

    target_months = get_month_range(args.start_month, args.end_month)
    price_preds   = price_preds[price_preds["TARGET_MONTH"].isin(target_months)]
    cost_preds    = cost_preds[cost_preds["TARGET_MONTH"].isin(target_months)]

    print(f"  Price: {len(price_preds):,} rows | "
          f"{price_preds['TARGET_MONTH'].nunique()} months")
    print(f"  Cost:  {len(cost_preds):,} rows | "
          f"{cost_preds['TARGET_MONTH'].nunique()} months")

    # ── Merge ─────────────────────────────────────────────────────────────────
    print("\n[2] Merging price + cost on (EID, PEAKID, TARGET_MONTH)...")

    combined = price_preds.merge(
        cost_preds[["EID","PEAKID","TARGET_MONTH","predicted_c"]],
        on=["EID","PEAKID","TARGET_MONTH"],
        how="left"
    )
    combined["predicted_c"] = combined["predicted_c"].fillna(0)

    n_with_c = (combined["predicted_c"] > 0).sum()
    print(f"  Combined rows:  {len(combined):,}")
    print(f"  With C > 0:     {n_with_c:,} ({100*n_with_c/len(combined):.1f}%)")
    print(f"  predicted_pr:   mean={combined['predicted_pr'].mean():.1f}  "
          f"max={combined['predicted_pr'].max():.1f}")
    print(f"  predicted_c:    mean={combined[combined['predicted_c']>0]['predicted_c'].mean():.1f}  "
          f"max={combined['predicted_c'].max():.1f}")

    # ── Score ─────────────────────────────────────────────────────────────────
    print(f"\n[3] Scoring with soft penalty α={args.alpha}...")

    combined = compute_scores(combined, alpha=args.alpha)

    print(f"  Score:   mean={combined['score'].mean():.1f}  "
          f"max={combined['score'].max():.1f}")
    print(f"  PR > C:  {combined['pr_exceeds_c'].sum():,} "
          f"({100*combined['pr_exceeds_c'].mean():.1f}% of candidates)")

    # ── Select ────────────────────────────────────────────────────────────────
    print(f"\n[4] Selecting top-{args.n_select} per month...")

    all_sel = []
    for tm in sorted(target_months):
        sel = select_top_n(combined, tm, n=args.n_select)
        if len(sel) > 0:
            all_sel.append(sel)

    if not all_sel:
        print("ERROR: no selections generated")
        sys.exit(1)

    selections      = pd.concat(all_sel, ignore_index=True)
    selections      = selections.drop_duplicates(
        subset=["TARGET_MONTH","PEAK_TYPE","EID"]
    )
    monthly_counts  = selections.groupby("TARGET_MONTH").size()

    # Constraint check
    violations = monthly_counts[
        (monthly_counts < MIN_SELECT_PER_MONTH) |
        (monthly_counts > MAX_SELECT_PER_MONTH)
    ]
    if len(violations) > 0:
        print(f"  WARNING: monthly constraint violations:\n"
              f"{violations.to_string()}")

    print(f"  ✓ {len(selections):,} selections | "
          f"{monthly_counts.mean():.1f} avg/month | "
          f"{selections['PEAK_TYPE'].value_counts().to_dict()}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if not args.no_validate:
        print("\n[5] Evaluating against ground truth...")

        costs = load_costs()
        pr    = load_prices_and_aggregate()

        cand_cache = OUTPUT_DIR / "sim_candidates.parquet"
        if cand_cache.exists():
            sim_candidates = pd.read_parquet(cand_cache)
        else:
            from data_loader import build_sim_candidate_universe
            sim_candidates = build_sim_candidate_universe(
                sim_monthly_paths=SIM_MONTHLY_PATHS,
                sim_daily_paths=SIM_DAILY_PATHS,
                pr=pr, costs=costs, cache_path=cand_cache,
            )

        # 2020-2023 evaluation
        eval_months = [m for m in target_months
                       if m in get_month_range("2020-01","2023-12")]
        if eval_months:
            truth = compute_ground_truth(
                pr=pr, costs=costs, sim_candidates=sim_candidates,
                months=eval_months
            )
            evaluate(selections, truth, eval_months,
                     label=f"Validation ({args.start_month} → {args.end_month})")
            evaluate_monthly(selections, truth, eval_months)

        # 2024 robustness check
        test_months = [m for m in target_months
                       if m in get_month_range("2024-01","2024-12")]
        if test_months:
            truth_2024 = compute_ground_truth(
                pr=pr, costs=costs, sim_candidates=sim_candidates,
                months=test_months
            )
            evaluate(selections, truth_2024, test_months,
                     label="2024 Robustness Check")
            evaluate_monthly(selections, truth_2024, test_months)

        if not eval_months and not test_months:
            print("  No ground truth available for this period (future months)")
    else:
        print("\n[5] Skipping evaluation (--no-validate)")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    selections.to_csv(out_path, index=False)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"  ✓ Saved → {out_path}")
    print(f"  Rows:      {len(selections):,}")
    print(f"  Months:    {monthly_counts.index.min()} → "
          f"{monthly_counts.index.max()}")
    print(f"  Runtime:   {elapsed:.1f} min")
    print(f"\n  Sample output (first 3 rows):")
    print(selections.head(3).to_string(index=False))
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()