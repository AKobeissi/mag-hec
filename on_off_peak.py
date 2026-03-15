# price_distribution.py
import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import load_prices_and_aggregate, get_month_range

OUTPUT_DIR = Path(r"C:\Users\akobe\OneDrive\Documents\mag-hec\output")

print("Loading monthly PR...")
pr = load_prices_and_aggregate()

# Filter to 2020-2023 only (training period)
pr = pr[pr["MONTH"] <= "2023-12"].copy()

print(f"Total rows: {len(pr):,}\n")

# ── Distribution by PEAKID ────────────────────────────────────────────────────
for pid, name in [(0, "OFF-Peak"), (1, "ON-Peak")]:
    sub = pr[pr["PEAKID"] == pid]["PR"]
    print(f"{'='*50}")
    print(f"  {name} (PEAKID={pid})")
    print(f"{'='*50}")
    print(f"  Count:           {len(sub):,}")
    print(f"  Zero PR rows:    {(sub==0).sum():,}  ({100*(sub==0).mean():.1f}%)")
    print(f"  Non-zero rows:   {(sub>0).sum():,}  ({100*(sub>0).mean():.1f}%)")

    active = sub[sub > 0]
    print(f"\n  Active EIDs only (PR > 0):")
    print(f"    Min:           {active.min():>12,.2f}")
    print(f"    5th pct:       {active.quantile(0.05):>12,.2f}")
    print(f"    25th pct:      {active.quantile(0.25):>12,.2f}")
    print(f"    Median:        {active.median():>12,.2f}")
    print(f"    Mean:          {active.mean():>12,.2f}")
    print(f"    75th pct:      {active.quantile(0.75):>12,.2f}")
    print(f"    95th pct:      {active.quantile(0.95):>12,.2f}")
    print(f"    99th pct:      {active.quantile(0.99):>12,.2f}")
    print(f"    Max:           {active.max():>12,.2f}")
    print(f"    Std dev:       {active.std():>12,.2f}")
    print(f"    Skewness:      {active.skew():>12,.3f}")

    # Buckets
    print(f"\n  PR value buckets (active EIDs):")
    buckets = [0, 100, 500, 1000, 5000, 10000, 50000, float("inf")]
    labels  = ["<100", "100-500", "500-1k", "1k-5k",
               "5k-10k", "10k-50k", ">50k"]
    for i, label in enumerate(labels):
        mask  = (active >= buckets[i]) & (active < buckets[i+1])
        count = mask.sum()
        pct   = 100 * count / len(active)
        bar   = "█" * int(pct / 2)
        print(f"    {label:>10s}: {count:>6,}  ({pct:>5.1f}%)  {bar}")
    print()

# ── Key comparison ────────────────────────────────────────────────────────────
print(f"{'='*50}")
print(f"  ON-Peak vs OFF-Peak Comparison")
print(f"{'='*50}")

on  = pr[pr["PEAKID"]==1]["PR"]
off = pr[pr["PEAKID"]==0]["PR"]

on_active  = on[on > 0]
off_active = off[off > 0]

print(f"\n  {'Metric':30s}  {'OFF-Peak':>12}  {'ON-Peak':>12}  {'Ratio ON/OFF':>12}")
print(f"  {'-'*70}")

metrics = [
    ("Activation rate (%)",
     100*(off>0).mean(), 100*(on>0).mean()),
    ("Median PR (active)",
     off_active.median(), on_active.median()),
    ("Mean PR (active)",
     off_active.mean(), on_active.mean()),
    ("95th pct PR (active)",
     off_active.quantile(0.95), on_active.quantile(0.95)),
    ("Std dev (active)",
     off_active.std(), on_active.std()),
    ("Skewness (active)",
     off_active.skew(), on_active.skew()),
]

for name, off_val, on_val in metrics:
    ratio = on_val / off_val if off_val != 0 else float("inf")
    print(f"  {name:30s}  {off_val:>12,.2f}  {on_val:>12,.2f}  {ratio:>12.3f}")

# ── Month-by-month breakdown ───────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Monthly mean PR by PEAKID (active EIDs only)")
print(f"{'='*50}")
print(f"  {'MONTH':>10}  {'OFF mean':>12}  {'ON mean':>12}  {'ON/OFF':>8}")

monthly = pr[pr["PR"] > 0].groupby(["MONTH","PEAKID"])["PR"].mean().unstack()
monthly.columns = ["OFF","ON"]
monthly["ratio"] = monthly["ON"] / monthly["OFF"]

for month, row in monthly.iterrows():
    flag = " ←" if abs(row["ratio"] - 1) > 0.3 else ""
    print(f"  {month:>10}  {row['OFF']:>12,.1f}  "
          f"{row['ON']:>12,.1f}  {row['ratio']:>8.3f}{flag}")

print(f"\n  Mean ratio ON/OFF: {monthly['ratio'].mean():.3f}")
print(f"  Std of ratio:      {monthly['ratio'].std():.3f}")
print(f"\n  → If ratio ≈ 1.0 consistently: ON and OFF behave similarly")
print(f"  → If ratio varies widely: train separate models per PEAKID")