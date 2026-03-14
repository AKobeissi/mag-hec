# test_scan.py — paste and run this directly
import polars as pl
from pathlib import Path
import traceback

base = Path(r"C:\Users\akobe\OneDrive\Documents\mag-hec\data")

monthly_paths = [
    base / "sim_monthly" / "sim_monthly_2020.parquet",
    base / "sim_monthly" / "sim_monthly_2021.parquet",
    base / "sim_monthly" / "sim_monthly_2022.parquet",
    base / "sim_monthly" / "sim_monthly_2023.parquet",
    base / "sim_monthly" / "sim_monthly_2024.parquet",
]

daily_paths = [
    base / "sim_daily" / "sim_daily_2020.parquet",
    base / "sim_daily" / "sim_daily_2021.parquet",
    base / "sim_daily" / "sim_daily_2022.parquet",
    base / "sim_daily" / "sim_daily_2023.parquet",
    base / "sim_daily" / "sim_daily_2024.parquet",
]

# Step 1: check files exist
print("=== FILE CHECK ===")
for p in monthly_paths + daily_paths:
    exists = p.exists()
    size   = f"{p.stat().st_size/1e9:.2f} GB" if exists else "MISSING"
    print(f"  {'✓' if exists else '✗'} {p.name:35s} {size}")

# Step 2: try scanning sim_monthly one file at a time
print("\n=== SCAN sim_monthly ONE FILE AT A TIME ===")
for p in monthly_paths:
    if not p.exists():
        print(f"  SKIP {p.name} — not found")
        continue
    try:
        result = (
            pl.scan_parquet(p)
            .select(["EID", "DATETIME", "PEAKID"])
            .limit(10)
            .collect()
        )
        print(f"  ✓ {p.name} — ok, schema: {result.schema}")
    except Exception as e:
        print(f"  ✗ {p.name} — ERROR: {e}")
        traceback.print_exc()

# Step 3: try scanning sim_daily one file at a time
print("\n=== SCAN sim_daily ONE FILE AT A TIME ===")
for p in daily_paths:
    if not p.exists():
        print(f"  SKIP {p.name} — not found")
        continue
    try:
        result = (
            pl.scan_parquet(p)
            .select(["EID", "DATETIME", "PEAKID"])
            .limit(10)
            .collect()
        )
        print(f"  ✓ {p.name} — ok, schema: {result.schema}")
    except Exception as e:
        print(f"  ✗ {p.name} — ERROR: {e}")
        traceback.print_exc()

# Step 4: try scanning ALL monthly together (this is what crashed)
print("\n=== SCAN ALL sim_monthly TOGETHER ===")
existing = [p for p in monthly_paths if p.exists()]
try:
    result = (
        pl.scan_parquet(existing)
        .select(["EID", "DATETIME", "PEAKID"])
        .limit(10)
        .collect()
    )
    print(f"  ✓ Combined scan ok — {len(result)} rows")
except Exception as e:
    print(f"  ✗ Combined scan ERROR: {e}")
    traceback.print_exc()