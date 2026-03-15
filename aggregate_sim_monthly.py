"""
aggregate_sim_monthly.py — Aggregate sim_monthly features by (EID, PEAKID, MONTH).

Produces a memory-safe mean aggregation of all non-price features from
sim_monthly, then joins the result to the cost table.

The sim_monthly parquet files are 2-3 GB each.  To stay safe:
  1. We use Polars **lazy scan** — nothing is read until .collect().
  2. We select only the columns we need *before* any computation.
  3. The group_by(EID, PEAKID, MONTH).mean() runs as a streaming
     aggregation, so the full hourly data is never materialised in RAM.
  4. The output is tiny (~12k rows/month × 60 months ≈ 720k rows).

Usage:
    python aggregate_sim_monthly.py
"""

import polars as pl
import pandas as pd
from pathlib import Path
from config import (
    SIM_MONTHLY_DIR, COSTS_DIR, OUTPUT_DIR,
    YEARS,
)

# ── Paths ────────────────────────────────────────────────────────────────────
SIM_FILES = sorted(SIM_MONTHLY_DIR.glob("*.parquet"))
COST_FILES = sorted(COSTS_DIR.glob("*.parquet"))

OUTPUT_AGG  = OUTPUT_DIR / "sim_monthly_agg.parquet"
OUTPUT_JOIN = OUTPUT_DIR / "sim_monthly_with_costs.parquet"

# ── Feature columns (everything EXCEPT the price column PSM) ─────────────────
FEATURE_COLS = [
    "ACTIVATIONLEVEL",
    "WINDIMPACT",
    "SOLARIMPACT",
    "HYDROIMPACT",
    "NONRENEWBALIMPACT",
    "EXTERNALIMPACT",
    "LOADIMPACT",
    "TRANSMISSIONOUTAGEIMPACT",
]

# ── Keys ─────────────────────────────────────────────────────────────────────
GROUP_KEYS = ["EID", "PEAKID", "MONTH"]


def aggregate_sim_monthly() -> pl.DataFrame:
    """
    Lazy-scan all sim_monthly parquets, derive MONTH from DATETIME,
    drop PSM (price) and SCENARIOID, then group by (EID, PEAKID, MONTH)
    and compute the mean of every feature column.

    Returns a collected Polars DataFrame.
    """
    if not SIM_FILES:
        raise FileNotFoundError(
            f"No parquet files found in {SIM_MONTHLY_DIR}"
        )

    print(f"\n{'='*60}")
    print(f"  Aggregating sim_monthly → mean by (EID, PEAKID, MONTH)")
    print(f"{'='*60}")
    print(f"  Files: {len(SIM_FILES)}")
    for f in SIM_FILES:
        print(f"    {f.name}  ({f.stat().st_size / 1e6:.0f} MB)")

    # ── Lazy scan — nothing loaded yet ────────────────────────────────────────
    lf = (
        pl.scan_parquet(SIM_FILES)
        # Only read the columns we actually need (saves I/O + RAM)
        .select(["EID", "PEAKID", "DATETIME"] + FEATURE_COLS)
        # Derive MONTH string from DATETIME
        .with_columns(
            pl.col("DATETIME").dt.strftime("%Y-%m").alias("MONTH")
        )
        # Drop DATETIME — no longer needed after extracting MONTH
        .drop("DATETIME")
    )

    # ── Aggregate: mean of each feature per (EID, PEAKID, MONTH) ─────────────
    # This averages across ALL scenarios AND all hours in the month in one pass.
    agg_exprs = [pl.col(c).mean().alias(c) for c in FEATURE_COLS]

    agg_lf = (
        lf.group_by(GROUP_KEYS)
        .agg(agg_exprs + [pl.len().alias("n_hours")])
        .sort(GROUP_KEYS)
    )

    # ── Collect with streaming to cap memory ──────────────────────────────────
    print("\n  Collecting (streaming)...")
    agg_df = agg_lf.collect(engine="streaming")

    n_rows = len(agg_df)
    n_eids = agg_df["EID"].n_unique()
    n_months = agg_df["MONTH"].n_unique()
    print(f"\n  ✓ Aggregated: {n_rows:,} rows")
    print(f"    Unique EIDs:    {n_eids:,}")
    print(f"    Unique months:  {n_months}")
    print(f"    Rows/month avg: {n_rows // max(n_months, 1):,}")
    print(f"    Columns:        {agg_df.columns}")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    agg_df.write_parquet(OUTPUT_AGG)
    print(f"\n  Saved → {OUTPUT_AGG}")

    return agg_df


def load_costs() -> pl.DataFrame:
    """Load and concat all cost parquet files into a Polars DataFrame."""
    if not COST_FILES:
        raise FileNotFoundError(f"No parquet files in {COSTS_DIR}")

    dfs = []
    for f in COST_FILES:
        df = pl.read_parquet(f, columns=["EID", "MONTH", "PEAKID", "C"])
        dfs.append(df)
        print(f"  costs/{f.name}: {len(df):,} rows")

    costs = pl.concat(dfs).unique(subset=["EID", "MONTH", "PEAKID"])

    # Ensure MONTH is a string to match the sim aggregation
    costs = costs.with_columns(pl.col("MONTH").cast(pl.Utf8))

    print(f"\n  ✓ Costs: {len(costs):,} rows | "
          f"{costs['EID'].n_unique():,} EIDs | "
          f"{costs['MONTH'].n_unique()} months")
    return costs


def join_to_costs(agg_df: pl.DataFrame, costs: pl.DataFrame) -> pl.DataFrame:
    """
    Left-join the aggregated sim features onto the cost table
    on (EID, PEAKID, MONTH).

    The result has one row per cost entry with sim features attached.
    Missing sim features (EID not in sim_monthly) get null.
    """
    print(f"\n{'='*60}")
    print(f"  Joining sim features → costs on (EID, PEAKID, MONTH)")
    print(f"{'='*60}")

    # Cast join keys to matching types
    agg_df = agg_df.with_columns([
        pl.col("EID").cast(pl.Int16),
        pl.col("PEAKID").cast(pl.Int8),
    ])
    costs = costs.with_columns([
        pl.col("EID").cast(pl.Int16),
        pl.col("PEAKID").cast(pl.Int8),
    ])

    joined = agg_df.join(costs, on=GROUP_KEYS, how="left")

    n_matched = joined.filter(pl.col("C").is_not_null()).height
    n_total = len(joined)
    print(f"  Agg rows:           {n_total:,}")
    print(f"  Matched with cost:  {n_matched:,} ({100*n_matched/max(n_total,1):.1f}%)")
    print(f"  No cost data:       {n_total - n_matched:,}")
    print(f"  Result columns:   {joined.columns}")

    # ── Save ──────────────────────────────────────────────────────────────────
    joined.write_parquet(OUTPUT_JOIN)
    print(f"\n  Saved → {OUTPUT_JOIN}")

    return joined


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: aggregate sim_monthly
    agg_df = aggregate_sim_monthly()
    print(agg_df.head(10))

    # Step 2: load costs
    print(f"\n{'─'*60}")
    print(f"  Loading costs...")
    print(f"{'─'*60}")
    costs = load_costs()

    # Step 3: join
    joined = join_to_costs(agg_df, costs)
    print(f"\n{'─'*60}")
    print("  Preview of joined table:")
    print(f"{'─'*60}")
    print(joined.head(10))

    print(f"\n  ✓ Done. Files saved to {OUTPUT_DIR}")
