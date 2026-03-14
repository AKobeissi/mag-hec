"""
data_loader.py — Data loading utilities for MAG Energy Solutions Challenge.

Strategy:
  costs       → pandas  (small monthly file, safe to load all at once)
  prices      → pandas  (aggregate to monthly immediately, free hourly data)
  sim_monthly → Polars lazy scan (2-3 GB files, never fully loaded into RAM)
  sim_daily   → Polars lazy scan (same approach)

CRITICAL — Ground Truth Universe:
  Both the prices and costs files are SPARSE (only non-zero values stored).
  An outer join of prices ∪ costs gives only ~729 rows/month and a false
  71% profit rate — because we only see EIDs that already had activity.

  The TRUE universe is all (EID, MONTH, PEAKID) in sim_monthly: ~12,000/month.
  When computed over the full sim universe:
    profitable = 518 / 12,038 ≈ 4.3%  ← matches the case doc's "< 5%"

  Fix: build sim_candidates first (all EIDs from sim), then left-join
  prices and costs onto it so the missing ones correctly get PR=0, C=0.

Confirmed schema (Polars preview):
  SCENARIOID(i8), EID(i16), DATETIME(datetime[ns]), PEAKID(i8),
  ACTIVATIONLEVEL(f32), WINDIMPACT(f32), SOLARIMPACT(f32),
  HYDROIMPACT(f32), NONRENEWBALIMPACT(f32), EXTERNALIMPACT(f32),
  LOADIMPACT(f32), TRANSMISSIONOUTAGEIMPACT(f32), PSM/PSD(f32)
"""

from pathlib import Path
from typing import Optional, List
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import polars as pl

from config import (
    COSTS_DIR, PRICES_DIR,
    SIM_MONTHLY_PATHS, SIM_DAILY_PATHS,
    COSTS_COLS, PRICES_COLS,
)

# Re-export for convenience so callers can do: from data_loader import SIM_MONTHLY_PATHS
__all__ = [
    "SIM_MONTHLY_PATHS", "SIM_DAILY_PATHS",
    "load_costs", "load_prices_and_aggregate",
    "build_sim_candidate_universe", "compute_ground_truth",
    "load_sim_monthly_for_target", "aggregate_sim_across_scenarios",
    "load_sim_daily_first7", "precompute_sim_monthly_all",
    "add_months", "get_month_range", "summarize_dataframe",
]


# ─────────────────────────────────────────────────────────────────────────────
# COSTS  (pandas — small file)
# ─────────────────────────────────────────────────────────────────────────────

def load_costs(data_dir: Path = COSTS_DIR) -> pd.DataFrame:
    """
    Load all cost parquet files.
    Returns DataFrame: EID, MONTH, PEAKID, C
    Missing (EID, MONTH, PEAKID) → cost implicitly 0.
    """
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=COSTS_COLS)
        parts.append(df)
        print(f"  costs/{f.name}: {len(df):,} rows")

    costs = pd.concat(parts, ignore_index=True)
    costs["PEAKID"] = costs["PEAKID"].astype(int)
    costs["C"]      = costs["C"].astype(float)
    costs["MONTH"]  = costs["MONTH"].astype(str)
    costs = costs.drop_duplicates(subset=["EID", "MONTH", "PEAKID"])

    print(f"\n  ✓ Costs: {len(costs):,} rows | "
          f"{costs['EID'].nunique():,} EIDs | "
          f"{costs['MONTH'].nunique()} months\n")
    return costs


# ─────────────────────────────────────────────────────────────────────────────
# PRICES → monthly |PR|  (pandas — aggregate immediately, free hourly data)
# ─────────────────────────────────────────────────────────────────────────────

def load_prices_and_aggregate(
    data_dir: Path = PRICES_DIR,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load hourly realized prices and aggregate to monthly |PR| per
    (EID, MONTH, PEAKID).  Never keeps full hourly table in memory.

    Returns DataFrame: EID, MONTH, PEAKID, PR
    where PR = |sum of PRICEREALIZED over the month|
    """
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=PRICES_COLS)
        df["DATETIME"] = pd.to_datetime(df["DATETIME"])
        df["MONTH"]    = df["DATETIME"].dt.to_period("M").astype(str)

        if start_month:
            df = df[df["MONTH"] >= start_month]
        if end_month:
            df = df[df["MONTH"] <= end_month]

        if len(df) == 0:
            print(f"  prices/{f.name}: outside date range, skipped")
            continue

        # Aggregate → free hourly data
        monthly = (
            df.groupby(["EID", "MONTH", "PEAKID"])["PRICEREALIZED"]
              .sum()
              .reset_index()
        )
        monthly["PR"] = monthly["PRICEREALIZED"].abs()
        monthly = monthly.drop(columns=["PRICEREALIZED"])

        parts.append(monthly)
        del df
        print(f"  prices/{f.name}: {len(monthly):,} (EID,MONTH,PEAKID) rows")

    if not parts:
        raise ValueError("No price data for the specified date range.")

    pr = pd.concat(parts, ignore_index=True)
    pr = pr.groupby(["EID", "MONTH", "PEAKID"])["PR"].sum().reset_index()
    pr["PEAKID"] = pr["PEAKID"].astype(int)
    pr["MONTH"]  = pr["MONTH"].astype(str)

    print(f"\n  ✓ PR monthly: {len(pr):,} rows | "
          f"{pr['EID'].nunique():,} EIDs | "
          f"{pr['MONTH'].nunique()} months\n")
    return pr


# ─────────────────────────────────────────────────────────────────────────────
# SIM CANDIDATE UNIVERSE  ← must be built BEFORE ground truth
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SIM CANDIDATE UNIVERSE  ← union of ALL 4 data sources
# ─────────────────────────────────────────────────────────────────────────────

def build_sim_candidate_universe(
    sim_monthly_paths: List[Path] = None,
    sim_daily_paths:   List[Path] = None,
    pr:                pd.DataFrame = None,
    costs:             pd.DataFrame = None,
    start_month: Optional[str] = None,
    end_month:   Optional[str] = None,
    cache_path:  Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build the FULL candidate universe: union of all 4 data sources.

    Per organizer clarification:
      "The ~5% figure is computed against the full opportunity universe —
       all (EID, MONTH, PEAKID) tuples appearing in ANY of the 4 data
       sources: prices, costs, sim_daily, and sim_monthly."

    2023 example from organizer:
      Full universe:    167,426 opportunities (~13,952/month)
      Profitable:         3,807 (2.3%)
      sim_monthly alone: ~12,038/month → sim_daily adds ~1,914/month more

    Parameters
    ----------
    sim_monthly_paths : list of sim_monthly parquet paths
    sim_daily_paths   : list of sim_daily parquet paths
    pr                : monthly PR DataFrame (already loaded) — contributes EIDs
    costs             : costs DataFrame (already loaded) — contributes EIDs
    start_month       : e.g. '2020-01'
    end_month         : e.g. '2024-12'
    cache_path        : save/load parquet cache (recommended — scans ~15+ GB)

    Returns
    -------
    DataFrame: EID, MONTH, PEAKID — one row per unique opportunity in universe
    """
    if sim_monthly_paths is None:
        from config import SIM_MONTHLY_PATHS as _mp
        sim_monthly_paths = _mp
    if sim_daily_paths is None:
        from config import SIM_DAILY_PATHS as _dp
        sim_daily_paths = _dp

    # ── Load from cache if available ─────────────────────────────────────────
    if cache_path is not None and Path(cache_path).exists():
        print(f"  Loading candidate universe from cache: {cache_path}")
        cached = pd.read_parquet(cache_path)
        if start_month:
            cached = cached[cached["MONTH"] >= start_month]
        if end_month:
            cached = cached[cached["MONTH"] <= end_month]
        n = len(cached)
        months = cached["MONTH"].nunique()
        avg = n // months if months else 0
        print(f"  ✓ {n:,} candidates | {cached['EID'].nunique():,} EIDs | "
              f"{months} months | avg {avg:,}/month\n")
        return cached

    print("  Building full candidate universe (union of all 4 sources)...")
    print("  [WARNING] First run scans ~15+ GB of parquet — will be cached.\n")

    # Date filter helper — uses Python datetime objects, not pl.lit string conversion
    # pl.lit("2020-01-01").str.to_datetime() is unreliable across Polars versions
    from datetime import datetime

    def make_date_filter(lf: pl.LazyFrame) -> pl.LazyFrame:
        filters = []
        if start_month:
            start_dt = datetime.strptime(start_month + "-01", "%Y-%m-%d")
            filters.append(pl.col("DATETIME") >= pl.lit(start_dt))
        if end_month:
            tp     = pd.Period(end_month, freq="M") + 1
            end_dt = datetime.strptime(str(tp) + "-01", "%Y-%m-%d")
            filters.append(pl.col("DATETIME") < pl.lit(end_dt))
        if filters:
            combined = filters[0]
            for f in filters[1:]:
                combined = combined & f
            return lf.filter(combined)
        return lf

    all_parts: List[pd.DataFrame] = []

    # ── Source 1: sim_monthly ─────────────────────────────────────────────────
    if sim_monthly_paths:
        print("  Scanning sim_monthly...")
        existing_paths = [p for p in sim_monthly_paths if Path(p).exists()]
        if not existing_paths:
            print("  WARNING: no sim_monthly files found on disk")
        else:
            # Scan one file at a time to handle schema differences
            # (e.g. datetime[ns] vs datetime[us] across years)
            monthly_parts = []
            for p in existing_paths:
                try:
                    df = (
                        pl.scan_parquet(p)
                        .select(["EID", "DATETIME", "PEAKID"])
                    )
                    df = make_date_filter(df)
                    chunk = (
                        df.with_columns(
                            pl.col("DATETIME")
                              .dt.strftime("%Y-%m")
                              .alias("MONTH")
                        )
                        .select(["EID", "MONTH", "PEAKID"])
                        .unique()
                        .collect()
                        .to_pandas()
                    )
                    monthly_parts.append(chunk)
                    print(f"    ✓ {Path(p).name}: "
                          f"{len(chunk):,} unique (EID,MONTH,PEAKID)")
                except Exception as e:
                    print(f"    ✗ {Path(p).name}: ERROR — {e}")

            if monthly_parts:
                df_monthly = (
                    pd.concat(monthly_parts, ignore_index=True)
                    .drop_duplicates()
                )
                print(f"    sim_monthly total: {len(df_monthly):,} unique rows")
                all_parts.append(df_monthly)
    else:
        print("  WARNING: no sim_monthly files configured")

    # ── Source 2: sim_daily ───────────────────────────────────────────────────
    if sim_daily_paths:
        print("  Scanning sim_daily...")
        existing_paths = [p for p in sim_daily_paths if Path(p).exists()]
        if not existing_paths:
            print("  WARNING: no sim_daily files found on disk")
        else:
            daily_parts = []
            for p in existing_paths:
                try:
                    df = (
                        pl.scan_parquet(p)
                        .select(["EID", "DATETIME", "PEAKID"])
                    )
                    df = make_date_filter(df)
                    chunk = (
                        df.with_columns(
                            pl.col("DATETIME")
                              .dt.strftime("%Y-%m")
                              .alias("MONTH")
                        )
                        .select(["EID", "MONTH", "PEAKID"])
                        .unique()
                        .collect()
                        .to_pandas()
                    )
                    daily_parts.append(chunk)
                    print(f"    ✓ {Path(p).name}: "
                          f"{len(chunk):,} unique (EID,MONTH,PEAKID)")
                except Exception as e:
                    print(f"    ✗ {Path(p).name}: ERROR — {e}")

            if daily_parts:
                df_daily = (
                    pd.concat(daily_parts, ignore_index=True)
                    .drop_duplicates()
                )
                print(f"    sim_daily total: {len(df_daily):,} unique rows")
                all_parts.append(df_daily)
    else:
        print("  WARNING: no sim_daily files configured")

    # ── Source 3: prices (already loaded as monthly PR) ───────────────────────
    if pr is not None and len(pr) > 0:
        df = pr[["EID", "MONTH", "PEAKID"]].drop_duplicates().copy()
        if start_month:
            df = df[df["MONTH"] >= start_month]
        if end_month:
            df = df[df["MONTH"] <= end_month]
        print(f"    prices:      {len(df):,} unique (EID,MONTH,PEAKID)")
        all_parts.append(df)

    # ── Source 4: costs ───────────────────────────────────────────────────────
    if costs is not None and len(costs) > 0:
        df = costs[["EID", "MONTH", "PEAKID"]].drop_duplicates().copy()
        if start_month:
            df = df[df["MONTH"] >= start_month]
        if end_month:
            df = df[df["MONTH"] <= end_month]
        print(f"    costs:       {len(df):,} unique (EID,MONTH,PEAKID)")
        all_parts.append(df)

    # ── Union all 4 sources ───────────────────────────────────────────────────
    candidates = (
        pd.concat(all_parts, ignore_index=True)
        .drop_duplicates(subset=["EID", "MONTH", "PEAKID"])
        .sort_values(["MONTH", "EID", "PEAKID"])
        .reset_index(drop=True)
    )

    candidates["EID"]    = candidates["EID"].astype(int)
    candidates["PEAKID"] = candidates["PEAKID"].astype(int)
    candidates["MONTH"]  = candidates["MONTH"].astype(str)

    n      = len(candidates)
    months = candidates["MONTH"].nunique()
    avg    = n // months if months else 0

    print(f"\n  ✓ Full universe: {n:,} candidates | "
          f"{candidates['EID'].nunique():,} unique EIDs | "
          f"{months} months | avg {avg:,}/month")

    # Sanity check against organizer's 2023 number
    n_2023 = len(candidates[
        (candidates["MONTH"] >= "2023-01") &
        (candidates["MONTH"] <= "2023-12")
    ])
    print(f"  2023 check: {n_2023:,} opportunities "
          f"(organizer says 167,426 for 2023)")

    # ── Save cache ────────────────────────────────────────────────────────────
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        candidates.to_parquet(cache_path, index=False)
        print(f"  Cached → {cache_path}\n")

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH  (corrected — uses sim universe as denominator)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ground_truth(
    pr: pd.DataFrame,
    costs: pd.DataFrame,
    sim_candidates: pd.DataFrame,
    months: Optional[list] = None,
) -> pd.DataFrame:
    """
    Correct ground truth computation using the full sim universe.

    Universe  = all (EID, MONTH, PEAKID) from sim_monthly     (~12,000/month)
    PR        = left-joined from prices; missing → 0           (sparse file)
    C         = left-joined from costs;  missing → 0           (sparse file)
    profit    = |PR| - C
    profitable = profit > 0

    This gives ~4-5% profitability, matching the case document.
    The old outer-join approach gave 71% because it only saw the
    pre-filtered subset of EIDs that already had non-zero prices or costs.

    Parameters
    ----------
    pr             : monthly |PR| per (EID, MONTH, PEAKID) from load_prices_and_aggregate
    costs          : monthly C  per (EID, MONTH, PEAKID) from load_costs
    sim_candidates : full universe from build_sim_candidate_universe
    months         : optional list to filter to specific months

    Returns
    -------
    DataFrame: EID, MONTH, PEAKID, PR, C, profit, is_profitable
    """
    if months is not None:
        sim_candidates = sim_candidates[sim_candidates["MONTH"].isin(months)]
        pr             = pr[pr["MONTH"].isin(months)]
        costs          = costs[costs["MONTH"].isin(months)]

    # Start from the full sim universe — this is the TRUE denominator
    truth = sim_candidates[["EID", "MONTH", "PEAKID"]].copy()

    # Left join prices — EIDs missing from prices file have PR = 0
    truth = truth.merge(pr, on=["EID", "MONTH", "PEAKID"], how="left")
    truth["PR"] = truth["PR"].fillna(0).astype(float)

    # Left join costs — EIDs missing from costs file have C = 0
    truth = truth.merge(costs, on=["EID", "MONTH", "PEAKID"], how="left")
    truth["C"] = truth["C"].fillna(0).astype(float)

    truth["profit"]        = truth["PR"] - truth["C"]
    truth["is_profitable"] = (truth["profit"] > 0).astype(int)

    n_total = len(truth)
    n_prof  = truth["is_profitable"].sum()
    rate    = 100 * n_prof / n_total if n_total > 0 else 0

    print(f"  ✓ Ground truth (full sim universe):")
    print(f"    {n_total:,} total opportunities | "
          f"{n_prof:,} profitable ({rate:.2f}%)")
    print(f"    {truth['EID'].nunique():,} unique EIDs | "
          f"{truth['MONTH'].nunique()} months")

    # Breakdown by PEAKID
    by_peak = truth.groupby("PEAKID")["is_profitable"].agg(["sum","mean","count"])
    by_peak.columns = ["n_profitable","rate","n_total"]
    by_peak["rate_pct"] = (by_peak["rate"] * 100).round(2)
    print(f"\n    By PEAKID:\n{by_peak.to_string()}\n")

    return truth


# ─────────────────────────────────────────────────────────────────────────────
# SIM_MONTHLY via Polars lazy scan
# ─────────────────────────────────────────────────────────────────────────────

def load_sim_monthly_for_target(
    target_month: str,
    paths: List[Path] = SIM_MONTHLY_PATHS,
) -> Optional[pd.DataFrame]:
    """
    Load and aggregate sim_monthly for a specific target month (M+1).

    Uses Polars lazy scan → filters are pushed down to the parquet read,
    so only the rows for target_month are actually loaded into RAM.

    Parameters
    ----------
    target_month : str  e.g. '2021-02'  — this is the month we're predicting

    Returns
    -------
    pandas DataFrame with columns:
        EID, PEAKID, SCENARIOID,
        psm_abs_sum, psm_sum, act_mean, act_max, act_std, act_positive_frac,
        wind_mean, solar_mean, hydro_mean, nonrenew_mean, external_mean,
        load_mean, transmission_mean, n_hours
    Returns None if no files available.
    """
    if not paths:
        print("  WARNING: no sim_monthly files found — check SIM_MONTHLY_PATHS in config.py")
        return None

    # Use Python datetime objects for reliable date filtering
    from datetime import datetime as dt
    tp       = pd.Period(target_month, freq="M")
    start_dt = dt(tp.year, tp.month, 1)
    next_tp  = tp + 1
    end_dt   = dt(next_tp.year, next_tp.month, 1)

    print(f"  Scanning sim_monthly for {target_month} "
          f"across {len(paths)} file(s)...")

    try:
        lf = (
            pl.scan_parquet(paths)
            .filter(
                (pl.col("DATETIME") >= pl.lit(start_dt)) &
                (pl.col("DATETIME") <  pl.lit(end_dt))
            )
            # Select only columns we need
            .select([
                "SCENARIOID", "EID", "PEAKID", "DATETIME",
                "PSM",
                "ACTIVATIONLEVEL",
                "WINDIMPACT", "SOLARIMPACT", "HYDROIMPACT",
                "NONRENEWBALIMPACT", "EXTERNALIMPACT",
                "LOADIMPACT", "TRANSMISSIONOUTAGEIMPACT",
            ])
            # Aggregate per (EID, PEAKID, SCENARIOID)
            .group_by(["EID", "PEAKID", "SCENARIOID"])
            .agg([
                pl.col("PSM").sum().abs().alias("psm_abs_sum"),
                pl.col("PSM").sum().alias("psm_sum"),
                pl.col("ACTIVATIONLEVEL").mean().alias("act_mean"),
                pl.col("ACTIVATIONLEVEL").max().alias("act_max"),
                pl.col("ACTIVATIONLEVEL").std().alias("act_std"),
                (pl.col("ACTIVATIONLEVEL") > 0).mean().alias("act_positive_frac"),
                pl.col("WINDIMPACT").mean().alias("wind_mean"),
                pl.col("SOLARIMPACT").mean().alias("solar_mean"),
                pl.col("HYDROIMPACT").mean().alias("hydro_mean"),
                pl.col("NONRENEWBALIMPACT").mean().alias("nonrenew_mean"),
                pl.col("EXTERNALIMPACT").mean().alias("external_mean"),
                pl.col("LOADIMPACT").mean().alias("load_mean"),
                pl.col("TRANSMISSIONOUTAGEIMPACT").mean().alias("transmission_mean"),
                pl.len().alias("n_hours"),
            ])
            .collect()   # ← actual data loading happens here
        )

        result = lf.to_pandas()

        # Cast types for consistency
        result["EID"]        = result["EID"].astype(int)
        result["PEAKID"]     = result["PEAKID"].astype(int)
        result["SCENARIOID"] = result["SCENARIOID"].astype(int)

        print(f"    → {len(result):,} (EID,PEAKID,SCENARIO) rows | "
              f"{result['EID'].nunique():,} unique EIDs")
        return result

    except Exception as e:
        print(f"  ERROR loading sim_monthly for {target_month}: {e}")
        return None


def aggregate_sim_across_scenarios(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-scenario rows into one row per (EID, PEAKID).
    Computes mean/min/max across scenarios and a scenario-agreement flag.

    Input:  output of load_sim_monthly_for_target()
    Output: one row per (EID, PEAKID) ready to merge into feature matrix
    """
    if sim_df is None or len(sim_df) == 0:
        return pd.DataFrame()

    # ── Mean/min/max/std across scenarios ────────────────────────────────────
    agg = (sim_df
        .groupby(["EID", "PEAKID"])
        .agg(
            psm_abs_mean      = ("psm_abs_sum",  "mean"),
            psm_abs_min       = ("psm_abs_sum",  "min"),
            psm_abs_max       = ("psm_abs_sum",  "max"),
            psm_abs_std       = ("psm_abs_sum",  "std"),
            psm_signed_mean   = ("psm_sum",      "mean"),
            act_mean          = ("act_mean",      "mean"),
            act_max           = ("act_max",       "max"),
            act_std           = ("act_std",       "mean"),
            act_pos_frac      = ("act_positive_frac", "mean"),
            wind_mean         = ("wind_mean",     "mean"),
            solar_mean        = ("solar_mean",    "mean"),
            hydro_mean        = ("hydro_mean",    "mean"),
            nonrenew_mean     = ("nonrenew_mean", "mean"),
            external_mean     = ("external_mean", "mean"),
            load_mean         = ("load_mean",     "mean"),
            transmission_mean = ("transmission_mean", "mean"),
            n_scenarios       = ("SCENARIOID",    "nunique"),
        )
        .reset_index()
    )

    # ── Scenario agreement: do all scenarios agree on sign of PSM? ────────────
    sign_pivot = (sim_df
        .pivot_table(
            index=["EID", "PEAKID"],
            columns="SCENARIOID",
            values="psm_sum",
            aggfunc="first",
        )
        .reset_index()
    )
    scenario_cols = [c for c in sign_pivot.columns
                     if isinstance(c, (int, float, np.integer))]

    if len(scenario_cols) >= 2:
        signs = sign_pivot[scenario_cols].apply(np.sign)
        agree = (signs.nunique(axis=1) == 1).astype(int)
        sign_pivot["scenario_agree"] = agree
        agg = agg.merge(
            sign_pivot[["EID", "PEAKID", "scenario_agree"]],
            on=["EID", "PEAKID"], how="left"
        )
    else:
        agg["scenario_agree"] = 0

    # ── Sum of source-based impacts (partial contribution to ACTIVATIONLEVEL) ─
    agg["source_impact_sum"] = (
        agg["wind_mean"] + agg["solar_mean"] + agg["hydro_mean"] +
        agg["nonrenew_mean"] + agg["external_mean"]
    )

    # ── Per-scenario PSM columns (s1, s2, s3) for optional use ───────────────
    psm_pivot = (sim_df
        .pivot_table(
            index=["EID", "PEAKID"],
            columns="SCENARIOID",
            values="psm_abs_sum",
            aggfunc="first",
        )
        .reset_index()
    )
    psm_pivot.columns = (
        ["EID", "PEAKID"] +
        [f"psm_s{int(c)}" for c in psm_pivot.columns[2:]]
    )
    agg = agg.merge(psm_pivot, on=["EID", "PEAKID"], how="left")

    agg = agg.fillna(0)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# SIM_DAILY via Polars — calibration signal only
# ─────────────────────────────────────────────────────────────────────────────

def load_sim_daily_first7(
    cutoff_month: str,
    paths: List[Path] = SIM_DAILY_PATHS,
) -> Optional[pd.DataFrame]:
    """
    Load sim_daily for the first 7 days of cutoff_month (M).
    These are allowed at decision time and can be used to check
    how well the model is tracking reality this month.

    Returns pandas DataFrame: EID, PEAKID, SCENARIOID,
                               psd_abs_sum, psd_mean, n_hours
    """
    if not paths:
        return None

    # Use Python datetime for reliable filtering
    from datetime import datetime as dt
    tp       = pd.Period(cutoff_month, freq="M")
    start_dt = dt(tp.year, tp.month, 1)
    end_dt   = dt(tp.year, tp.month, 8)   # day 8 00:00:00 = HE 24 of day 7

    print(f"  Scanning sim_daily for first 7 days of {cutoff_month}...")

    try:
        lf = (
            pl.scan_parquet(paths)
            .filter(
                (pl.col("DATETIME") >= pl.lit(start_dt)) &
                (pl.col("DATETIME") <= pl.lit(end_dt))
            )
            .select(["SCENARIOID", "EID", "PEAKID", "PSD"])
            .group_by(["EID", "PEAKID", "SCENARIOID"])
            .agg([
                pl.col("PSD").sum().abs().alias("psd_abs_sum"),
                pl.col("PSD").mean().abs().alias("psd_abs_mean"),
                pl.len().alias("n_hours"),
            ])
            .collect()
        )

        result = lf.to_pandas()
        result["EID"]        = result["EID"].astype(int)
        result["PEAKID"]     = result["PEAKID"].astype(int)
        result["SCENARIOID"] = result["SCENARIOID"].astype(int)
        return result

    except Exception as e:
        print(f"  WARNING: sim_daily load failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PRE-COMPUTE: full sim_monthly aggregation across all months
# (do this once and cache — avoids re-scanning for every feature build)
# ─────────────────────────────────────────────────────────────────────────────

def precompute_sim_monthly_all(
    target_months: list,
    paths: List[Path] = SIM_MONTHLY_PATHS,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Pre-compute sim_monthly features for a list of target months.
    Saves to parquet cache so you only do this once.

    Parameters
    ----------
    target_months : list of 'YYYY-MM' strings
    cache_path    : optional path to save/load cache (e.g. 'output/sim_cache.parquet')

    Returns
    -------
    DataFrame with all (EID, PEAKID, TARGET_MONTH) sim features
    """
    if cache_path is not None and Path(cache_path).exists():
        print(f"  Loading sim_monthly cache from {cache_path}")
        cached = pd.read_parquet(cache_path)
        # Check if all requested months are already cached
        cached_months = set(cached["TARGET_MONTH"].unique())
        missing = [m for m in target_months if m not in cached_months]
        if not missing:
            print(f"  ✓ All {len(target_months)} months found in cache")
            return cached[cached["TARGET_MONTH"].isin(target_months)]
        else:
            print(f"  Cache exists but missing {len(missing)} months: "
                  f"{missing[:5]}{'...' if len(missing)>5 else ''}")
            # Fall through to compute missing months
            existing = cached[cached["TARGET_MONTH"].isin(cached_months - set(missing))]
            to_compute = missing
    else:
        existing = pd.DataFrame()
        to_compute = target_months

    results = []
    for i, tm in enumerate(to_compute):
        print(f"\n  [{i+1}/{len(to_compute)}] sim_monthly: {tm}")
        sim_raw = load_sim_monthly_for_target(tm, paths)
        if sim_raw is None or len(sim_raw) == 0:
            print(f"    → no data, skipping")
            continue
        sim_agg = aggregate_sim_across_scenarios(sim_raw)
        sim_agg["TARGET_MONTH"] = tm
        results.append(sim_agg)

    if not results and len(existing) == 0:
        print("  WARNING: no sim_monthly data computed.")
        return pd.DataFrame()

    new_data  = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    all_data  = pd.concat([existing, new_data], ignore_index=True) if len(existing) > 0 else new_data

    if cache_path is not None:
        Path(cache_path).parent.mkdir(exist_ok=True)
        all_data.to_parquet(cache_path, index=False)
        print(f"\n  ✓ Sim cache saved → {cache_path}")

    return all_data


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def add_months(month_str: str, n: int) -> str:
    """Add n months to a YYYY-MM string. Negative n goes backwards."""
    return str(pd.Period(month_str, freq="M") + n)


def get_month_range(start_month: str, end_month: str) -> list:
    """Return sorted list of 'YYYY-MM' strings between start and end inclusive."""
    return [str(m) for m in pd.period_range(start_month, end_month, freq="M")]


def summarize_dataframe(df: pd.DataFrame, name: str = "DataFrame"):
    """Print a quick summary."""
    print(f"\n── {name} ─────────────────────────────────")
    print(f"  Shape:   {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    if "MONTH" in df.columns:
        months = sorted(df["MONTH"].unique())
        print(f"  Months:  {months[0]} → {months[-1]}  ({len(months)} total)")
    if "TARGET_MONTH" in df.columns:
        months = sorted(df["TARGET_MONTH"].unique())
        print(f"  Target months: {months[0]} → {months[-1]}  ({len(months)} total)")
    if "EID" in df.columns:
        print(f"  EIDs:    {df['EID'].nunique():,} unique")
    if "is_profitable" in df.columns:
        r = df["is_profitable"].mean()
        print(f"  Profit rate: {100*r:.2f}%")
    print(f"\n{df.head(3)}\n")