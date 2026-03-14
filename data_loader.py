"""
data_loader.py — Data loading utilities for MAG Energy Solutions Challenge.

Strategy:
  costs     → pandas  (small monthly file, safe to load all at once)
  prices    → pandas  (load + immediately aggregate to monthly, free hourly)
  sim_monthly → Polars lazy scan (2-3 GB files, never fully loaded into RAM)
  sim_daily   → Polars lazy scan (same approach)

Confirmed schema from your Polars preview:
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
    SIM_COLS_MONTHLY, SIM_COLS_DAILY,
)


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
# GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────

def compute_ground_truth(
    pr: pd.DataFrame,
    costs: pd.DataFrame,
    months: Optional[list] = None,
) -> pd.DataFrame:
    """
    Join |PR| with costs → profit = |PR| - C, label = (profit > 0).
    Missing PR or C → 0 (implicit zero rule).

    Returns: EID, MONTH, PEAKID, PR, C, profit, is_profitable
    """
    if months is not None:
        pr    = pr[pr["MONTH"].isin(months)]
        costs = costs[costs["MONTH"].isin(months)]

    truth = pr.merge(costs, on=["EID", "MONTH", "PEAKID"], how="outer")
    truth["PR"] = truth["PR"].fillna(0).astype(float)
    truth["C"]  = truth["C"].fillna(0).astype(float)
    truth["profit"]        = truth["PR"] - truth["C"]
    truth["is_profitable"] = (truth["profit"] > 0).astype(int)

    n_total = len(truth)
    n_prof  = truth["is_profitable"].sum()
    print(f"  ✓ Ground truth: {n_total:,} opportunities | "
          f"{n_prof:,} profitable ({100*n_prof/n_total:.2f}%)\n")
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

    # Parse target month boundaries for filter
    target_start = pl.lit(target_month + "-01").str.to_datetime("%Y-%m-%d")
    # End = first day of the month after target
    tp = pd.Period(target_month, freq="M")
    next_month = str(tp + 1)
    target_end = pl.lit(next_month + "-01").str.to_datetime("%Y-%m-%d")

    print(f"  Scanning sim_monthly for {target_month} "
          f"across {len(paths)} file(s)...")

    try:
        lf = (
            pl.scan_parquet(paths)
            # Filter to only target month rows (pushed down to parquet read)
            .filter(
                (pl.col("DATETIME") >= target_start) &
                (pl.col("DATETIME") < target_end)
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

    # Days 1-7 of cutoff_month (HE convention: last valid = day8 00:00:00)
    month_start = cutoff_month + "-01"
    # Day 8 00:00:00 = HE 24 of day 7 = last valid timestamp
    tp = pd.Period(cutoff_month, freq="M")
    day8 = f"{cutoff_month}-08"

    start_dt = pl.lit(month_start).str.to_datetime("%Y-%m-%d")
    end_dt   = pl.lit(day8).str.to_datetime("%Y-%m-%d")

    print(f"  Scanning sim_daily for first 7 days of {cutoff_month}...")

    try:
        lf = (
            pl.scan_parquet(paths)
            .filter(
                (pl.col("DATETIME") >= start_dt) &
                (pl.col("DATETIME") <= end_dt)
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