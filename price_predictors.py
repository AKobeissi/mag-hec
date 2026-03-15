# """
# predict_price.py — Predict |PR_{M+1}| using historical prices + sim scenarios.

# Simple approach:
#   Features = historical realized |PR| + PSM from 3 scenarios for M+1
#   Target   = |PR_{M+1}| (regression)
#   Model    = LightGBM regressor trained on 2020-2022

# Anti-leakage:
#   For each target month M+1:
#     - Historical PR: only months < M+1  (realized, past only)
#     - PSM:           sim_monthly for M+1 (explicitly allowed, produced before day 7 of M)
#     - C for M+1:     FORBIDDEN           (predicted separately in cost_predictor.py)

# Usage:
#     python predict_price.py
#     python predict_price.py --start-month 2023-01 --end-month 2023-12
# """

# import gc
# import sys
# import time
# import warnings
# from datetime import datetime as dt
# from pathlib import Path

# warnings.filterwarnings("ignore")

# import numpy as np
# import pandas as pd
# import polars as pl

# SCRIPT_DIR = Path(__file__).resolve().parent
# ROOT_DIR   = SCRIPT_DIR.parent if SCRIPT_DIR.name in ("analysis","src") else SCRIPT_DIR
# sys.path.insert(0, str(ROOT_DIR))

# from config import (
#     OUTPUT_DIR, SIM_MONTHLY_PATHS, SIM_DAILY_PATHS, RANDOM_SEED
# )
# from data_loader import (
#     load_costs,
#     load_prices_and_aggregate,
#     build_sim_candidate_universe,
#     compute_ground_truth,
#     get_month_range,
#     add_months,
#     SIM_MONTHLY_PATHS,
#     SIM_DAILY_PATHS,
# )

# try:
#     import lightgbm as lgb
#     LGBM_AVAILABLE = True
# except ImportError:
#     LGBM_AVAILABLE = False
#     print("WARNING: pip install lightgbm")

# OUTPUT_DIR.mkdir(exist_ok=True)
# PRICE_PRED_CACHE = OUTPUT_DIR / "predicted_prices.parquet"
# SIM_FEAT_CACHE   = OUTPUT_DIR / "sim_monthly_cache.parquet"


# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 1: LOAD PSM FOR ONE MONTH (memory-safe)
# # ─────────────────────────────────────────────────────────────────────────────

# def load_psm_one_month(target_month: str) -> pd.DataFrame:
#     """
#     Load sim_monthly for ONE month using Polars lazy scan.
#     Never loads a full 3GB file — filter is pushed to parquet read.

#     Returns DataFrame: EID, PEAKID, SCENARIOID,
#                        psm_abs_sum, act_mean, act_pos_frac,
#                        wind_mean, solar_mean, hydro_mean,
#                        nonrenew_mean, external_mean,
#                        load_mean, transmission_mean
#     """
#     if not SIM_MONTHLY_PATHS:
#         return pd.DataFrame()

#     tp       = pd.Period(target_month, freq="M")
#     start_dt = dt(tp.year, tp.month, 1)
#     end_dt   = dt((tp + 1).year, (tp + 1).month, 1)

#     try:
#         result = (
#             pl.scan_parquet(SIM_MONTHLY_PATHS)
#             .filter(
#                 (pl.col("DATETIME") >= pl.lit(start_dt)) &
#                 (pl.col("DATETIME") <  pl.lit(end_dt))
#             )
#             .select([
#                 "EID", "PEAKID", "SCENARIOID", "PSM",
#                 "ACTIVATIONLEVEL",
#                 "WINDIMPACT", "SOLARIMPACT", "HYDROIMPACT",
#                 "NONRENEWBALIMPACT", "EXTERNALIMPACT",
#                 "LOADIMPACT", "TRANSMISSIONOUTAGEIMPACT",
#             ])
#             .group_by(["EID", "PEAKID", "SCENARIOID"])
#             .agg([
#                 pl.col("PSM").sum().abs().alias("psm_abs_sum"),
#                 pl.col("PSM").sum().alias("psm_sum"),
#                 pl.col("ACTIVATIONLEVEL").mean().alias("act_mean"),
#                 pl.col("ACTIVATIONLEVEL").max().alias("act_max"),
#                 (pl.col("ACTIVATIONLEVEL") > 0).mean().alias("act_pos_frac"),
#                 pl.col("WINDIMPACT").mean().alias("wind_mean"),
#                 pl.col("SOLARIMPACT").mean().alias("solar_mean"),
#                 pl.col("HYDROIMPACT").mean().alias("hydro_mean"),
#                 pl.col("NONRENEWBALIMPACT").mean().alias("nonrenew_mean"),
#                 pl.col("EXTERNALIMPACT").mean().alias("external_mean"),
#                 pl.col("LOADIMPACT").mean().alias("load_mean"),
#                 pl.col("TRANSMISSIONOUTAGEIMPACT").mean().alias("transmission_mean"),
#             ])
#             .collect()
#             .to_pandas()
#         )

#         result["EID"]        = result["EID"].astype(np.int32)
#         result["PEAKID"]     = result["PEAKID"].astype(np.int8)
#         result["SCENARIOID"] = result["SCENARIOID"].astype(np.int8)

#         gc.collect()
#         return result

#     except Exception as e:
#         print(f"    ERROR loading {target_month}: {e}")
#         return pd.DataFrame()


# def aggregate_scenarios(sim_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Collapse 3 scenario rows into 1 row per (EID, PEAKID).
#     Keeps per-scenario PSM columns (psm_s1, psm_s2, psm_s3) as features.
#     """
#     if sim_df is None or len(sim_df) == 0:
#         return pd.DataFrame()

#     # Mean/min/max/std across scenarios
#     agg = (
#         sim_df.groupby(["EID", "PEAKID"])
#         .agg(
#             psm_abs_mean = ("psm_abs_sum", "mean"),
#             psm_abs_min  = ("psm_abs_sum", "min"),
#             psm_abs_max  = ("psm_abs_sum", "max"),
#             psm_abs_std  = ("psm_abs_sum", "std"),
#             act_mean     = ("act_mean",     "mean"),
#             act_max      = ("act_max",      "max"),
#             act_pos_frac = ("act_pos_frac", "mean"),
#             wind_mean    = ("wind_mean",    "mean"),
#             solar_mean   = ("solar_mean",   "mean"),
#             hydro_mean   = ("hydro_mean",   "mean"),
#             nonrenew_mean= ("nonrenew_mean","mean"),
#             external_mean= ("external_mean","mean"),
#             load_mean    = ("load_mean",    "mean"),
#             transmission_mean=("transmission_mean","mean"),
#         )
#         .reset_index()
#     )

#     # Per-scenario PSM columns — the model sees each scenario separately
#     try:
#         psm_pivot = sim_df.pivot_table(
#             index=["EID","PEAKID"],
#             columns="SCENARIOID",
#             values="psm_abs_sum",
#             aggfunc="first"
#         ).reset_index()
#         s_cols = [c for c in psm_pivot.columns
#                   if isinstance(c, (int, float, np.integer))]
#         psm_pivot.columns = (
#             ["EID","PEAKID"] + [f"psm_s{int(c)}" for c in s_cols]
#         )
#         agg = agg.merge(psm_pivot, on=["EID","PEAKID"], how="left")
#     except Exception:
#         for s in [1,2,3]:
#             agg[f"psm_s{s}"] = 0.0

#     # Scenario agreement flag
#     try:
#         sign_pivot = sim_df.pivot_table(
#             index=["EID","PEAKID"],
#             columns="SCENARIOID",
#             values="psm_sum",
#             aggfunc="first"
#         ).reset_index()
#         s_cols = [c for c in sign_pivot.columns
#                   if isinstance(c, (int, float, np.integer))]
#         if len(s_cols) >= 2:
#             signs = sign_pivot[s_cols].apply(np.sign)
#             agree = (signs.nunique(axis=1) == 1).astype(np.int8)
#             sign_pivot["scenario_agree"] = agree
#             agg = agg.merge(
#                 sign_pivot[["EID","PEAKID","scenario_agree"]],
#                 on=["EID","PEAKID"], how="left"
#             )
#         else:
#             agg["scenario_agree"] = 0
#     except Exception:
#         agg["scenario_agree"] = 0

#     # Sum of source-based impacts
#     agg["source_impact_sum"] = (
#         agg["wind_mean"] + agg["solar_mean"] + agg["hydro_mean"] +
#         agg["nonrenew_mean"] + agg["external_mean"]
#     )

#     return agg.fillna(0)


# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 2: BUILD SIM CACHE (one month at a time, checkpoint every 6)
# # ─────────────────────────────────────────────────────────────────────────────

# def build_sim_cache(target_months: list, rebuild: bool = False) -> pd.DataFrame:
#     """
#     Build or load sim feature cache.
#     Scans one month at a time — never loads more than ~200MB.
#     Saves checkpoint every 6 months.
#     """
#     existing = pd.DataFrame()

#     if SIM_FEAT_CACHE.exists() and not rebuild:
#         print(f"  Loading sim cache...")
#         existing = pd.read_parquet(SIM_FEAT_CACHE)
#         cached   = set(existing["TARGET_MONTH"].unique())
#         missing  = [m for m in target_months if m not in cached]
#         if not missing:
#             print(f"  ✓ All {len(target_months)} months cached")
#             return existing
#         print(f"  {len(cached)} cached, {len(missing)} to compute")
#         target_months = missing

#     print(f"\n  Scanning sim_monthly: {len(target_months)} months "
#           f"(~30-60s each)...")
#     new_parts = []
#     t0 = time.time()

#     for i, tm in enumerate(target_months):
#         t1 = time.time()
#         print(f"  [{i+1:3d}/{len(target_months)}] {tm}", end=" ... ")

#         raw = load_psm_one_month(tm)
#         if raw is None or len(raw) == 0:
#             print("no data")
#             continue

#         agg = aggregate_scenarios(raw)
#         del raw
#         gc.collect()

#         if len(agg) == 0:
#             print("empty")
#             continue

#         agg["TARGET_MONTH"] = tm
#         new_parts.append(agg)

#         elapsed   = time.time() - t1
#         remaining = (len(target_months) - i - 1) * elapsed
#         print(f"{len(agg):,} EIDs | {elapsed:.1f}s | "
#               f"~{remaining/60:.0f}min left")

#         # Checkpoint every 6 months
#         if (i + 1) % 6 == 0 and new_parts:
#             chk = pd.concat(
#                 ([existing] if len(existing) > 0 else []) + new_parts,
#                 ignore_index=True
#             )
#             chk.to_parquet(SIM_FEAT_CACHE, index=False)
#             print(f"    ✓ Checkpoint ({len(chk):,} rows)")
#             del chk
#             gc.collect()

#     if not new_parts:
#         return existing

#     final = pd.concat(
#         ([existing] if len(existing) > 0 else []) + new_parts,
#         ignore_index=True
#     )
#     final.to_parquet(SIM_FEAT_CACHE, index=False)
#     print(f"\n  ✓ Sim cache: {len(final):,} rows | "
#           f"{final['TARGET_MONTH'].nunique()} months | "
#           f"{(time.time()-t0)/60:.1f}min")
#     return final


# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 3: BUILD FEATURES FOR PRICE PREDICTION
# # For each (EID, PEAKID, TARGET_MONTH):
# #   - Historical PR features (what has this EID done before?)
# #   - PSM features for M+1   (what does the sim predict?)
# # ─────────────────────────────────────────────────────────────────────────────

# def build_price_features(
#     candidates:   pd.DataFrame,
#     target_month: str,
#     pr_history:   pd.DataFrame,   # monthly |PR| for all past months
#     sim_agg:      pd.DataFrame,   # aggregated sim for target_month
# ) -> pd.DataFrame:
#     """
#     Build feature matrix for price prediction.

#     ANTI-LEAKAGE CHECK:
#       pr_history  must be filtered to months < target_month BEFORE calling
#       sim_agg     is for target_month (allowed — sim is produced before day 7)

#     Features:
#       Historical PR  → persistence, seasonality, activation rate
#       PSM scenarios  → sim's forward price estimate (primary signal)
#       Combined       → PSM vs historical ratio
#     """
#     same_month_ly  = add_months(target_month, -12)
#     same_month_2ly = add_months(target_month, -24)

#     # Pre-group PR history for fast lookup — O(n) not O(n²)
#     lookback = add_months(target_month, -24)
#     pr_window = pr_history[pr_history["MONTH"] >= lookback].copy()
#     pr_grouped = {
#         key: grp.sort_values("MONTH")
#         for key, grp in pr_window.groupby(["EID","PEAKID"])
#     }
#     empty_pr = pd.DataFrame(columns=pr_window.columns)

#     # Index sim by (EID, PEAKID) for O(1) lookup
#     sim_index = {}
#     if sim_agg is not None and len(sim_agg) > 0:
#         for _, row in sim_agg.iterrows():
#             sim_index[(int(row["EID"]), int(row["PEAKID"]))] = row

#     rows = []
#     for _, cand in candidates.iterrows():
#         eid    = int(cand["EID"])
#         peakid = int(cand["PEAKID"])

#         g   = pr_grouped.get((eid, peakid), empty_pr)
#         sim = sim_index.get((eid, peakid), None)

#         feat = {"EID": eid, "PEAKID": peakid}

#         # ── Historical PR features ────────────────────────────────────────────
#         if len(g) > 0:
#             # Raw averages
#             feat["pr_avg_3m"]   = float(g["PR"].tail(3).mean())
#             feat["pr_avg_6m"]   = float(g["PR"].tail(6).mean())
#             feat["pr_avg_12m"]  = float(g["PR"].tail(12).mean())
#             feat["pr_last"]     = float(g["PR"].iloc[-1])
#             feat["pr_max_12m"]  = float(g["PR"].tail(12).max())
#             feat["pr_std_12m"]  = float(g["PR"].tail(12).std()) \
#                                   if len(g) >= 2 else 0.0

#             # Activation rate — how often does this EID have PR > 0?
#             feat["pr_active_rate_12m"] = float((g["PR"] > 0).tail(12).mean())
#             feat["pr_active_rate_24m"] = float((g["PR"] > 0).tail(24).mean())
#             feat["pr_ever_active"]     = int(g["PR"].sum() > 0)
#             feat["pr_n_active_12m"]    = int((g["PR"] > 0).tail(12).sum())

#             # Average PR only when active (not diluted by zero months)
#             active = g[g["PR"] > 0]
#             feat["pr_avg_when_active"] = float(active["PR"].mean()) \
#                                          if len(active) > 0 else 0.0

#             # Seasonal features — same month last year
#             g_ly  = g[g["MONTH"] == same_month_ly]
#             g_2ly = g[g["MONTH"] == same_month_2ly]
#             feat["pr_same_month_ly"]  = float(g_ly["PR"].iloc[0])  \
#                                         if len(g_ly)  > 0 else feat["pr_avg_12m"]
#             feat["pr_same_month_2ly"] = float(g_2ly["PR"].iloc[0]) \
#                                         if len(g_2ly) > 0 else feat["pr_avg_12m"]

#             feat["n_months_history"]  = len(g)

#         else:
#             # New EID — no history at all
#             for k in ["pr_avg_3m","pr_avg_6m","pr_avg_12m","pr_last",
#                       "pr_max_12m","pr_std_12m","pr_active_rate_12m",
#                       "pr_active_rate_24m","pr_ever_active","pr_n_active_12m",
#                       "pr_avg_when_active","pr_same_month_ly",
#                       "pr_same_month_2ly","n_months_history"]:
#                 feat[k] = 0.0

#         # ── PSM features (sim scenarios for M+1) ─────────────────────────────
#         if sim is not None:
#             feat["psm_s1"]             = float(sim.get("psm_s1", 0))
#             feat["psm_s2"]             = float(sim.get("psm_s2", 0))
#             feat["psm_s3"]             = float(sim.get("psm_s3", 0))
#             feat["psm_abs_mean"]       = float(sim.get("psm_abs_mean", 0))
#             feat["psm_abs_min"]        = float(sim.get("psm_abs_min", 0))
#             feat["psm_abs_max"]        = float(sim.get("psm_abs_max", 0))
#             feat["psm_abs_std"]        = float(sim.get("psm_abs_std", 0))
#             feat["psm_scenario_agree"] = int(sim.get("scenario_agree", 0))
#             feat["act_mean"]           = float(sim.get("act_mean", 0))
#             feat["act_max"]            = float(sim.get("act_max", 0))
#             feat["act_pos_frac"]       = float(sim.get("act_pos_frac", 0))
#             feat["wind_mean"]          = float(sim.get("wind_mean", 0))
#             feat["solar_mean"]         = float(sim.get("solar_mean", 0))
#             feat["hydro_mean"]         = float(sim.get("hydro_mean", 0))
#             feat["nonrenew_mean"]      = float(sim.get("nonrenew_mean", 0))
#             feat["external_mean"]      = float(sim.get("external_mean", 0))
#             feat["load_mean"]          = float(sim.get("load_mean", 0))
#             feat["transmission_mean"]  = float(sim.get("transmission_mean", 0))
#             feat["source_impact_sum"]  = float(sim.get("source_impact_sum", 0))
#             feat["sim_in_universe"]    = 1
#         else:
#             # Not in sim universe — no forward signal
#             for k in ["psm_s1","psm_s2","psm_s3","psm_abs_mean","psm_abs_min",
#                       "psm_abs_max","psm_abs_std","psm_scenario_agree",
#                       "act_mean","act_max","act_pos_frac","wind_mean",
#                       "solar_mean","hydro_mean","nonrenew_mean","external_mean",
#                       "load_mean","transmission_mean","source_impact_sum"]:
#                 feat[k] = 0.0
#             feat["sim_in_universe"] = 0

#         # ── Combined features ─────────────────────────────────────────────────
#         pr_12m = feat.get("pr_avg_12m", 0)
#         psm    = feat.get("psm_abs_mean", 0)
#         feat["psm_vs_pr_ratio"]     = psm / (pr_12m + 1e-9)
#         feat["psm_cv"]              = (feat["psm_abs_std"] /
#                                        (psm + 1e-9))
#         feat["psm_times_act_frac"]  = psm * feat.get("act_pos_frac", 0)

#         # Temporal
#         month_num = pd.Period(target_month, freq="M").month
#         feat["month_of_year"] = month_num
#         feat["month_sin"]     = float(np.sin(2 * np.pi * month_num / 12))
#         feat["month_cos"]     = float(np.cos(2 * np.pi * month_num / 12))
#         feat["is_winter"]     = int(month_num in [11,12,1,2,3])
#         feat["is_summer"]     = int(month_num in [6,7,8,9])

#         rows.append(feat)

#     result = pd.DataFrame(rows).fillna(0)
#     result["TARGET_MONTH"] = target_month
#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 4: TRAIN PRICE MODEL AND PREDICT
# # LightGBM regressor: predicts |PR_{M+1}|
# # ─────────────────────────────────────────────────────────────────────────────

# FEATURE_COLS = [
#     # Historical PR
#     "pr_avg_3m", "pr_avg_6m", "pr_avg_12m", "pr_last", "pr_max_12m",
#     "pr_std_12m", "pr_active_rate_12m", "pr_active_rate_24m",
#     "pr_ever_active", "pr_n_active_12m", "pr_avg_when_active",
#     "pr_same_month_ly", "pr_same_month_2ly", "n_months_history",
#     # PSM scenarios
#     "psm_s1", "psm_s2", "psm_s3",
#     "psm_abs_mean", "psm_abs_min", "psm_abs_max", "psm_abs_std",
#     "psm_scenario_agree",
#     # Simulation features
#     "act_mean", "act_max", "act_pos_frac",
#     "wind_mean", "solar_mean", "hydro_mean",
#     "nonrenew_mean", "external_mean",
#     "load_mean", "transmission_mean",
#     "source_impact_sum", "sim_in_universe",
#     # Combined
#     "psm_vs_pr_ratio", "psm_cv", "psm_times_act_frac",
#     # Temporal
#     "month_of_year", "month_sin", "month_cos",
#     "is_winter", "is_summer",
#     # Peak type
#     "PEAKID",
# ]


# def train_price_model(
#     feature_matrix: pd.DataFrame,
#     truth_train: pd.DataFrame,
# ) -> "lgb.LGBMRegressor":
#     """
#     Train LightGBM regressor to predict |PR_{M+1}|.

#     Training target: log1p(PR) — log transform because PR is heavy-tailed
#     (range $0 to $230,000 — log normalises the scale)
#     Prediction:      expm1(pred) to convert back to original scale

#     Uses only truth_train (2020-2022) — never validation or test data.
#     """
#     if not LGBM_AVAILABLE:
#         raise ImportError("pip install lightgbm")

#     # Join features with labels
#     labels = truth_train[["EID","MONTH","PEAKID","PR"]].rename(
#         columns={"MONTH": "TARGET_MONTH"}
#     )
#     train = feature_matrix.merge(
#         labels, on=["EID","PEAKID","TARGET_MONTH"], how="inner"
#     )

#     feat_cols = [c for c in FEATURE_COLS if c in train.columns]
#     X = train[feat_cols].fillna(0)
#     y = np.log1p(train["PR"].fillna(0))   # log-transform

#     print(f"  Training price model:")
#     print(f"    Samples:      {len(X):,}")
#     print(f"    Features:     {len(feat_cols)}")
#     print(f"    PR > 0:       {(train['PR'] > 0).sum():,} "
#           f"({100*(train['PR']>0).mean():.1f}%)")
#     print(f"    PR stats:     "
#           f"mean={train['PR'].mean():.1f}  "
#           f"median={train['PR'].median():.1f}  "
#           f"max={train['PR'].max():.1f}")

#     model = lgb.LGBMRegressor(
#         n_estimators=500,
#         learning_rate=0.05,
#         num_leaves=63,
#         max_depth=7,
#         min_child_samples=20,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         reg_alpha=0.1,
#         reg_lambda=0.1,
#         random_state=RANDOM_SEED,
#         n_jobs=-1,
#         verbose=-1,
#     )
#     model.fit(X, y)

#     # Print top features
#     imp = pd.DataFrame({
#         "feature":    feat_cols,
#         "importance": model.feature_importances_
#     }).sort_values("importance", ascending=False)
#     print(f"\n  Top 10 features (price model):")
#     for _, row in imp.head(10).iterrows():
#         bar = "█" * int(row["importance"] / imp["importance"].max() * 25)
#         print(f"    {row['feature']:35s} {int(row['importance']):6d}  {bar}")

#     return model, feat_cols


# def predict_prices(
#     model,
#     feat_cols: list,
#     feature_matrix: pd.DataFrame,
#     target_months: list,
# ) -> pd.DataFrame:
#     """
#     Apply trained model to predict |PR| for target months.
#     Returns DataFrame with predicted_pr per (EID, PEAKID, TARGET_MONTH).
#     """
#     target_feats = feature_matrix[
#         feature_matrix["TARGET_MONTH"].isin(target_months)
#     ].copy()

#     if len(target_feats) == 0:
#         print("  ERROR: no features for target months")
#         return pd.DataFrame()

#     X = target_feats[feat_cols].fillna(0)

#     # Predict — convert back from log scale
#     pred_log = model.predict(X)
#     predicted_pr = np.expm1(pred_log)
#     predicted_pr = np.maximum(predicted_pr, 0)   # no negative prices

#     result = target_feats[["EID","PEAKID","TARGET_MONTH"]].copy()
#     result["predicted_pr"] = predicted_pr

#     # Keep useful sim columns for the decision step
#     for col in ["psm_abs_mean","psm_s1","psm_s2","psm_s3",
#                 "psm_scenario_agree","act_mean","act_pos_frac","psm_cv"]:
#         if col in target_feats.columns:
#             result[col] = target_feats[col].values

#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# # STEP 5: VALIDATE
# # ─────────────────────────────────────────────────────────────────────────────

# def validate(
#     predictions: pd.DataFrame,
#     truth: pd.DataFrame,
#     val_months: list,
# ):
#     """
#     Compare predicted |PR| with realized |PR| on validation months.
#     Uses ground truth AFTER the fact — never used for training.
#     """
#     print(f"\n{'='*55}")
#     print("  PRICE PREDICTION VALIDATION")
#     print(f"  {val_months[0]} → {val_months[-1]}")
#     print(f"{'='*55}\n")

#     preds = predictions[predictions["TARGET_MONTH"].isin(val_months)].copy()
#     truth_val = truth[truth["MONTH"].isin(val_months)].copy()
#     truth_val = truth_val.rename(columns={"MONTH":"TARGET_MONTH"})

#     merged = preds.merge(
#         truth_val[["EID","PEAKID","TARGET_MONTH","PR","is_profitable"]],
#         on=["EID","PEAKID","TARGET_MONTH"],
#         how="inner"
#     ).fillna(0)

#     print(f"  Rows: {len(merged):,}")

#     # ── Overall correlation ───────────────────────────────────────────────────
#     active = merged[merged["PR"] > 0]
#     if len(active) > 10:
#         corr_all    = merged[["PR","predicted_pr"]].corr().iloc[0,1]
#         corr_active = active[["PR","predicted_pr"]].corr().iloc[0,1]
#         mae_active  = (active["PR"] - active["predicted_pr"]).abs().mean()
#         print(f"  Correlation (all EIDs):     r = {corr_all:.3f}")
#         print(f"  Correlation (PR > 0 only):  r = {corr_active:.3f}")
#         print(f"  Mean abs error (PR > 0):    {mae_active:.2f}")

#     # ── Activation accuracy ───────────────────────────────────────────────────
#     merged["pred_active"]   = (merged["predicted_pr"] > 0).astype(int)
#     merged["actual_active"] = (merged["PR"] > 0).astype(int)
#     pred_active = merged[merged["pred_active"] == 1]
#     if len(pred_active) > 0:
#         prec = (pred_active["PR"] > 0).mean()
#         rec  = (merged[merged["actual_active"]==1]["pred_active"]).mean()
#         print(f"\n  Activation accuracy:")
#         print(f"    P(PR > 0 | predicted > 0): {100*prec:.1f}%")
#         print(f"    P(predicted > 0 | PR > 0): {100*rec:.1f}%")

#     # ── By PEAKID ─────────────────────────────────────────────────────────────
#     print(f"\n  By PEAKID (PR > 0 only):")
#     for pid, pname in [(0,"OFF"),(1,"ON")]:
#         sub = active[active["PEAKID"] == pid]
#         if len(sub) < 5:
#             continue
#         c   = sub[["PR","predicted_pr"]].corr().iloc[0,1]
#         mae = (sub["PR"] - sub["predicted_pr"]).abs().mean()
#         print(f"    {pname}: r={c:.3f}  MAE={mae:.2f}  n={len(sub):,}")

#     # ── Decision quality with actual C ───────────────────────────────────────
#     if "C" in truth_val.columns or True:
#         merged2 = merged.merge(
#             truth_val[["EID","PEAKID","TARGET_MONTH","C"]],
#             on=["EID","PEAKID","TARGET_MONTH"], how="left"
#         ).fillna(0)
#         merged2["pred_profitable"] = (
#             merged2["predicted_pr"] > merged2["C"]
#         ).astype(int)
#         if merged2["pred_profitable"].sum() > 0:
#             prec = merged2[merged2["pred_profitable"]==1]["is_profitable"].mean()
#             rec  = merged2[merged2["is_profitable"]==1]["pred_profitable"].mean()
#             f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
#             print(f"\n  Decision (predicted_pr > actual C):")
#             print(f"    Precision:  {100*prec:.1f}%")
#             print(f"    Recall:     {100*rec:.1f}%")
#             print(f"    F1:         {f1:.4f}")
#             print(f"    (Upper bound — uses actual C, not predicted C)")


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────────────────────

# def main():
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument("--start-month",   default="2020-02")
#     p.add_argument("--end-month",     default="2024-12")
#     p.add_argument("--train-end",     default="2022-12",
#                    help="Last month used for training (anti-leakage)")
#     p.add_argument("--rebuild-cache", action="store_true")
#     args = p.parse_args()

#     np.random.seed(RANDOM_SEED)

#     print("\n" + "="*55)
#     print("  Price Prediction — LightGBM on PSM + PR history")
#     print(f"  Train:   2020-02 → {args.train_end}")
#     print(f"  Predict: {args.start_month} → {args.end_month}")
#     print("="*55)

#     t0 = time.time()

#     # ── Load base data ────────────────────────────────────────────────────────
#     print("\n[1] Loading base data...")

#     costs = load_costs()
#     pr    = load_prices_and_aggregate()

#     cand_cache = OUTPUT_DIR / "sim_candidates.parquet"
#     if cand_cache.exists():
#         sim_candidates = pd.read_parquet(cand_cache)
#         sim_candidates = sim_candidates[
#             (sim_candidates["MONTH"] >= "2020-01") &
#             (sim_candidates["MONTH"] <= "2024-12")
#         ]
#     else:
#         from data_loader import build_sim_candidate_universe
#         sim_candidates = build_sim_candidate_universe(
#             sim_monthly_paths=SIM_MONTHLY_PATHS,
#             sim_daily_paths=SIM_DAILY_PATHS,
#             pr=pr, costs=costs,
#             start_month="2020-01", end_month="2024-12",
#             cache_path=cand_cache,
#         )

#     # Ground truth — 2020-2023 only
#     truth = compute_ground_truth(
#         pr=pr, costs=costs, sim_candidates=sim_candidates,
#         months=get_month_range("2020-01","2023-12")
#     )

#     # ── Build sim cache ───────────────────────────────────────────────────────
#     print("\n[2] Building sim feature cache...")
#     all_months = sorted(set(
#         get_month_range("2020-02", args.train_end) +
#         get_month_range(args.start_month, args.end_month)
#     ))
#     sim_cache_df = build_sim_cache(all_months, rebuild=args.rebuild_cache)

#     # ── Build feature matrices ────────────────────────────────────────────────
#     print(f"\n[3] Building feature matrices...")

#     all_feat_months = sorted(set(
#         get_month_range("2020-02", args.train_end) +
#         get_month_range(args.start_month, args.end_month)
#     ))

#     # Group sim cache by month for fast lookup
#     sim_by_month = {}
#     for tm, grp in sim_cache_df.groupby("TARGET_MONTH"):
#         sim_by_month[tm] = grp.drop(columns=["TARGET_MONTH"])

#     all_features = []
#     for i, tm in enumerate(all_feat_months):
#         if i % 12 == 0:
#             print(f"  Building features: {tm} "
#                   f"[{i+1}/{len(all_feat_months)}]...")

#         cands = (sim_candidates[sim_candidates["MONTH"] == tm]
#                  [["EID","PEAKID"]].drop_duplicates())
#         if len(cands) == 0:
#             continue

#         # ANTI-LEAKAGE: only use PR from months before target
#         pr_past = pr[pr["MONTH"] < tm]

#         feats = build_price_features(
#             candidates=cands,
#             target_month=tm,
#             pr_history=pr_past,
#             sim_agg=sim_by_month.get(tm, None),
#         )
#         all_features.append(feats)

#     feature_matrix = pd.concat(all_features, ignore_index=True)
#     print(f"  ✓ Feature matrix: {feature_matrix.shape}")

#     # ── Train model ───────────────────────────────────────────────────────────
#     print(f"\n[4] Training price model (target: |PR|)...")

#     train_months = get_month_range("2020-02", args.train_end)
#     truth_train  = truth[truth["MONTH"].isin(train_months)]

#     train_feats  = feature_matrix[
#         feature_matrix["TARGET_MONTH"].isin(train_months)
#     ]

#     model, feat_cols = train_price_model(train_feats, truth_train)

#     # ── Generate predictions ──────────────────────────────────────────────────
#     print(f"\n[5] Generating price predictions...")

#     target_months = get_month_range(args.start_month, args.end_month)
#     predictions   = predict_prices(model, feat_cols, feature_matrix, target_months)

#     print(f"  ✓ Predictions: {len(predictions):,} rows")
#     print(f"  predicted_pr stats:")
#     print(f"    mean:   {predictions['predicted_pr'].mean():.2f}")
#     print(f"    median: {predictions['predicted_pr'].median():.2f}")
#     print(f"    > 0:    {100*(predictions['predicted_pr']>0).mean():.1f}%")
#     print(f"    max:    {predictions['predicted_pr'].max():.2f}")

#     # ── Validate on known months ──────────────────────────────────────────────
#     truth_months = set(truth["MONTH"].unique())
#     val_months   = [m for m in target_months if m in truth_months]
#     if val_months:
#         validate(predictions, truth, val_months)

#     # ── Save ──────────────────────────────────────────────────────────────────
#     predictions.to_parquet(PRICE_PRED_CACHE, index=False)

#     print(f"\n{'='*55}")
#     print(f"  ✓ Saved → {PRICE_PRED_CACHE}")
#     print(f"  Months:   {predictions['TARGET_MONTH'].nunique()}")
#     print(f"  EIDs:     {predictions['EID'].nunique():,}")
#     print(f"  Runtime:  {(time.time()-t0)/60:.1f} min")
#     print(f"""
#   Next step — combine with cost predictions:
#     predicted_pr   (this file)     →  output/predicted_prices.parquet
#     predicted_c    (cost_predictor) →  output/predicted_costs.parquet

#   Decision: select if predicted_pr - predicted_c > 0
#   Run: python predict_profit.py to generate opportunities.csv
# """)


# if __name__ == "__main__":
#     main()


"""
predict_price.py — Predict |PR_{M+1}| using historical prices + sim scenarios.

Simple approach:
  Features = historical realized |PR| + PSM from 3 scenarios for M+1
  Target   = |PR_{M+1}| (regression)
  Model    = LightGBM regressor trained on 2020-2022

Anti-leakage:
  For each target month M+1:
    - Historical PR: only months < M+1  (realized, past only)
    - PSM:           sim_monthly for M+1 (explicitly allowed, produced before day 7 of M)
    - C for M+1:     FORBIDDEN           (predicted separately in cost_predictor.py)

Usage:
    python predict_price.py
    python predict_price.py --start-month 2023-01 --end-month 2023-12
"""

import gc
import sys
import time
import warnings
from datetime import datetime as dt
from pathlib import Path

warnings.filterwarnings("ignore")

from lightgbm import train
import numpy as np
import pandas as pd
import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent if SCRIPT_DIR.name in ("analysis","src") else SCRIPT_DIR
sys.path.insert(0, str(ROOT_DIR))

from config import (
    OUTPUT_DIR, SIM_MONTHLY_PATHS, SIM_DAILY_PATHS, RANDOM_SEED
)
from data_loader import (
    load_costs,
    load_prices_and_aggregate,
    build_sim_candidate_universe,
    compute_ground_truth,
    get_month_range,
    add_months,
    SIM_MONTHLY_PATHS,
    SIM_DAILY_PATHS,
)

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("WARNING: pip install lightgbm")

OUTPUT_DIR.mkdir(exist_ok=True)
PRICE_PRED_CACHE = OUTPUT_DIR / "predicted_prices.parquet"
SIM_FEAT_CACHE   = OUTPUT_DIR / "sim_monthly_cache.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD PSM FOR ONE MONTH (memory-safe)
# ─────────────────────────────────────────────────────────────────────────────

def load_psm_one_month(target_month: str) -> pd.DataFrame:
    """
    Load sim_monthly for ONE month using Polars lazy scan.
    Never loads a full 3GB file — filter is pushed to parquet read.

    Returns DataFrame: EID, PEAKID, SCENARIOID,
                       psm_abs_sum, act_mean, act_pos_frac,
                       wind_mean, solar_mean, hydro_mean,
                       nonrenew_mean, external_mean,
                       load_mean, transmission_mean
    """
    if not SIM_MONTHLY_PATHS:
        return pd.DataFrame()

    tp       = pd.Period(target_month, freq="M")
    start_dt = dt(tp.year, tp.month, 1)
    end_dt   = dt((tp + 1).year, (tp + 1).month, 1)

    try:
        result = (
            pl.scan_parquet(SIM_MONTHLY_PATHS)
            .filter(
                (pl.col("DATETIME") >= pl.lit(start_dt)) &
                (pl.col("DATETIME") <  pl.lit(end_dt))
            )
            .select([
                "EID", "PEAKID", "SCENARIOID", "PSM",
                "ACTIVATIONLEVEL",
                "WINDIMPACT", "SOLARIMPACT", "HYDROIMPACT",
                "NONRENEWBALIMPACT", "EXTERNALIMPACT",
                "LOADIMPACT", "TRANSMISSIONOUTAGEIMPACT",
            ])
            .group_by(["EID", "PEAKID", "SCENARIOID"])
            .agg([
                pl.col("PSM").sum().abs().alias("psm_abs_sum"),
                pl.col("PSM").sum().alias("psm_sum"),
                pl.col("ACTIVATIONLEVEL").mean().alias("act_mean"),
                pl.col("ACTIVATIONLEVEL").max().alias("act_max"),
                (pl.col("ACTIVATIONLEVEL") > 0).mean().alias("act_pos_frac"),
                pl.col("WINDIMPACT").mean().alias("wind_mean"),
                pl.col("SOLARIMPACT").mean().alias("solar_mean"),
                pl.col("HYDROIMPACT").mean().alias("hydro_mean"),
                pl.col("NONRENEWBALIMPACT").mean().alias("nonrenew_mean"),
                pl.col("EXTERNALIMPACT").mean().alias("external_mean"),
                pl.col("LOADIMPACT").mean().alias("load_mean"),
                pl.col("TRANSMISSIONOUTAGEIMPACT").mean().alias("transmission_mean"),
            ])
            .collect()
            .to_pandas()
        )

        result["EID"]        = result["EID"].astype(np.int32)
        result["PEAKID"]     = result["PEAKID"].astype(np.int8)
        result["SCENARIOID"] = result["SCENARIOID"].astype(np.int8)

        gc.collect()
        return result

    except Exception as e:
        print(f"    ERROR loading {target_month}: {e}")
        return pd.DataFrame()


def aggregate_scenarios(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse 3 scenario rows into 1 row per (EID, PEAKID).
    Keeps per-scenario PSM columns (psm_s1, psm_s2, psm_s3) as features.
    """
    if sim_df is None or len(sim_df) == 0:
        return pd.DataFrame()

    # Mean/min/max/std across scenarios
    agg = (
        sim_df.groupby(["EID", "PEAKID"])
        .agg(
            psm_abs_mean = ("psm_abs_sum", "mean"),
            psm_abs_min  = ("psm_abs_sum", "min"),
            psm_abs_max  = ("psm_abs_sum", "max"),
            psm_abs_std  = ("psm_abs_sum", "std"),
            act_mean     = ("act_mean",     "mean"),
            act_max      = ("act_max",      "max"),
            act_pos_frac = ("act_pos_frac", "mean"),
            wind_mean    = ("wind_mean",    "mean"),
            solar_mean   = ("solar_mean",   "mean"),
            hydro_mean   = ("hydro_mean",   "mean"),
            nonrenew_mean= ("nonrenew_mean","mean"),
            external_mean= ("external_mean","mean"),
            load_mean    = ("load_mean",    "mean"),
            transmission_mean=("transmission_mean","mean"),
        )
        .reset_index()
    )

    # Per-scenario PSM columns — the model sees each scenario separately
    try:
        psm_pivot = sim_df.pivot_table(
            index=["EID","PEAKID"],
            columns="SCENARIOID",
            values="psm_abs_sum",
            aggfunc="first"
        ).reset_index()
        s_cols = [c for c in psm_pivot.columns
                  if isinstance(c, (int, float, np.integer))]
        psm_pivot.columns = (
            ["EID","PEAKID"] + [f"psm_s{int(c)}" for c in s_cols]
        )
        agg = agg.merge(psm_pivot, on=["EID","PEAKID"], how="left")
    except Exception:
        for s in [1,2,3]:
            agg[f"psm_s{s}"] = 0.0

    # Scenario agreement flag
    try:
        sign_pivot = sim_df.pivot_table(
            index=["EID","PEAKID"],
            columns="SCENARIOID",
            values="psm_sum",
            aggfunc="first"
        ).reset_index()
        s_cols = [c for c in sign_pivot.columns
                  if isinstance(c, (int, float, np.integer))]
        if len(s_cols) >= 2:
            signs = sign_pivot[s_cols].apply(np.sign)
            agree = (signs.nunique(axis=1) == 1).astype(np.int8)
            sign_pivot["scenario_agree"] = agree
            agg = agg.merge(
                sign_pivot[["EID","PEAKID","scenario_agree"]],
                on=["EID","PEAKID"], how="left"
            )
        else:
            agg["scenario_agree"] = 0
    except Exception:
        agg["scenario_agree"] = 0

    # Sum of source-based impacts
    agg["source_impact_sum"] = (
        agg["wind_mean"] + agg["solar_mean"] + agg["hydro_mean"] +
        agg["nonrenew_mean"] + agg["external_mean"]
    )

    return agg.fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: BUILD SIM CACHE (one month at a time, checkpoint every 6)
# ─────────────────────────────────────────────────────────────────────────────

def build_sim_cache(target_months: list, rebuild: bool = False) -> pd.DataFrame:
    """
    Build or load sim feature cache.
    Scans one month at a time — never loads more than ~200MB.
    Saves checkpoint every 6 months.
    """
    existing = pd.DataFrame()

    if SIM_FEAT_CACHE.exists() and not rebuild:
        print(f"  Loading sim cache...")
        existing = pd.read_parquet(SIM_FEAT_CACHE)
        cached   = set(existing["TARGET_MONTH"].unique())
        missing  = [m for m in target_months if m not in cached]
        if not missing:
            print(f"  ✓ All {len(target_months)} months cached")
            return existing
        print(f"  {len(cached)} cached, {len(missing)} to compute")
        target_months = missing

    print(f"\n  Scanning sim_monthly: {len(target_months)} months "
          f"(~30-60s each)...")
    new_parts = []
    t0 = time.time()

    for i, tm in enumerate(target_months):
        t1 = time.time()
        print(f"  [{i+1:3d}/{len(target_months)}] {tm}", end=" ... ")

        raw = load_psm_one_month(tm)
        if raw is None or len(raw) == 0:
            print("no data")
            continue

        agg = aggregate_scenarios(raw)
        del raw
        gc.collect()

        if len(agg) == 0:
            print("empty")
            continue

        agg["TARGET_MONTH"] = tm
        new_parts.append(agg)

        elapsed   = time.time() - t1
        remaining = (len(target_months) - i - 1) * elapsed
        print(f"{len(agg):,} EIDs | {elapsed:.1f}s | "
              f"~{remaining/60:.0f}min left")

        # Checkpoint every 6 months
        if (i + 1) % 6 == 0 and new_parts:
            chk = pd.concat(
                ([existing] if len(existing) > 0 else []) + new_parts,
                ignore_index=True
            )
            chk.to_parquet(SIM_FEAT_CACHE, index=False)
            print(f"    ✓ Checkpoint ({len(chk):,} rows)")
            del chk
            gc.collect()

    if not new_parts:
        return existing

    final = pd.concat(
        ([existing] if len(existing) > 0 else []) + new_parts,
        ignore_index=True
    )
    final.to_parquet(SIM_FEAT_CACHE, index=False)
    print(f"\n  ✓ Sim cache: {len(final):,} rows | "
          f"{final['TARGET_MONTH'].nunique()} months | "
          f"{(time.time()-t0)/60:.1f}min")
    return final


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: BUILD FEATURES FOR PRICE PREDICTION
# For each (EID, PEAKID, TARGET_MONTH):
#   - Historical PR features (what has this EID done before?)
#   - PSM features for M+1   (what does the sim predict?)
# ─────────────────────────────────────────────────────────────────────────────

def build_price_features(
    candidates:   pd.DataFrame,
    target_month: str,
    pr_history:   pd.DataFrame,   # monthly |PR| for all past months
    sim_agg:      pd.DataFrame,   # aggregated sim for target_month
) -> pd.DataFrame:
    """
    Build feature matrix for price prediction.

    ANTI-LEAKAGE CHECK:
      pr_history  must be filtered to months < target_month BEFORE calling
      sim_agg     is for target_month (allowed — sim is produced before day 7)

    Features:
      Historical PR  → persistence, seasonality, activation rate
      PSM scenarios  → sim's forward price estimate (primary signal)
      Combined       → PSM vs historical ratio
    """
    same_month_ly  = add_months(target_month, -12)
    same_month_2ly = add_months(target_month, -24)

    # Pre-group PR history for fast lookup — O(n) not O(n²)
    lookback = add_months(target_month, -24)
    pr_window = pr_history[pr_history["MONTH"] >= lookback].copy()
    pr_grouped = {
        key: grp.sort_values("MONTH")
        for key, grp in pr_window.groupby(["EID","PEAKID"])
    }
    empty_pr = pd.DataFrame(columns=pr_window.columns)

    # Index sim by (EID, PEAKID) for O(1) lookup
    sim_index = {}
    if sim_agg is not None and len(sim_agg) > 0:
        for _, row in sim_agg.iterrows():
            sim_index[(int(row["EID"]), int(row["PEAKID"]))] = row

    rows = []
    for _, cand in candidates.iterrows():
        eid    = int(cand["EID"])
        peakid = int(cand["PEAKID"])

        g   = pr_grouped.get((eid, peakid), empty_pr)
        sim = sim_index.get((eid, peakid), None)

        feat = {"EID": eid, "PEAKID": peakid}

        # ── Historical PR features ────────────────────────────────────────────
        if len(g) > 0:
            # Raw averages
            feat["pr_avg_3m"]   = float(g["PR"].tail(3).mean())
            feat["pr_avg_6m"]   = float(g["PR"].tail(6).mean())
            feat["pr_avg_12m"]  = float(g["PR"].tail(12).mean())
            feat["pr_last"]     = float(g["PR"].iloc[-1])
            feat["pr_max_12m"]  = float(g["PR"].tail(12).max())
            feat["pr_std_12m"]  = float(g["PR"].tail(12).std()) \
                                  if len(g) >= 2 else 0.0

            # Activation rate — how often does this EID have PR > 0?
            feat["pr_active_rate_12m"] = float((g["PR"] > 0).tail(12).mean())
            feat["pr_active_rate_24m"] = float((g["PR"] > 0).tail(24).mean())
            feat["pr_ever_active"]     = int(g["PR"].sum() > 0)
            feat["pr_n_active_12m"]    = int((g["PR"] > 0).tail(12).sum())

            # Average PR only when active (not diluted by zero months)
            active = g[g["PR"] > 0]
            feat["pr_avg_when_active"] = float(active["PR"].mean()) \
                                         if len(active) > 0 else 0.0

            # Seasonal features — same month last year
            g_ly  = g[g["MONTH"] == same_month_ly]
            g_2ly = g[g["MONTH"] == same_month_2ly]
            feat["pr_same_month_ly"]  = float(g_ly["PR"].iloc[0])  \
                                        if len(g_ly)  > 0 else feat["pr_avg_12m"]
            feat["pr_same_month_2ly"] = float(g_2ly["PR"].iloc[0]) \
                                        if len(g_2ly) > 0 else feat["pr_avg_12m"]

            feat["n_months_history"]  = len(g)

        else:
            # New EID — no history at all
            for k in ["pr_avg_3m","pr_avg_6m","pr_avg_12m","pr_last",
                      "pr_max_12m","pr_std_12m","pr_active_rate_12m",
                      "pr_active_rate_24m","pr_ever_active","pr_n_active_12m",
                      "pr_avg_when_active","pr_same_month_ly",
                      "pr_same_month_2ly","n_months_history"]:
                feat[k] = 0.0

        # ── PSM features (sim scenarios for M+1) ─────────────────────────────
        if sim is not None:
            feat["psm_s1"]             = float(sim.get("psm_s1", 0))
            feat["psm_s2"]             = float(sim.get("psm_s2", 0))
            feat["psm_s3"]             = float(sim.get("psm_s3", 0))
            feat["psm_abs_mean"]       = float(sim.get("psm_abs_mean", 0))
            feat["psm_abs_min"]        = float(sim.get("psm_abs_min", 0))
            feat["psm_abs_max"]        = float(sim.get("psm_abs_max", 0))
            feat["psm_abs_std"]        = float(sim.get("psm_abs_std", 0))
            feat["psm_scenario_agree"] = int(sim.get("scenario_agree", 0))
            feat["act_mean"]           = float(sim.get("act_mean", 0))
            feat["act_max"]            = float(sim.get("act_max", 0))
            feat["act_pos_frac"]       = float(sim.get("act_pos_frac", 0))
            feat["wind_mean"]          = float(sim.get("wind_mean", 0))
            feat["solar_mean"]         = float(sim.get("solar_mean", 0))
            feat["hydro_mean"]         = float(sim.get("hydro_mean", 0))
            feat["nonrenew_mean"]      = float(sim.get("nonrenew_mean", 0))
            feat["external_mean"]      = float(sim.get("external_mean", 0))
            feat["load_mean"]          = float(sim.get("load_mean", 0))
            feat["transmission_mean"]  = float(sim.get("transmission_mean", 0))
            feat["source_impact_sum"]  = float(sim.get("source_impact_sum", 0))
            feat["sim_in_universe"]    = 1
        else:
            # Not in sim universe — no forward signal
            for k in ["psm_s1","psm_s2","psm_s3","psm_abs_mean","psm_abs_min",
                      "psm_abs_max","psm_abs_std","psm_scenario_agree",
                      "act_mean","act_max","act_pos_frac","wind_mean",
                      "solar_mean","hydro_mean","nonrenew_mean","external_mean",
                      "load_mean","transmission_mean","source_impact_sum"]:
                feat[k] = 0.0
            feat["sim_in_universe"] = 0

        # ── Combined features ─────────────────────────────────────────────────
        pr_12m = feat.get("pr_avg_12m", 0)
        psm    = feat.get("psm_abs_mean", 0)
        feat["psm_vs_pr_ratio"]     = psm / (pr_12m + 1e-9)
        feat["psm_cv"]              = (feat["psm_abs_std"] /
                                       (psm + 1e-9))
        feat["psm_times_act_frac"]  = psm * feat.get("act_pos_frac", 0)

        # Temporal
        month_num = pd.Period(target_month, freq="M").month
        feat["month_of_year"] = month_num
        feat["month_sin"]     = float(np.sin(2 * np.pi * month_num / 12))
        feat["month_cos"]     = float(np.cos(2 * np.pi * month_num / 12))
        feat["is_winter"]     = int(month_num in [11,12,1,2,3])
        feat["is_summer"]     = int(month_num in [6,7,8,9])

        rows.append(feat)

    result = pd.DataFrame(rows).fillna(0)
    result["TARGET_MONTH"] = target_month
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: TRAIN PRICE MODEL AND PREDICT
# LightGBM regressor: predicts |PR_{M+1}|
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Historical PR
    "pr_avg_3m", "pr_avg_6m", "pr_avg_12m", "pr_last", "pr_max_12m",
    "pr_std_12m", "pr_active_rate_12m", "pr_active_rate_24m",
    "pr_ever_active", "pr_n_active_12m", "pr_avg_when_active",
    "pr_same_month_ly", "pr_same_month_2ly", "n_months_history",
    # PSM scenarios
    "psm_s1", "psm_s2", "psm_s3",
    "psm_abs_mean", "psm_abs_min", "psm_abs_max", "psm_abs_std",
    "psm_scenario_agree",
    # Simulation features
    "act_mean", "act_max", "act_pos_frac",
    "wind_mean", "solar_mean", "hydro_mean",
    "nonrenew_mean", "external_mean",
    "load_mean", "transmission_mean",
    "source_impact_sum", "sim_in_universe",
    # Combined
    "psm_vs_pr_ratio", "psm_cv", "psm_times_act_frac",
    # Temporal
    "month_of_year", "month_sin", "month_cos",
    "is_winter", "is_summer",
    # Peak type
    "PEAKID",
]


def train_price_model(
    feature_matrix: pd.DataFrame,
    truth_train: pd.DataFrame,
) -> tuple:
    """
    Two-stage hurdle model for zero-inflated PR distribution.

    WHY TWO STAGES:
      96% of PR values are exactly 0 (EID did not activate)
      A single regressor predicts small positives everywhere instead of zeros
      — it cannot distinguish "will not activate" from "will activate with small PR"

    Stage 1 — Classifier: will this EID activate? (PR > 0 vs PR = 0)
      Target:  is_active = (PR > 0)
      Model:   LightGBM classifier with heavy positive weight
               (4.2% positive rate → scale_pos_weight ≈ 23)

    Stage 2 — Regressor: given activation, how large is PR?
      Target:  log1p(PR) for rows where PR > 0 ONLY
      Model:   LightGBM regressor trained only on active rows
               (no zero dilution — model learns true price distribution)

    Final prediction:
      predicted_pr = P(active) × predicted_magnitude
                   = stage1_prob × stage2_value

    This is called a hurdle model — standard approach for zero-inflated data.
    """
    if not LGBM_AVAILABLE:
        raise ImportError("pip install lightgbm")

    # Join features with labels
    labels = truth_train[["EID","MONTH","PEAKID","PR"]].rename(
        columns={"MONTH": "TARGET_MONTH"}
    )
    train = feature_matrix.merge(
        labels, on=["EID","PEAKID","TARGET_MONTH"], how="inner"
    )

    feat_cols    = [c for c in FEATURE_COLS if c in train.columns]
    X            = train[feat_cols].fillna(0)
    is_active    = (train["PR"] > 0).astype(int)
    n_active     = is_active.sum()
    n_total      = len(train)
    pos_rate     = n_active / n_total

    print(f"  Two-stage hurdle model:")
    print(f"    Total samples:  {n_total:,}")
    print(f"    Active (PR>0):  {n_active:,} ({100*pos_rate:.1f}%)")
    print(f"    Inactive:       {n_total-n_active:,} ({100*(1-pos_rate):.1f}%)")
    print(f"    Features:       {len(feat_cols)}")
    print(f"    PR stats (active only):")
    active_pr = train.loc[is_active==1, "PR"]
    print(f"      mean={active_pr.mean():.1f}  "
          f"median={active_pr.median():.1f}  "
          f"max={active_pr.max():.1f}")

    # ── Stage 1: Classifier ───────────────────────────────────────────────────
    print(f"\n  [Stage 1] Training activation classifier...")
    pos_weight = max(1.0, (1 - pos_rate) / pos_rate)
    print(f"    scale_pos_weight: {pos_weight:.1f}  "
          f"(balances {100*pos_rate:.1f}% positive rate)")

    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X, is_active)

    # Print classifier feature importance
    imp_clf = pd.DataFrame({
        "feature":    feat_cols,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)
    print(f"\n  Top 10 features (activation classifier):")
    for _, row in imp_clf.head(10).iterrows():
        bar = "█" * int(row["importance"] / imp_clf["importance"].max() * 25)
        print(f"    {row['feature']:35s} {int(row['importance']):6d}  {bar}")

    # ── Stage 2: Regressor on ACTIVE rows only ────────────────────────────────
    print(f"\n  [Stage 2] Training price regressor (active EIDs only)...")
    active_mask  = is_active == 1
    X_active     = X[active_mask]
    train_active = train.loc[active_mask, "PR"]

    # Winsorize at 99th pct — trains for general patterns, not one-off extreme events
    p99      = train_active.quantile(0.99)
    y_active = np.log1p(train_active.clip(upper=p99).fillna(0))

    print(f"    Active training samples: {len(X_active):,}")
    print(f"    PR 99th pct cap:         {p99:,.0f}")
    print(f"    PR values capped:        {(train_active > p99).sum():,} "
    f"({100*(train_active > p99).mean():.1f}%)")

    reg = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=7,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    reg.fit(X_active, y_active)

    # Print regressor feature importance
    imp_reg = pd.DataFrame({
        "feature":    feat_cols,
        "importance": reg.feature_importances_
    }).sort_values("importance", ascending=False)
    print(f"\n  Top 10 features (price regressor):")
    for _, row in imp_reg.head(10).iterrows():
        bar = "█" * int(row["importance"] / imp_reg["importance"].max() * 25)
        print(f"    {row['feature']:35s} {int(row['importance']):6d}  {bar}")

    return clf, reg, feat_cols


def predict_prices(
    clf,
    reg,
    feat_cols: list,
    feature_matrix: pd.DataFrame,
    target_months: list,
) -> pd.DataFrame:
    """
    Apply two-stage hurdle model to predict |PR| for target months.

    predicted_pr = P(active) × predicted_magnitude_if_active

    Both components are kept in the output so predict_profit.py
    can use them separately if needed.
    """
    target_feats = feature_matrix[
        feature_matrix["TARGET_MONTH"].isin(target_months)
    ].copy()

    if len(target_feats) == 0:
        print("  ERROR: no features for target months")
        return pd.DataFrame()

    X = target_feats[feat_cols].fillna(0)

    # Stage 1: P(PR > 0) for every candidate
    prob_active = clf.predict_proba(X)[:, 1]

    # Stage 2: predicted PR magnitude (log scale → original scale)
    pred_log  = reg.predict(X)
    pred_mag  = np.expm1(pred_log)
    pred_mag  = np.maximum(pred_mag, 0)

    # Combined: expected PR = P(active) × magnitude
    predicted_pr = prob_active * pred_mag

    result = target_feats[["EID","PEAKID","TARGET_MONTH"]].copy()
    result["predicted_pr"]   = predicted_pr
    result["prob_active"]    = prob_active    # P(PR > 0) — useful for ranking
    result["pred_magnitude"] = pred_mag       # conditional price if active

    # Keep useful sim columns for the decision step
    for col in ["psm_abs_mean","psm_s1","psm_s2","psm_s3",
                "psm_scenario_agree","act_mean","act_pos_frac","psm_cv"]:
        if col in target_feats.columns:
            result[col] = target_feats[col].values

    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

def validate(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    val_months: list,
):
    """
    Compare predicted |PR| with realized |PR| on validation months.
    Uses ground truth AFTER the fact — never used for training.
    """
    print(f"\n{'='*55}")
    print("  PRICE PREDICTION VALIDATION")
    print(f"  {val_months[0]} → {val_months[-1]}")
    print(f"{'='*55}\n")

    preds = predictions[predictions["TARGET_MONTH"].isin(val_months)].copy()
    truth_val = truth[truth["MONTH"].isin(val_months)].copy()
    truth_val = truth_val.rename(columns={"MONTH":"TARGET_MONTH"})

    merged = preds.merge(
        truth_val[["EID","PEAKID","TARGET_MONTH","PR","is_profitable"]],
        on=["EID","PEAKID","TARGET_MONTH"],
        how="inner"
    ).fillna(0)

    print(f"  Rows: {len(merged):,}")

    # ── Overall correlation ───────────────────────────────────────────────────
    active = merged[merged["PR"] > 0]
    if len(active) > 10:
        corr_all    = merged[["PR","predicted_pr"]].corr().iloc[0,1]
        corr_active = active[["PR","predicted_pr"]].corr().iloc[0,1]
        mae_active  = (active["PR"] - active["predicted_pr"]).abs().mean()
        print(f"  Correlation (all EIDs):     r = {corr_all:.3f}")
        print(f"  Correlation (PR > 0 only):  r = {corr_active:.3f}")
        print(f"  Mean abs error (PR > 0):    {mae_active:.2f}")

    # ── Activation accuracy ───────────────────────────────────────────────────
    merged["pred_active"]   = (merged["predicted_pr"] > 0).astype(int)
    merged["actual_active"] = (merged["PR"] > 0).astype(int)
    pred_active = merged[merged["pred_active"] == 1]
    if len(pred_active) > 0:
        prec = (pred_active["PR"] > 0).mean()
        rec  = (merged[merged["actual_active"]==1]["pred_active"]).mean()
        print(f"\n  Activation accuracy:")
        print(f"    P(PR > 0 | predicted > 0): {100*prec:.1f}%")
        print(f"    P(predicted > 0 | PR > 0): {100*rec:.1f}%")

    # ── By PEAKID ─────────────────────────────────────────────────────────────
    print(f"\n  By PEAKID (PR > 0 only):")
    for pid, pname in [(0,"OFF"),(1,"ON")]:
        sub = active[active["PEAKID"] == pid]
        if len(sub) < 5:
            continue
        c   = sub[["PR","predicted_pr"]].corr().iloc[0,1]
        mae = (sub["PR"] - sub["predicted_pr"]).abs().mean()
        print(f"    {pname}: r={c:.3f}  MAE={mae:.2f}  n={len(sub):,}")

    # ── Decision quality with actual C ───────────────────────────────────────
    if "C" in truth_val.columns or True:
        merged2 = merged.merge(
            truth_val[["EID","PEAKID","TARGET_MONTH","C"]],
            on=["EID","PEAKID","TARGET_MONTH"], how="left"
        ).fillna(0)
        merged2["pred_profitable"] = (
            merged2["predicted_pr"] > merged2["C"]
        ).astype(int)
        if merged2["pred_profitable"].sum() > 0:
            prec = merged2[merged2["pred_profitable"]==1]["is_profitable"].mean()
            rec  = merged2[merged2["is_profitable"]==1]["pred_profitable"].mean()
            f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
            print(f"\n  Decision (predicted_pr > actual C):")
            print(f"    Precision:  {100*prec:.1f}%")
            print(f"    Recall:     {100*rec:.1f}%")
            print(f"    F1:         {f1:.4f}")
            print(f"    (Upper bound — uses actual C, not predicted C)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start-month",   default="2020-02")
    p.add_argument("--end-month",     default="2024-12")
    p.add_argument("--train-end",     default="2022-12",
                   help="Last month used for training (anti-leakage)")
    p.add_argument("--rebuild-cache", action="store_true")
    args = p.parse_args()

    np.random.seed(RANDOM_SEED)

    print("\n" + "="*55)
    print("  Price Prediction — LightGBM on PSM + PR history")
    print(f"  Train:   2020-02 → {args.train_end}")
    print(f"  Predict: {args.start_month} → {args.end_month}")
    print("="*55)

    t0 = time.time()

    # ── Load base data ────────────────────────────────────────────────────────
    print("\n[1] Loading base data...")

    costs = load_costs()
    pr    = load_prices_and_aggregate()

    cand_cache = OUTPUT_DIR / "sim_candidates.parquet"
    if cand_cache.exists():
        sim_candidates = pd.read_parquet(cand_cache)
        sim_candidates = sim_candidates[
            (sim_candidates["MONTH"] >= "2020-01") &
            (sim_candidates["MONTH"] <= "2024-12")
        ]
    else:
        from data_loader import build_sim_candidate_universe
        sim_candidates = build_sim_candidate_universe(
            sim_monthly_paths=SIM_MONTHLY_PATHS,
            sim_daily_paths=SIM_DAILY_PATHS,
            pr=pr, costs=costs,
            start_month="2020-01", end_month="2024-12",
            cache_path=cand_cache,
        )

    # Ground truth — 2020-2023 only
    truth = compute_ground_truth(
        pr=pr, costs=costs, sim_candidates=sim_candidates,
        months=get_month_range("2020-01","2023-12")
    )

    # ── Build sim cache ───────────────────────────────────────────────────────
    print("\n[2] Building sim feature cache...")
    all_months = sorted(set(
        get_month_range("2020-02", args.train_end) +
        get_month_range(args.start_month, args.end_month)
    ))
    sim_cache_df = build_sim_cache(all_months, rebuild=args.rebuild_cache)

    # ── Build feature matrices ────────────────────────────────────────────────
    print(f"\n[3] Building feature matrices...")

    all_feat_months = sorted(set(
        get_month_range("2020-02", args.train_end) +
        get_month_range(args.start_month, args.end_month)
    ))

    # Group sim cache by month for fast lookup
    sim_by_month = {}
    for tm, grp in sim_cache_df.groupby("TARGET_MONTH"):
        sim_by_month[tm] = grp.drop(columns=["TARGET_MONTH"])

    all_features = []
    for i, tm in enumerate(all_feat_months):
        if i % 12 == 0:
            print(f"  Building features: {tm} "
                  f"[{i+1}/{len(all_feat_months)}]...")

        cands = (sim_candidates[sim_candidates["MONTH"] == tm]
                 [["EID","PEAKID"]].drop_duplicates())
        if len(cands) == 0:
            continue

        # ANTI-LEAKAGE: only use PR from months before target
        pr_past = pr[pr["MONTH"] < tm]

        feats = build_price_features(
            candidates=cands,
            target_month=tm,
            pr_history=pr_past,
            sim_agg=sim_by_month.get(tm, None),
        )
        all_features.append(feats)

    feature_matrix = pd.concat(all_features, ignore_index=True)
    print(f"  ✓ Feature matrix: {feature_matrix.shape}")

    # ── Train model ───────────────────────────────────────────────────────────
    print(f"\n[4] Training price model (target: |PR|)...")

    train_months = get_month_range("2020-02", args.train_end)
    truth_train  = truth[truth["MONTH"].isin(train_months)]

    train_feats  = feature_matrix[
        feature_matrix["TARGET_MONTH"].isin(train_months)
    ]

    clf, reg, feat_cols = train_price_model(train_feats, truth_train)

    # ── Generate predictions ──────────────────────────────────────────────────
    print(f"\n[5] Generating price predictions...")

    target_months = get_month_range(args.start_month, args.end_month)
    predictions   = predict_prices(clf, reg, feat_cols, feature_matrix, target_months)

    print(f"  ✓ Predictions: {len(predictions):,} rows")
    print(f"  predicted_pr stats:")
    print(f"    mean:   {predictions['predicted_pr'].mean():.2f}")
    print(f"    median: {predictions['predicted_pr'].median():.2f}")
    print(f"    > 0:    {100*(predictions['predicted_pr']>0).mean():.1f}%")
    print(f"    max:    {predictions['predicted_pr'].max():.2f}")

    # ── Validate on known months ──────────────────────────────────────────────
    truth_months = set(truth["MONTH"].unique())
    val_months   = [m for m in target_months if m in truth_months]
    if val_months:
        validate(predictions, truth, val_months)

    # ── Save ──────────────────────────────────────────────────────────────────
    predictions.to_parquet(PRICE_PRED_CACHE, index=False)

    print(f"\n{'='*55}")
    print(f"  ✓ Saved → {PRICE_PRED_CACHE}")
    print(f"  Months:   {predictions['TARGET_MONTH'].nunique()}")
    print(f"  EIDs:     {predictions['EID'].nunique():,}")
    print(f"  Runtime:  {(time.time()-t0)/60:.1f} min")
    print(f"""
  Next step — combine with cost predictions:
    predicted_pr   (this file)     →  output/predicted_prices.parquet
    predicted_c    (cost_predictor) →  output/predicted_costs.parquet

  Decision: select if predicted_pr - predicted_c > 0
  Run: python predict_profit.py to generate opportunities.csv
""")


if __name__ == "__main__":
    main()