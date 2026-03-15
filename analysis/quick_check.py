# quick_data_check.py — run this to see your training data shape
import pandas as pd
from pathlib import Path
from data_loader import (
    load_costs, load_prices_and_aggregate,
    compute_ground_truth, get_month_range, add_months
)
from config import OUTPUT_DIR
from features import build_features, get_feature_cols
from data_loader import (
    load_sim_monthly_for_target,
    aggregate_sim_across_scenarios,
    SIM_MONTHLY_PATHS,
)

# Load base data
costs = load_costs()
pr    = load_prices_and_aggregate()
sim_candidates = pd.read_parquet(OUTPUT_DIR / "sim_candidates.parquet")
truth = compute_ground_truth(pr, costs, sim_candidates,
                              months=get_month_range("2020-01","2023-12"))

# Build features for ONE month as a sanity check
target = "2022-06"
truth_past = truth[truth["MONTH"] < target]
candidates = sim_candidates[sim_candidates["MONTH"] == target][["EID","PEAKID"]]

sim_raw = load_sim_monthly_for_target(target, SIM_MONTHLY_PATHS)
sim_agg = aggregate_sim_across_scenarios(sim_raw)

feats = build_features(candidates, target, truth_past, sim_agg)

# Join with labels to see what the model will learn from
labeled = feats.merge(
    truth[truth["MONTH"]==target][["EID","PEAKID","is_profitable","profit"]]
    .rename(columns={"MONTH":"TARGET_MONTH"}),
    on=["EID","PEAKID"], how="left"
).fillna(0)

print(f"\nFeature matrix shape: {feats.shape}")
print(f"Positive rate:        {labeled['is_profitable'].mean():.3f}")
print(f"\nSample features for a profitable EID:")
print(labeled[labeled["is_profitable"]==1].head(2)[get_feature_cols(feats)[:10]])
print(f"\nSample features for an unprofitable EID:")
print(labeled[labeled["is_profitable"]==0].head(2)[get_feature_cols(feats)[:10]])