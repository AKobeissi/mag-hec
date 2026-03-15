import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import load_costs, load_prices_and_aggregate, compute_ground_truth, get_month_range

OUTPUT_DIR = Path(r"C:\Users\akobe\OneDrive\Documents\mag-hec\output")

price_preds = pd.read_parquet(OUTPUT_DIR / "predicted_prices.parquet")
cost_preds  = pd.read_parquet(OUTPUT_DIR / "predicted_costs.parquet")
costs       = load_costs()
pr          = load_prices_and_aggregate()
sim_candidates = pd.read_parquet(OUTPUT_DIR / "sim_candidates.parquet")
truth_df    = compute_ground_truth(pr, costs, sim_candidates,
                                    months=get_month_range("2023-01","2023-12"))

combined = price_preds.merge(
    cost_preds[["EID","PEAKID","TARGET_MONTH","predicted_c"]],
    on=["EID","PEAKID","TARGET_MONTH"], how="left"
).fillna(0)

combined = combined.merge(
    truth_df.rename(columns={"MONTH":"TARGET_MONTH"})[
        ["EID","PEAKID","TARGET_MONTH","PR","C","profit","is_profitable"]
    ],
    on=["EID","PEAKID","TARGET_MONTH"], how="inner"
).fillna(0)

val_months = get_month_range("2023-01","2023-12")

# Test multiple scoring strategies
strategies = {
    "A: price only":       combined["predicted_pr"],
    "B: price - cost":     combined["predicted_pr"] - combined["predicted_c"],
    "C: soft α=0.3":       combined["predicted_pr"] * np.exp(
                               -0.3 * combined["predicted_c"] /
                               (combined["predicted_pr"] + 1e-9)),
    "D: soft α=0.5":       combined["predicted_pr"] * np.exp(
                               -0.5 * combined["predicted_c"] /
                               (combined["predicted_pr"] + 1e-9)),
    "E: soft α=1.0":       combined["predicted_pr"] * np.exp(
                               -1.0 * combined["predicted_c"] /
                               (combined["predicted_pr"] + 1e-9)),
}

print(f"\n{'Strategy':<20} {'Precision':>10} {'Net Profit':>14} "
      f"{'Std':>12} {'Worst Month':>14}")
print("-" * 75)

for name, score_col in strategies.items():
    combined["score"] = score_col
    results = []
    for month in val_months:
        grp    = combined[combined["TARGET_MONTH"] == month]
        top100 = grp.nlargest(100, "score")
        results.append({
            "month":  month,
            "n_prof": top100["is_profitable"].sum(),
            "profit": top100["profit"].sum()
        })
    df_r = pd.DataFrame(results)
    prec = df_r["n_prof"].sum() / 1200
    print(f"{name:<20} {100*prec:>9.1f}%  "
          f"{df_r['profit'].sum():>14,.0f}  "
          f"{df_r['profit'].std():>12,.0f}  "
          f"{df_r['profit'].min():>14,.0f}")