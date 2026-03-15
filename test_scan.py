import pandas as pd
from pathlib import Path
from data_loader import load_costs, load_prices_and_aggregate, compute_ground_truth, get_month_range

OUTPUT_DIR = Path(r"C:\Users\akobe\OneDrive\Documents\mag-hec\output")

# Load your actual 2023 selections
opps = pd.read_csv(r"C:\Users\akobe\OneDrive\Documents\mag-hec\opportunities.csv")

# Load ground truth
costs = load_costs()
pr    = load_prices_and_aggregate()
sim_c = pd.read_parquet(OUTPUT_DIR / "sim_candidates.parquet")
truth = compute_ground_truth(pr, costs, sim_c,
                              months=get_month_range("2023-01","2023-12"))
truth = truth.rename(columns={"MONTH":"TARGET_MONTH"})
opps["PEAKID"] = opps["PEAK_TYPE"].map({"OFF":0,"ON":1})

merged = opps.merge(
    truth[["EID","PEAKID","TARGET_MONTH","profit","is_profitable"]],
    on=["EID","PEAKID","TARGET_MONTH"], how="left"
).fillna(0)

wins  = merged[merged["is_profitable"] == 1]["profit"]
loses = merged[merged["is_profitable"] == 0]["profit"]

print("=== ACTUAL ASYMMETRIC BET IN YOUR SELECTIONS ===\n")
print(f"Total selections:    {len(merged):,}")
print(f"Wins (profitable):   {len(wins):,}  ({100*len(wins)/len(merged):.1f}%)")
print(f"Losses:              {len(loses):,}  ({100*len(loses)/len(merged):.1f}%)")
print(f"\nAvg profit per WIN:  ${wins.mean():,.2f}")
print(f"Avg loss per LOSS:   ${loses.mean():,.2f}")
print(f"Ratio:               {abs(wins.mean()/loses.mean()):.1f}:1")
print(f"\nExpected value per selection:")
print(f"  = {100*len(wins)/len(merged):.1f}% × ${wins.mean():,.0f} + "
      f"{100*len(loses)/len(merged):.1f}% × ${loses.mean():,.0f}")
ev = (len(wins)/len(merged)) * wins.mean() + (len(loses)/len(merged)) * loses.mean()
print(f"  = ${ev:,.2f}")
print(f"\nTotal profit check:  ${merged['profit'].sum():,.2f}")
print(f"(Should match evaluate.py: $662,272)")

print(f"\n=== FOR YOUR CHART ===")
print(f"Use THESE numbers — from your actual selections:")
print(f"  avg_win  = ${wins.mean():,.0f}")
print(f"  avg_loss = ${loses.mean():,.0f}")
print(f"  ratio    = {abs(wins.mean()/loses.mean()):.0f}:1")
print(f"  breakeven precision = {abs(loses.mean())/(wins.mean()+abs(loses.mean()))*100:.2f}%")