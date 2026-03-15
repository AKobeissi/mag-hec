import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(r"C:\Users\akobe\OneDrive\Documents\mag-hec\output")

sim = pd.read_parquet(OUTPUT_DIR / "sim_monthly_cache.parquet")

top15 = (sim
    .sort_values("psm_abs_mean", ascending=False)
    .head(15)[["TARGET_MONTH","PEAKID","EID","psm_abs_mean","psm_s1","psm_s2","psm_s3","scenario_agree"]]
    .reset_index(drop=True)
)

top15.index += 1
top15["PEAKID"] = top15["PEAKID"].map({0:"OFF", 1:"ON"})
top15[["psm_abs_mean","psm_s1","psm_s2","psm_s3"]] = \
    top15[["psm_abs_mean","psm_s1","psm_s2","psm_s3"]].map("{:,.2f}".format)

print("\nTop 15 Highest Average PSM (mean across 3 scenarios)\n")
print(top15.to_string())