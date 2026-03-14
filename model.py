"""
model.py — Model training, scoring, and opportunity selection.

Implements:
  - LightGBM binary classifier (profitable vs not)
  - LightGBM regressor (predict profit magnitude)
  - Ensemble scoring
  - Walk-forward validation
  - Opportunity selection with constraints
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("WARNING: lightgbm not installed. Run: pip install lightgbm")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

from config import (
    LGBM_PARAMS, RANDOM_SEED,
    MIN_SELECT_PER_MONTH, MAX_SELECT_PER_MONTH, DEFAULT_SELECT,
    PEAKID_TO_NAME
)
from features import get_feature_columns, get_id_columns
from data_loader import add_months


# ─────────────────────────────────────────────────────────────────────────────
# SCORING — Baseline (no ML required)
# ─────────────────────────────────────────────────────────────────────────────

def baseline_score(features: pd.DataFrame) -> pd.DataFrame:
    """
    Simple baseline: rank by estimated historical profit proxy.
    No ML, no training data needed.
    Uses est_profit_ly (same month last year) as primary signal,
    falling back to est_profit_3m.
    """
    scores = features[["EID", "PEAKID", "TARGET_MONTH"]].copy()

    if "est_profit_ly" in features.columns:
        scores["score"] = (
            0.5 * features["est_profit_ly"].fillna(0) +
            0.3 * features.get("est_profit_3m", pd.Series(0, index=features.index)).fillna(0) +
            0.2 * features.get("profit_rate_12m", pd.Series(0, index=features.index)).fillna(0)
        )
    else:
        scores["score"] = features.get(
            "profit_avg_12m",
            pd.Series(0, index=features.index)
        ).fillna(0)

    return scores


def sim_score(features: pd.DataFrame) -> pd.DataFrame:
    """
    Simulation-based score: rank by simulated profit proxy.
    Primary signal once sim_monthly data is available.
    """
    scores = features[["EID", "PEAKID", "TARGET_MONTH"]].copy()

    if "sim_profit_mean" in features.columns:
        # Core: simulated price - estimated cost
        scores["score"] = (
            0.5 * features["sim_profit_mean"].fillna(0) +
            0.2 * features["psm_abs_mean"].fillna(0) +
            0.15 * features.get("profit_rate_12m", pd.Series(0, index=features.index)).fillna(0) +
            0.1  * features.get("act_mean", pd.Series(0, index=features.index)).fillna(0) +
            0.05 * features.get("scenario_agree", pd.Series(0, index=features.index)).fillna(0)
        )
    else:
        # Fall back to historical baseline
        scores = baseline_score(features)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# LIGHTGBM MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ProfitabilityModel:
    """
    Ensemble of LightGBM classifier and regressor for profitability prediction.

    Classifier: P(profitable | features)
    Regressor:  E[profit | features]
    Final score: classifier_prob * max(0, regressor_output)
    """

    def __init__(self, params: dict = None):
        self.params      = params or LGBM_PARAMS
        self.classifier  = None
        self.regressor   = None
        self.feature_cols = None
        self.is_fitted   = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_class: pd.Series,
        y_profit: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val_class: Optional[pd.Series] = None,
    ):
        """
        Train both classifier and regressor.

        Parameters
        ----------
        X_train    : feature matrix (training)
        y_class    : binary labels (0/1) for training
        y_profit   : continuous profit values for training
        X_val      : optional validation features (for early stopping)
        y_val_class: optional validation labels
        """
        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm required. Run: pip install lightgbm")

        self.feature_cols = get_feature_columns(X_train)
        X = X_train[self.feature_cols].fillna(0)

        print(f"  Training on {len(X):,} samples | "
              f"{y_class.sum():,} positive ({100*y_class.mean():.2f}%)")
        print(f"  Features: {len(self.feature_cols)}")

        # ── Classifier ───────────────────────────────────────────────────────
        clf_params = {**self.params, "objective": "binary", "metric": "binary_logloss"}
        self.classifier = lgb.LGBMClassifier(**clf_params)

        if X_val is not None and y_val_class is not None:
            X_v = X_val[self.feature_cols].fillna(0)
            self.classifier.fit(
                X, y_class,
                eval_set=[(X_v, y_val_class)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(100)]
            )
        else:
            self.classifier.fit(X, y_class)

        # ── Regressor (predict profit magnitude) ─────────────────────────────
        reg_params = {**self.params, "objective": "regression", "metric": "rmse"}
        self.regressor = lgb.LGBMRegressor(**reg_params)

        if X_val is not None and y_val_class is not None:
            X_v = X_val[self.feature_cols].fillna(0)
            y_val_profit = y_profit.iloc[:len(y_val_class)] if len(y_val_class) else None
            self.regressor.fit(
                X, y_profit,
                eval_set=[(X_v, y_val_class)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(100)]
            )
        else:
            self.regressor.fit(X, y_profit)

        self.is_fitted = True
        print("  ✓ Model training complete")

    def score(self, X: pd.DataFrame) -> pd.Series:
        """
        Return composite score for each row.
        Higher score = more likely profitable = should be selected.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        Xf = X[self.feature_cols].fillna(0)

        prob        = self.classifier.predict_proba(Xf)[:, 1]
        pred_profit = self.regressor.predict(Xf)

        # Composite score: probability * expected profit (if positive)
        # This naturally deprioritizes both unlikely AND unprofitable picks
        composite = prob * np.maximum(pred_profit, 0)

        # Fallback: if all predicted profits are ≤ 0, just use probability
        if composite.max() == 0:
            composite = prob

        return pd.Series(composite, index=X.index, name="score")

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance from classifier."""
        if not self.is_fitted:
            return pd.DataFrame()
        imp = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance_clf": self.classifier.feature_importances_,
            "importance_reg": self.regressor.feature_importances_,
        }).sort_values("importance_clf", ascending=False)
        imp["importance_avg"] = (imp["importance_clf"] + imp["importance_reg"]) / 2
        return imp


# ─────────────────────────────────────────────────────────────────────────────
# SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_opportunities(
    scores: pd.DataFrame,
    target_month: str,
    n_select: int = DEFAULT_SELECT,
    min_score: float = 0.0,
    enforce_peak_balance: bool = False,
) -> pd.DataFrame:
    """
    Given scored candidates, select top-N opportunities.

    Parameters
    ----------
    scores         : DataFrame with EID, PEAKID, TARGET_MONTH, score
    target_month   : str
    n_select       : number to select (will be clipped to [10, 100])
    min_score      : minimum score threshold (0 = no filtering)
    enforce_peak_balance : if True, ensure at least n_select//4 from each PEAKID

    Returns
    -------
    DataFrame: TARGET_MONTH, PEAK_TYPE, EID  (ready for opportunities.csv)
    """
    n_select = int(np.clip(n_select, MIN_SELECT_PER_MONTH, MAX_SELECT_PER_MONTH))

    month_scores = scores[scores["TARGET_MONTH"] == target_month].copy()

    if len(month_scores) == 0:
        print(f"  WARNING: no candidates for {target_month}")
        return pd.DataFrame(columns=["TARGET_MONTH", "PEAK_TYPE", "EID"])

    # Apply minimum score filter
    if min_score > 0:
        month_scores = month_scores[month_scores["score"] >= min_score]

    # Ensure enough candidates
    if len(month_scores) < MIN_SELECT_PER_MONTH:
        # Relax filter
        month_scores = scores[scores["TARGET_MONTH"] == target_month].copy()

    month_scores = month_scores.sort_values("score", ascending=False)

    if enforce_peak_balance:
        n_each = max(n_select // 4, MIN_SELECT_PER_MONTH // 2)
        on_top  = month_scores[month_scores["PEAKID"] == 1].head(n_each)
        off_top = month_scores[month_scores["PEAKID"] == 0].head(n_each)
        # Fill remaining slots with best overall
        selected_ids = set(
            on_top["EID"].tolist() + off_top["EID"].tolist()
        )
        remaining = (month_scores[~month_scores["EID"].isin(selected_ids)]
                     .head(n_select - len(on_top) - len(off_top)))
        selected = pd.concat([on_top, off_top, remaining])
    else:
        selected = month_scores.head(n_select)

    # Format output
    output = selected[["EID", "PEAKID"]].copy()
    output["TARGET_MONTH"] = target_month
    output["PEAK_TYPE"]    = output["PEAKID"].map(PEAKID_TO_NAME)
    output = output[["TARGET_MONTH", "PEAK_TYPE", "EID"]].drop_duplicates()

    # Enforce min/max
    if len(output) < MIN_SELECT_PER_MONTH:
        print(f"  WARNING: only {len(output)} selections for {target_month} "
              f"(minimum is {MIN_SELECT_PER_MONTH})")
    if len(output) > MAX_SELECT_PER_MONTH:
        output = output.head(MAX_SELECT_PER_MONTH)

    return output.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_selections(
    selections: pd.DataFrame,
    truth: pd.DataFrame,
    months: list,
) -> dict:
    """
    Compute F1-score and total net profit for a set of selections.
    Mirrors the official evaluate.py logic.

    Parameters
    ----------
    selections : DataFrame with TARGET_MONTH, PEAK_TYPE, EID
    truth      : DataFrame with EID, MONTH, PEAKID, PR, C, profit, is_profitable
    months     : list of YYYY-MM months to evaluate over

    Returns
    -------
    dict with keys: f1_on, f1_off, f1_avg, total_profit, n_selected,
                    n_profitable, n_losing, monthly_results
    """
    # Normalize selections
    sel = selections.copy()
    sel["PEAKID"]  = sel["PEAK_TYPE"].map({"OFF": 0, "ON": 1})
    sel["MONTH"]   = sel["TARGET_MONTH"]
    sel = sel[sel["MONTH"].isin(months)]
    sel = sel.drop_duplicates(subset=["MONTH", "PEAKID", "EID"])
    sel["IS_SELECTED"] = True

    truth_m = truth[truth["MONTH"].isin(months)].copy()
    truth_m["IS_PROFITABLE"] = truth_m["is_profitable"].astype(bool)

    # Merge
    merged = truth_m.merge(
        sel[["EID","MONTH","PEAKID","IS_SELECTED"]],
        on=["EID","MONTH","PEAKID"],
        how="outer"
    )
    merged["IS_PROFITABLE"] = merged["IS_PROFITABLE"].fillna(False)
    merged["IS_SELECTED"]   = merged["IS_SELECTED"].fillna(False)
    merged["PR"]            = merged["PR"].fillna(0)
    merged["C"]             = merged["C"].fillna(0)
    merged["profit"]        = merged["PR"] - merged["C"]

    # ── F1 per PEAKID ─────────────────────────────────────────────────────────
    results = {}
    for peak_id, peak_name in [(0, "OFF"), (1, "ON")]:
        sub = merged[merged["PEAKID"] == peak_id]
        tp = ((sub["IS_SELECTED"]) & (sub["IS_PROFITABLE"])).sum()
        fp = ((sub["IS_SELECTED"]) & (~sub["IS_PROFITABLE"])).sum()
        fn = ((~sub["IS_SELECTED"]) & (sub["IS_PROFITABLE"])).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)

        results[peak_name] = {
            "TP": int(tp), "FP": int(fp), "FN": int(fn),
            "Precision": round(precision, 4),
            "Recall":    round(recall, 4),
            "F1":        round(f1, 4),
        }

    f1_avg = (results["OFF"]["F1"] + results["ON"]["F1"]) / 2

    # ── Total net profit ──────────────────────────────────────────────────────
    selected_rows  = merged[merged["IS_SELECTED"]]
    total_profit   = selected_rows["profit"].sum()
    n_selected     = int(merged["IS_SELECTED"].sum())
    n_profitable   = int((selected_rows["profit"] > 0).sum())
    n_losing       = int((selected_rows["profit"] <= 0).sum())

    # ── Monthly breakdown ─────────────────────────────────────────────────────
    monthly = []
    for month in sorted(months):
        sub_m = merged[merged["MONTH"] == month]
        sel_m = sub_m[sub_m["IS_SELECTED"]]
        prof_m = set(
            sub_m[sub_m["IS_PROFITABLE"]]
            .apply(lambda r: (r["EID"], r["PEAKID"]), axis=1)
        )
        n_sel = int(sub_m["IS_SELECTED"].sum())
        tp_m  = sum(1 for _, r in sel_m.iterrows()
                    if (r["EID"], r["PEAKID"]) in prof_m)
        fp_m  = n_sel - tp_m
        profit_m = sel_m["profit"].sum()
        monthly.append({
            "MONTH": month, "n_sel": n_sel, "TP": tp_m,
            "FP": fp_m, "profit": profit_m
        })

    return {
        "f1_on":       results["ON"]["F1"],
        "f1_off":      results["OFF"]["F1"],
        "f1_avg":      round(f1_avg, 4),
        "total_profit": round(total_profit, 2),
        "n_selected":  n_selected,
        "n_profitable": n_profitable,
        "n_losing":    n_losing,
        "detail":      results,
        "monthly":     pd.DataFrame(monthly),
    }


def print_evaluation(eval_result: dict, label: str = "Evaluation"):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  F1 OFF:        {eval_result['f1_off']:.4f}")
    print(f"  F1 ON:         {eval_result['f1_on']:.4f}")
    print(f"  F1 Average:    {eval_result['f1_avg']:.4f}")
    print(f"  Total Profit:  {eval_result['total_profit']:,.2f}")
    print(f"  Selections:    {eval_result['n_selected']:,}")
    print(f"  Profitable:    {eval_result['n_profitable']:,}")
    print(f"  Losing:        {eval_result['n_losing']:,}")
    if eval_result["n_selected"] > 0:
        hit_rate = eval_result["n_profitable"] / eval_result["n_selected"]
        print(f"  Hit Rate:      {100*hit_rate:.1f}%")

    if "monthly" in eval_result:
        print(f"\n  {'MONTH':>10}  {'Sel':>5}  {'TP':>5}  {'FP':>5}  {'Profit':>12}")
        for _, row in eval_result["monthly"].iterrows():
            print(f"  {row['MONTH']:>10}  {int(row['n_sel']):>5}  "
                  f"{int(row['TP']):>5}  {int(row['FP']):>5}  "
                  f"{row['profit']:>12,.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_validate(
    truth: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    val_months: list,
    train_end_month: str,
    n_select: int = DEFAULT_SELECT,
    use_ml: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Walk-forward validation: for each validation month, train on past,
    predict on that month.

    Parameters
    ----------
    truth          : full ground truth (all months)
    feature_matrix : full feature matrix (all months)
    val_months     : list of months to validate on
    train_end_month: last month of initial training data
    n_select       : selections per month
    use_ml         : if True, use LightGBM; if False, use heuristic scoring

    Returns
    -------
    (all_selections DataFrame, eval_results dict)
    """
    all_selections = []
    model = ProfitabilityModel() if use_ml else None

    for i, val_month in enumerate(val_months):
        print(f"\n── Walk-forward: predicting {val_month} "
              f"[{i+1}/{len(val_months)}] ──")

        # Training data: everything strictly before val_month
        train_months = [m for m in feature_matrix["TARGET_MONTH"].unique()
                        if m < val_month]

        if len(train_months) == 0:
            print(f"  SKIP: no training data before {val_month}")
            continue

        # Get features and labels for training
        train_feats = feature_matrix[
            feature_matrix["TARGET_MONTH"].isin(train_months)
        ].copy()

        # Labels: is the TARGET_MONTH profitable for each (EID, PEAKID)?
        train_labels = _get_labels(train_feats, truth)

        if train_labels is None or train_labels["is_profitable"].sum() < 5:
            print(f"  SKIP: insufficient positive labels in training data")
            continue

        # Get features for validation month
        val_feats = feature_matrix[
            feature_matrix["TARGET_MONTH"] == val_month
        ].copy()

        if len(val_feats) == 0:
            print(f"  SKIP: no features for {val_month}")
            continue

        # Score and select
        if use_ml and LGBM_AVAILABLE:
            try:
                feat_cols = get_feature_columns(train_feats)
                X_train = train_feats[feat_cols].fillna(0)
                y_class  = train_labels["is_profitable"]
                y_profit = train_labels["profit"]

                model.fit(X_train, y_class, y_profit)
                scores_series = model.score(val_feats)

                scores = val_feats[["EID","PEAKID","TARGET_MONTH"]].copy()
                scores["score"] = scores_series.values
            except Exception as e:
                print(f"  WARNING: ML scoring failed ({e}), falling back to heuristic")
                scores = sim_score(val_feats) if "sim_profit_mean" in val_feats.columns \
                         else baseline_score(val_feats)
        else:
            if "sim_profit_mean" in val_feats.columns:
                scores = sim_score(val_feats)
            else:
                scores = baseline_score(val_feats)

        selections = select_opportunities(scores, val_month, n_select=n_select)
        all_selections.append(selections)
        print(f"  → Selected {len(selections)} opportunities")

    if not all_selections:
        return pd.DataFrame(), {}

    all_sel = pd.concat(all_selections, ignore_index=True)

    # Evaluate
    eval_result = evaluate_selections(all_sel, truth, val_months)
    print_evaluation(eval_result, f"Walk-Forward Results ({len(val_months)} months)")

    return all_sel, eval_result


def _get_labels(
    features: pd.DataFrame,
    truth: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """
    Join features with truth to get labels for training.
    The label for a row in features is: was (EID, PEAKID) profitable
    in TARGET_MONTH?
    """
    merged = features[["EID","PEAKID","TARGET_MONTH"]].merge(
        truth[["EID","MONTH","PEAKID","profit","is_profitable"]].rename(
            columns={"MONTH": "TARGET_MONTH"}
        ),
        on=["EID","PEAKID","TARGET_MONTH"],
        how="left"
    )
    merged["is_profitable"] = merged["is_profitable"].fillna(0).astype(int)
    merged["profit"]        = merged["profit"].fillna(0)

    if merged["is_profitable"].sum() == 0:
        return None

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# TUNE K (number of selections)
# ─────────────────────────────────────────────────────────────────────────────

def tune_k(
    scores: pd.DataFrame,
    truth: pd.DataFrame,
    months: list,
    k_values: list = None,
) -> int:
    """
    Find the optimal number of selections per month by trying different K values
    and evaluating on a validation set.

    Parameters
    ----------
    scores   : DataFrame with EID, PEAKID, TARGET_MONTH, score
    truth    : ground truth
    months   : validation months
    k_values : list of K values to try

    Returns
    -------
    Best K value
    """
    if k_values is None:
        k_values = [10, 20, 30, 40, 50, 60, 75, 100]

    results = []
    print("\n── Tuning K (selections per month) ──")

    for k in k_values:
        all_sel = []
        for month in months:
            sel = select_opportunities(scores, month, n_select=k)
            all_sel.append(sel)

        if not all_sel:
            continue

        all_sel_df = pd.concat(all_sel, ignore_index=True)
        ev = evaluate_selections(all_sel_df, truth, months)

        # Combined score: we care about both F1 and profit
        # Normalize profit to rough scale
        combined = ev["f1_avg"] + ev["total_profit"] / 10000
        results.append({
            "k": k,
            "f1_avg": ev["f1_avg"],
            "profit": ev["total_profit"],
            "combined": combined
        })
        print(f"  K={k:3d}:  F1={ev['f1_avg']:.4f}  "
              f"Profit={ev['total_profit']:>10,.2f}  Combined={combined:.4f}")

    if not results:
        return DEFAULT_SELECT

    results_df = pd.DataFrame(results)
    best_k = results_df.loc[results_df["combined"].idxmax(), "k"]
    print(f"\n  ✓ Best K = {best_k}")
    return int(best_k)