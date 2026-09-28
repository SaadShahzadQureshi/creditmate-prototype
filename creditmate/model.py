"""Build features, train the farm-data model, and compare it with a land-title rule."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Land size and land titles are left out on purpose: the point is to score
# farmers on what the farm produces, not on what they own. Crop type and
# irrigation are left out too: they matter through yields and their stability,
# which the model already sees, and adding them separately did not improve it.
FEATURES = [
    "yield_index",
    "yield_volatility",
    "has_livestock",
    "herd_size",
    "herd_condition",
    "wallet_months",
    "log_mobile_inflow",
    "sells_through_arthi",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw farm records into the numeric table the model learns from."""
    X = pd.DataFrame(index=df.index)
    X["yield_index"] = df["yield_index"]
    X["yield_volatility"] = df["yield_volatility"]
    X["has_livestock"] = (df["herd_size"] > 0).astype(int)
    X["herd_size"] = df["herd_size"]
    # A farmer with no herd has no condition score; use the neutral midpoint.
    X["herd_condition"] = df["herd_condition"].fillna(3.0)
    X["wallet_months"] = df["wallet_months"]
    # Inflows are skewed, so the model sees them on a log scale.
    X["log_mobile_inflow"] = np.log1p(df["mobile_inflow_pkr"])
    X["sells_through_arthi"] = df["sells_through_arthi"].astype(int)
    return X[FEATURES]


def collateral_rule(df: pd.DataFrame, min_acres: float = 5.0) -> np.ndarray:
    """Approve only farmers with a land title and at least min_acres: a typical bank rule."""
    return (df["has_land_title"] & (df["land_acres"] >= min_acres)).to_numpy()


def train_model(X: pd.DataFrame, y: pd.Series):
    """Logistic regression: simple, explainable, and easy to turn into a scorecard."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X, y)
    return model


def risk_drivers(model, feature_names) -> pd.Series:
    """Standardized coefficients: positive raises default risk, negative lowers it."""
    coefs = model.named_steps["logisticregression"].coef_[0]
    return pd.Series(coefs, index=list(feature_names)).sort_values()


def cutoff_for_default_rate(pd_scores: np.ndarray, y: np.ndarray, target: float) -> float:
    """Highest probability-of-default cutoff that keeps approved defaults at or below target.

    Farmers are approved from lowest risk upward until the default rate among
    those approved would exceed the target.
    """
    order = np.argsort(pd_scores)
    running_rate = np.cumsum(y[order]) / np.arange(1, len(order) + 1)
    ok = np.where(running_rate <= target)[0]
    if len(ok) == 0:
        return float(pd_scores.min()) - 1e-9
    return float(pd_scores[order][ok.max()])


def _summary(approved: np.ndarray, y: np.ndarray, has_title: np.ndarray) -> dict:
    return {
        "approved_count": int(approved.sum()),
        "approval_rate": float(approved.mean()),
        "default_rate_approved": float(y[approved].mean()) if approved.any() else 0.0,
        "share_approved_without_title": float((~has_title[approved]).mean()) if approved.any() else 0.0,
    }


def compare_with_collateral(df_test, pd_test, y_test, target_cutoff):
    """Compare the ways of deciding on the same held-out test farmers.

    - Land-title rule: approve titled farms of 5 acres or more.
    - Model at the same approval rate: approve as many farmers as the rule does,
      picking the lowest-risk ones, and compare defaults and reach.
    - Model at a target default rate: apply the cutoff chosen on the
      validation farmers (never on the test farmers) and see how many it approves.
    """
    has_title = df_test["has_land_title"].to_numpy()
    rule = collateral_rule(df_test)
    same_rate_cutoff = float(np.quantile(pd_test, rule.mean()))
    return {
        "auc_farm_data_model": float(roc_auc_score(y_test, pd_test)),
        "land_title_rule": _summary(rule, y_test, has_title),
        "model_at_same_approval_rate": _summary(pd_test <= same_rate_cutoff, y_test, has_title),
        "model_at_target_default_rate": _summary(pd_test <= target_cutoff, y_test, has_title),
        "cutoff_same_approval_rate": same_rate_cutoff,
        "cutoff_target_default_rate": float(target_cutoff),
    }
