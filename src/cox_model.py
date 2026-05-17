"""
cox_model.py
============

Cox proportional hazards regression for time-to-churn modeling.
"""

from __future__ import annotations

import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

CATEGORICAL_COLS = ("plan_tier", "payment_method", "age_band", "region")
NUMERIC_COLS = ("monthly_charges", "support_tickets_30d", "bandwidth_gb_30d", "had_billing_dispute")


def prepare_features(
    df: pd.DataFrame,
    categorical: tuple[str, ...] = CATEGORICAL_COLS,
    numeric: tuple[str, ...] = NUMERIC_COLS,
    time_col: str = "tenure_days",
    event_col: str = "observed",
    drop_first: bool = True,
) -> pd.DataFrame:
    """Build a Cox-ready frame with one-hot categoricals plus duration/event columns."""
    keep = list(numeric) + [time_col, event_col]
    out = df[keep + list(categorical)].copy()
    out = pd.get_dummies(out, columns=list(categorical), drop_first=drop_first, dtype=float)
    return out


def fit_cox(
    df: pd.DataFrame,
    time_col: str = "tenure_days",
    event_col: str = "observed",
    penalizer: float = 0.01,
) -> CoxPHFitter:
    """Fit a Cox proportional hazards model with light L2 regularization."""
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col=time_col, event_col=event_col)
    return cph


def hazard_ratios(model: CoxPHFitter) -> pd.DataFrame:
    """Return a tidy hazard-ratio table with 95% confidence intervals."""
    s = model.summary
    out = pd.DataFrame({
        "covariate": s.index,
        "hazard_ratio": s["exp(coef)"].values,
        "ci_lower": s["exp(coef) lower 95%"].values,
        "ci_upper": s["exp(coef) upper 95%"].values,
        "p_value": s["p"].values,
    })
    out["abs_log_hr"] = out["hazard_ratio"].apply(lambda x: abs(__import__("math").log(x)))
    return out.sort_values("abs_log_hr", ascending=False).drop(columns="abs_log_hr").reset_index(drop=True)


def check_proportional_hazards(
    model: CoxPHFitter,
    df: pd.DataFrame,
    time_col: str = "tenure_days",
    event_col: str = "observed",
) -> pd.DataFrame:
    """Test the proportional hazards assumption via Schoenfeld residuals."""
    results = proportional_hazard_test(model, df, time_transform="rank")
    return results.summary.reset_index().rename(columns={"index": "covariate"})


def risk_score_at_horizon(model: CoxPHFitter, df: pd.DataFrame, days: int = 90) -> pd.Series:
    """Return per-subscriber probability of churn by a given day."""
    surv = model.predict_survival_function(df, times=[days])
    return (1 - surv.iloc[0]).rename(f"churn_risk_{days}d")
