"""
kaplan_meier.py
===============

Kaplan-Meier survival curve estimation, group summaries, log-rank testing,
and empirical hazard-window detection.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test


@dataclass
class GroupSurvival:
    group: str
    n: int
    n_events: int
    median_survival_days: float | None
    survival_at_30d: float
    survival_at_90d: float
    survival_at_180d: float


def fit_overall(df: pd.DataFrame, time_col: str = "tenure_days", event_col: str = "observed") -> KaplanMeierFitter:
    """Fit one Kaplan-Meier curve to the full cohort."""
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df[time_col], event_observed=df[event_col], label="overall")
    return kmf


def fit_by_group(
    df: pd.DataFrame,
    group_col: str,
    time_col: str = "tenure_days",
    event_col: str = "observed",
) -> dict[str, KaplanMeierFitter]:
    """Fit one Kaplan-Meier curve per group level."""
    fitted = {}
    for group, sub in df.groupby(group_col):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=sub[time_col], event_observed=sub[event_col], label=str(group))
        fitted[str(group)] = kmf
    return fitted


def summary_by_group(
    df: pd.DataFrame,
    group_col: str,
    time_col: str = "tenure_days",
    event_col: str = "observed",
) -> pd.DataFrame:
    """Return n, events, median survival, and S(t) at 30/90/180 days."""
    rows = []
    for group, sub in df.groupby(group_col):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=sub[time_col], event_observed=sub[event_col])
        med = kmf.median_survival_time_
        median_days = None if pd.isna(med) or med == np.inf else float(med)

        def s_at(t: int) -> float:
            return float(kmf.survival_function_at_times([t]).iloc[0])

        rows.append(GroupSurvival(
            group=str(group),
            n=len(sub),
            n_events=int(sub[event_col].sum()),
            median_survival_days=median_days,
            survival_at_30d=s_at(30),
            survival_at_90d=s_at(90),
            survival_at_180d=s_at(180),
        ))
    return pd.DataFrame([r.__dict__ for r in rows])


def logrank(df: pd.DataFrame, group_col: str, time_col: str = "tenure_days", event_col: str = "observed"):
    """Multivariate log-rank test across all levels of group_col."""
    return multivariate_logrank_test(
        event_durations=df[time_col],
        groups=df[group_col],
        event_observed=df[event_col],
    )


def peak_hazard_window(
    df: pd.DataFrame,
    time_col: str = "tenure_days",
    event_col: str = "observed",
    bin_days: int = 15,
    max_day: int = 365,
) -> pd.DataFrame:
    """Compute empirical hazard per time bin: events_in_bin / at_risk_at_bin_start."""
    bins = np.arange(0, max_day + bin_days, bin_days)
    rows = []
    for start, end in zip(bins[:-1], bins[1:]):
        at_risk = int((df[time_col] >= start).sum())
        events = int(((df[time_col] >= start) & (df[time_col] < end) & (df[event_col] == 1)).sum())
        hazard = events / at_risk if at_risk else np.nan
        rows.append({
            "day_start": int(start),
            "day_end": int(end),
            "at_risk": at_risk,
            "events": events,
            "hazard": hazard,
        })
    return pd.DataFrame(rows)
