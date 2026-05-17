"""Tests for the synthetic churn survival analysis pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from synth_subscribers import generate_subscribers
from kaplan_meier import fit_overall, summary_by_group, logrank, peak_hazard_window
from cox_model import prepare_features, fit_cox, hazard_ratios, check_proportional_hazards, risk_score_at_horizon


@pytest.fixture(scope="module")
def cohort():
    return generate_subscribers(n=30_000, seed=2024)


def test_cohort_has_expected_columns(cohort):
    expected = {
        "subscriber_id", "tenure_days", "observed", "plan_tier", "monthly_charges",
        "payment_method", "had_billing_dispute", "support_tickets_30d",
        "bandwidth_gb_30d", "age_band", "region",
    }
    assert expected.issubset(cohort.columns)


def test_event_rate_is_realistic(cohort):
    rate = cohort["observed"].mean()
    assert 0.20 < rate < 0.50


def test_billing_dispute_subscribers_churn_more(cohort):
    dispute_rate = cohort.loc[cohort["had_billing_dispute"] == 1, "observed"].mean()
    no_dispute_rate = cohort.loc[cohort["had_billing_dispute"] == 0, "observed"].mean()
    assert dispute_rate > 2 * no_dispute_rate


def test_overall_km_is_monotonic_non_increasing(cohort):
    kmf = fit_overall(cohort)
    s = kmf.survival_function_.iloc[:, 0].values
    assert np.all(np.diff(s) <= 1e-9)


def test_km_starts_at_one(cohort):
    kmf = fit_overall(cohort)
    assert abs(kmf.survival_function_.iloc[0, 0] - 1.0) < 1e-9


def test_summary_by_group_runs(cohort):
    out = summary_by_group(cohort, "plan_tier")
    assert set(out["group"]) == {"basic", "standard", "premium"}
    assert (out["survival_at_30d"] >= out["survival_at_90d"]).all()
    assert (out["survival_at_90d"] >= out["survival_at_180d"]).all()


def test_peak_hazard_window_is_early_life(cohort):
    haz = peak_hazard_window(cohort, bin_days=15, max_day=540).iloc[1:].dropna()
    peak_bin = haz.loc[haz["hazard"].idxmax()]
    assert 30 <= peak_bin["day_start"] <= 120


def test_logrank_detects_plan_differences(cohort):
    result = logrank(cohort, "plan_tier")
    assert result.p_value < 0.001


def test_logrank_does_not_flag_random_group(cohort):
    rng = np.random.default_rng(123)
    df = cohort.copy()
    df["random_group"] = rng.integers(0, 3, size=len(df))
    result = logrank(df, "random_group")
    assert result.p_value > 0.05


def test_cox_recovers_billing_dispute_hazard_ratio(cohort):
    features = prepare_features(cohort)
    cph = fit_cox(features)
    hrs = hazard_ratios(cph)
    dispute = hrs.loc[hrs["covariate"] == "had_billing_dispute"].iloc[0]
    assert 2.0 < dispute["hazard_ratio"] < 4.5
    assert dispute["p_value"] < 0.001


def test_cox_flags_manual_payment(cohort):
    features = prepare_features(cohort)
    cph = fit_cox(features)
    hrs = hazard_ratios(cph)
    manual = hrs.loc[hrs["covariate"] == "payment_method_manual"].iloc[0]
    assert manual["hazard_ratio"] > 1.2
    assert manual["p_value"] < 0.01


def test_risk_score_in_unit_interval(cohort):
    sample = cohort.sample(1_000, random_state=0)
    features = prepare_features(sample)
    cph = fit_cox(features)
    scores = risk_score_at_horizon(cph, features, days=90)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_proportional_hazards_check_returns_table(cohort):
    sample = cohort.sample(5_000, random_state=0)
    features = prepare_features(sample)
    cph = fit_cox(features)
    check = check_proportional_hazards(cph, features)
    assert "covariate" in check.columns
    assert "p" in check.columns or "p_value" in check.columns
