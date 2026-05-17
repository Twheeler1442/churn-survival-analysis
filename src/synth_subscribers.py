"""
synth_subscribers.py
====================

Synthetic telecom-style subscriber cohort for survival analysis.

The generator plants known hazard structure so the analysis pipeline can be
validated end-to-end without real customer data.
"""

from __future__ import annotations

from datetime import date, timedelta
import numpy as np
import pandas as pd

PLAN_TIERS = np.array(["basic", "standard", "premium"])
PAYMENT_METHODS = np.array(["auto_card", "auto_bank", "manual"])
AGE_BANDS = np.array(["18-29", "30-44", "45-59", "60+"])
REGIONS = np.array(["west", "south", "midwest", "northeast"])


def generate_subscribers(n: int = 150_000, seed: int = 42) -> pd.DataFrame:
    """Generate a right-censored subscriber cohort with planted churn effects."""
    rng = np.random.default_rng(seed)

    plan_tier = rng.choice(PLAN_TIERS, size=n, p=[0.42, 0.40, 0.18])
    payment_method = rng.choice(PAYMENT_METHODS, size=n, p=[0.52, 0.28, 0.20])
    age_band = rng.choice(AGE_BANDS, size=n, p=[0.24, 0.36, 0.26, 0.14])
    region = rng.choice(REGIONS, size=n, p=[0.30, 0.32, 0.20, 0.18])

    monthly_charges = rng.normal(74, 18, size=n).clip(25, 180)
    support_tickets_30d = rng.poisson(0.55, size=n)
    bandwidth_gb_30d = rng.gamma(2.1, 155, size=n).clip(5, 3500)
    had_billing_dispute = rng.binomial(1, 0.08, size=n)

    # Planted multiplicative hazard structure.
    log_hazard = (
        1.16 * had_billing_dispute
        + 0.53 * (payment_method == "manual")
        - 0.36 * (plan_tier == "premium")
        - 0.16 * (plan_tier == "standard")
        - 0.42 * (age_band == "60+")
        + 0.12 * (support_tickets_30d >= 2)
        + 0.006 * (monthly_charges - 74)
    )
    hazard_multiplier = np.exp(log_hazard)

    # Mixture event-time model: early-life churn cliff plus long tail.
    early = rng.binomial(1, 0.34, size=n).astype(bool)
    base_event_time = np.where(
        early,
        rng.gamma(shape=5.0, scale=12.0, size=n),
        rng.gamma(shape=5.2, scale=88.0, size=n),
    )
    event_time = base_event_time / hazard_multiplier
    event_time = np.maximum(1, np.round(event_time)).astype(int)

    # Signup dates and study-end censoring.
    study_start = date(2023, 1, 1)
    study_end = date(2024, 12, 31)
    signup_offset = rng.integers(0, 365, size=n)
    signup_dates = np.array([study_start + timedelta(days=int(x)) for x in signup_offset])
    max_followup = np.array([(study_end - d).days for d in signup_dates])

    observed = (event_time <= max_followup).astype(int)
    tenure_days = np.where(observed == 1, event_time, max_followup)
    tenure_days = np.maximum(1, tenure_days.astype(int))

    return pd.DataFrame({
        "subscriber_id": np.arange(1, n + 1),
        "signup_date": signup_dates,
        "tenure_days": tenure_days,
        "observed": observed,
        "plan_tier": plan_tier,
        "monthly_charges": monthly_charges.round(2),
        "payment_method": payment_method,
        "had_billing_dispute": had_billing_dispute,
        "support_tickets_30d": support_tickets_30d,
        "bandwidth_gb_30d": bandwidth_gb_30d.round(1),
        "age_band": age_band,
        "region": region,
    })


if __name__ == "__main__":
    import os

    os.makedirs("data", exist_ok=True)
    df = generate_subscribers()
    df.to_parquet("data/synthetic_subscribers.parquet", index=False)
    print(f"Wrote {len(df):,} rows to data/synthetic_subscribers.parquet")
