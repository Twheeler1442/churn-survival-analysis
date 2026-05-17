"""
run_simulated_example.py
========================

Small, direct runnable example using simulated subscriber data.

Run from the repo root:

    pip install -r requirements.txt
    python examples/run_simulated_example.py --n 25000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synth_subscribers import generate_subscribers
from kaplan_meier import fit_overall, logrank, peak_hazard_window
from cox_model import prepare_features, fit_cox, hazard_ratios, risk_score_at_horizon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25_000, help="Number of simulated subscribers")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducible simulation")
    args = parser.parse_args()

    print(f"\n=== Simulated churn survival analysis | n={args.n:,} ===\n")

    df = generate_subscribers(n=args.n, seed=args.seed)
    print("Synthetic cohort created")
    print(f"Rows:       {len(df):,}")
    print(f"Churn rate: {df['observed'].mean():.1%}")
    print(f"Median observed tenure: {df['tenure_days'].median():.0f} days\n")

    kmf = fit_overall(df)
    print("Kaplan-Meier overall survival")
    print(f"Median survival time: {kmf.median_survival_time_:.0f} days")
    print(f"S(90 days):  {float(kmf.survival_function_at_times([90]).iloc[0]):.1%}")
    print(f"S(180 days): {float(kmf.survival_function_at_times([180]).iloc[0]):.1%}\n")

    print("Log-rank test: billing dispute vs no billing dispute")
    lr = logrank(df, group_col="had_billing_dispute")
    print(f"chi2={lr.test_statistic:.2f}, p={lr.p_value:.4g}\n")

    print("Peak empirical hazard windows")
    hazard = peak_hazard_window(df, bin_days=15, max_day=365)
    print(hazard.dropna().nlargest(5, "hazard")[["day_start", "day_end", "at_risk", "events", "hazard"]].to_string(index=False))
    print()

    print("Fitting Cox proportional hazards model...")
    features = prepare_features(df)
    cph = fit_cox(features)
    hr = hazard_ratios(cph)
    print("\nTop hazard ratios")
    print(hr.head(10).to_string(index=False, formatters={
        "hazard_ratio": "{:.2f}".format,
        "ci_lower": "{:.2f}".format,
        "ci_upper": "{:.2f}".format,
        "p_value": "{:.2g}".format,
    }))

    risks = risk_score_at_horizon(cph, features, days=90)
    scored = df.copy()
    scored["churn_risk_90d"] = risks.values
    print("\nHighest-risk simulated subscribers")
    cols = ["subscriber_id", "plan_tier", "payment_method", "had_billing_dispute", "churn_risk_90d"]
    print(scored.nlargest(10, "churn_risk_90d")[cols].to_string(index=False, formatters={
        "churn_risk_90d": "{:.1%}".format,
    }))


if __name__ == "__main__":
    main()
