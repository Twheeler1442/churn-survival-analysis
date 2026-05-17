"""
demo_survival_analysis.py
=========================

Complete survival analysis workflow on synthetic subscriber data.

Usage:
    python examples/demo_survival_analysis.py --n 50000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synth_subscribers import generate_subscribers
from kaplan_meier import fit_overall, summary_by_group, logrank, peak_hazard_window
from cox_model import prepare_features, fit_cox, hazard_ratios, check_proportional_hazards, risk_score_at_horizon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    print("\n=== Customer churn survival analysis demo ===\n")
    df = generate_subscribers(args.n, seed=args.seed)
    print(f"Subscribers: {len(df):,}")
    print(f"Observed churn rate: {df['observed'].mean():.1%}")
    print(f"Median observed tenure: {df['tenure_days'].median():.0f} days\n")

    kmf = fit_overall(df)
    print("Overall Kaplan-Meier")
    print(f"Median survival: {kmf.median_survival_time_:.0f} days")
    print(f"S(30):  {float(kmf.survival_function_at_times([30]).iloc[0]):.1%}")
    print(f"S(90):  {float(kmf.survival_function_at_times([90]).iloc[0]):.1%}")
    print(f"S(180): {float(kmf.survival_function_at_times([180]).iloc[0]):.1%}\n")

    print("Survival summary by plan tier")
    print(summary_by_group(df, "plan_tier").to_string(index=False))
    print()

    print("Log-rank tests")
    for group_col in ["plan_tier", "payment_method", "had_billing_dispute", "age_band"]:
        result = logrank(df, group_col)
        print(f"{group_col:24} chi2={result.test_statistic:10.2f}  p={result.p_value:.4g}")
    print()

    print("Peak hazard windows")
    hazard = peak_hazard_window(df, bin_days=15, max_day=365)
    print(hazard.dropna().nlargest(8, "hazard").to_string(index=False))
    print()

    print("Fitting Cox proportional hazards model...")
    features = prepare_features(df)
    cph = fit_cox(features)
    print("\nHazard ratios")
    print(hazard_ratios(cph).head(15).to_string(index=False))

    print("\nProportional hazards diagnostic sample")
    sample = features.sample(min(10_000, len(features)), random_state=0)
    cph_sample = fit_cox(sample)
    print(check_proportional_hazards(cph_sample, sample).head(10).to_string(index=False))

    risks = risk_score_at_horizon(cph, features, days=90)
    scored = df.copy()
    scored["churn_risk_90d"] = risks.values
    print("\nTop 15 subscribers by predicted 90-day churn risk")
    print(scored.nlargest(15, "churn_risk_90d")[[
        "subscriber_id", "plan_tier", "payment_method", "had_billing_dispute", "support_tickets_30d", "churn_risk_90d"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
