# Customer Churn Survival Analysis

Time-to-churn modeling on a 150K-subscriber telecom cohort using Kaplan-Meier estimation, log-rank testing, and Cox proportional hazards regression. Answers the question that binary churn classifiers can't: **not just who will churn, but when.**

> **Note on data:** This is a portfolio reimplementation of a production analysis. All data is synthetically generated; no real customer records or proprietary schemas are reproduced. The synthetic cohort is engineered to mirror the documented findings: peak churn window in days 45-75, billing disputes as a ~3x churn accelerant.

## Headline findings (from synthetic cohort)

| Finding | Result |
|---------|--------|
| Peak hazard window | Days 30-75 post-signup |
| Billing dispute hazard ratio | **3.12** (95% CI 2.94-3.32, p<0.001) |
| Manual payment hazard ratio | 1.70 |
| Premium plan hazard ratio | 0.70 (vs basic baseline) |
| 60+ age band hazard ratio | 0.66 |
| Overall event rate | ~32% over 540-day window |

The Cox model recovers the planted hazard ratios from the synthetic data within statistical tolerance, validating the analysis pipeline end-to-end.

## Quickstart

```bash
git clone https://github.com/Twheeler1442/churn-survival-analysis.git
cd churn-survival-analysis
pip install -r requirements.txt
python examples/demo_survival_analysis.py --n 50000
```

Or open the notebook:

```bash
jupyter notebook notebooks/01_walkthrough.ipynb
```

## What to click / run

The main runnable simulated-data example is:

```text
examples/run_simulated_example.py
```

From GitHub, click:

**Code → Codespaces → Create codespace on main**

Then run:

```bash
pip install -r requirements.txt
python examples/run_simulated_example.py --n 25000
```

This generates a fresh synthetic subscriber cohort, fits Kaplan-Meier curves, runs log-rank tests, fits a Cox proportional hazards model, and prints the highest-risk simulated subscribers.

## Usage

```python
from synth_subscribers import generate_subscribers
from kaplan_meier import fit_overall, logrank, peak_hazard_window
from cox_model import prepare_features, fit_cox, hazard_ratios, risk_score_at_horizon

df = generate_subscribers(n=150_000, seed=42)

# Non-parametric: what does survival look like?
kmf = fit_overall(df)
print(f"Median survival: {kmf.median_survival_time_:.0f} days")

# Where's the peak risk?
haz = peak_hazard_window(df, bin_days=15)
print(haz.nlargest(3, 'hazard'))

# Group difference test
lr = logrank(df, group_col='had_billing_dispute')
print(f"Log-rank p-value: {lr.p_value:.4g}")

# Quantify effects with Cox PH
features = prepare_features(df)
cph = fit_cox(features)
print(hazard_ratios(cph))

# Score new subscribers
risks = risk_score_at_horizon(cph, features, days=90)
```

## Statistical methodology

| Method | Use |
|--------|-----|
| Kaplan-Meier estimator | Non-parametric S(t) for cohorts and stratified groups |
| Log-rank test (multivariate) | Hypothesis test for survival curve differences |
| Empirical hazard binning | Identify peak-risk time windows without parametric assumptions |
| Cox proportional hazards regression | Quantify covariate effects on hazard, with 95% confidence intervals |
| Schoenfeld residual test | Check the proportional hazards assumption |

See [`docs/methodology.md`](docs/methodology.md) for the deeper rationale, including when to stratify vs. add time-varying coefficients.

## Synthetic data design

The cohort generator (`synth_subscribers.py`) builds 150K subscribers with engineered hazard structure:

- Two-component event time distribution: an early-cliff cohort (gamma centered ~60 days) and a long-tail cohort (gamma centered ~460 days)
- Covariate-driven hazard multipliers planted with documented log-hazard coefficients
- Right-censoring applied based on signup date and study window
- Realistic event rates (~30-35%) and median tenure (~150-200 days)

This is the only data source in the repo. The Cox model's job is to recover the planted parameters; the tests assert that it does.

## Repository layout

```text
churn-survival-analysis/
├── src/
│   ├── synth_subscribers.py   Cohort generator with planted hazard structure
│   ├── kaplan_meier.py        KM curves, log-rank, peak hazard windows
│   └── cox_model.py           Cox PH fit, HR table, PH assumption check, risk scoring
├── tests/
│   └── test_survival.py       13 tests, all passing
├── notebooks/
│   └── 01_walkthrough.ipynb   Full analysis with plots
├── examples/
│   ├── demo_survival_analysis.py
│   └── run_simulated_example.py   Short runnable simulated-data example
└── docs/
    └── methodology.md         Statistical rationale and assumption diagnostics
```

## Tests

```bash
pytest tests/ -v
```

13 tests pass, covering: cohort schema, planted hazard ratio recovery, KM monotonicity, log-rank sensitivity, peak-hazard window detection, and risk-score range validity.

## Stack

Python, NumPy, pandas, **lifelines** (KaplanMeierFitter, CoxPHFitter, log-rank, Schoenfeld), matplotlib, pyarrow.

## License

MIT. See [`LICENSE`](LICENSE).
