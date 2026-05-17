# Methodology

## Why survival analysis and not just classification?

A churn classifier answers: *will this subscriber churn in the next N days?*

Survival analysis answers a different and more useful question: *when?* It also handles censoring natively — subscribers who haven't churned yet at the end of the study window contribute information without being thrown away or mislabeled.

| Tool | What it does | When to use it |
|------|-------------|----------------|
| Kaplan-Meier | Non-parametric estimate of S(t) = P(active at time t) | Visualize cohort survival and compare groups |
| Cox PH | Semi-parametric regression estimating covariate effects on hazard | Quantify adjusted effects and hazard ratios |
| Log-rank | Tests whether survival curves differ | Validate group-level differences |
| Schoenfeld residuals | Checks proportional hazards | Diagnose whether Cox coefficients are stable over time |

## Cohort and event definition

- **Subscribers:** all signups in the study window
- **Time origin:** signup date
- **Event:** churn / disconnect
- **Censoring:** subscriber still active at study end

Right-censoring is the only censoring handled in this demo. Left-truncation and interval-censoring are common in real subscription data but are intentionally out of scope here.

## Cox PH assumption check

The proportional hazards assumption says covariate effects are constant over time. It is tested with Schoenfeld residuals.

When violated:

1. **Stratify** on the offending covariate using `strata=` in `CoxPHFitter.fit`.
2. **Use time-varying coefficients** by reshaping into long format and fitting a time-varying Cox model.
3. **Treat Cox HRs as time-averaged effects** and disclose the limitation.

In this synthetic demo, billing disputes often create an early concentrated effect, which is realistic for subscriber onboarding and billing-friction churn.

## Interpretation cheat sheet

- **HR > 1:** covariate increases hazard, meaning faster churn.
- **HR < 1:** covariate decreases hazard, meaning slower churn.
- **HR = 2:** roughly double the instantaneous churn hazard.
- **CI crosses 1:** statistical evidence is weak.

Hazard ratios are relative hazards, not probabilities. To get a probability, use `risk_score_at_horizon(model, df, days=90)`.

## Not modeled here

- **Competing risks:** churn could mean switching to a sister service instead of leaving entirely.
- **Recurrent events:** some subscribers cancel, reactivate, then cancel again.
- **Frailty / random effects:** region or sales-rep clustering could produce correlated hazards.

Those are reasonable extensions after the simpler Kaplan-Meier + Cox pipeline is understood.
