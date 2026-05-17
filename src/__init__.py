"""Customer churn survival analysis — Kaplan-Meier + Cox PH."""

from synth_subscribers import generate_subscribers
from kaplan_meier import fit_overall, fit_by_group, summary_by_group, logrank, peak_hazard_window
from cox_model import prepare_features, fit_cox, hazard_ratios, check_proportional_hazards, risk_score_at_horizon

__version__ = "1.0.0"
