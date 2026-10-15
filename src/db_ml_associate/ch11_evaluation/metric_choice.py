"""Choosing the metric that matches a scenario's objective (exam objective S3.13)."""

from enum import StrEnum


class Scenario(StrEnum):
    """Business objectives that appear in exam scenarios."""

    MISSED_POSITIVES_COSTLY = "missing a positive case is very costly (e.g. fraud, disease)"
    FALSE_ALARMS_COSTLY = "acting on a false positive is very costly (e.g. blocking a good customer)"
    IMBALANCED_BOTH_ERRORS_MATTER = "positives are rare and both error types matter"
    WELL_CALIBRATED_PROBABILITIES = "the predicted probabilities themselves are used (e.g. expected cost)"
    RANKING_ACROSS_THRESHOLDS = "compare classifiers independently of the decision threshold"
    BALANCED_CLASSES_EQUAL_COSTS = "classes are balanced and all errors cost the same"
    LARGE_ERRORS_ESPECIALLY_BAD = "regression where a few large errors are much worse than many small ones"
    ROBUST_TO_OUTLIERS = "regression with outliers that should not dominate the score"
    SHARE_OF_VARIANCE_EXPLAINED = "regression where you report how much variation the model explains"


# --8<-- [start:metric_choice]
METRIC_FOR_SCENARIO = {
    Scenario.MISSED_POSITIVES_COSTLY: "recall",
    Scenario.FALSE_ALARMS_COSTLY: "precision",
    Scenario.IMBALANCED_BOTH_ERRORS_MATTER: "F1",
    Scenario.WELL_CALIBRATED_PROBABILITIES: "log loss",
    Scenario.RANKING_ACROSS_THRESHOLDS: "ROC AUC",
    Scenario.BALANCED_CLASSES_EQUAL_COSTS: "accuracy",
    Scenario.LARGE_ERRORS_ESPECIALLY_BAD: "RMSE",
    Scenario.ROBUST_TO_OUTLIERS: "MAE",
    Scenario.SHARE_OF_VARIANCE_EXPLAINED: "R-squared",
}
# --8<-- [end:metric_choice]


def choose_metric(scenario: Scenario) -> str:
    """Return the most appropriate metric for a scenario objective."""
    return METRIC_FOR_SCENARIO[scenario]
