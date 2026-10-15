"""Hyperopt ``fmin`` and ``SparkTrials`` (exam objectives S3.05, S3.06, S3.07).

Platform change: Hyperopt is no longer maintained and is not included in Databricks Runtime ML
after 16.4 LTS ML; install it with ``%pip install hyperopt==0.2.7`` (it imports ``pkg_resources``,
provided by the runtime's setuptools 78.1.1). Databricks recommends Optuna or Ray Tune for new
work. The exam still asks about ``fmin``.

``SparkTrials`` runs each trial as a Spark task, so it needs a ``sparkContext``: classic
(dedicated) compute, not serverless or Spark Connect.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, SparkTrials, Trials, fmin, hp, space_eval, tpe
from pyspark.sql import SparkSession
from sklearn.model_selection import StratifiedKFold, cross_val_score

from db_ml_associate.ch09_pipelines.sklearn_pipeline import build_sklearn_churn_pipeline
from db_ml_associate.common import CHURN_LABEL, DEFAULT_SEED

# --8<-- [start:search_space]
SEARCH_SPACE = {
    "C": hp.loguniform("C", np.log(0.01), np.log(100)),  # samples exp(uniform(log 0.01, log 100))
    "class_weight": hp.choice("class_weight", [None, "balanced"]),
}
# --8<-- [end:search_space]


def cross_validated_f1(
    train: pd.DataFrame,
    params: dict[str, Any],
    folds: int = 3,
    seed: int = DEFAULT_SEED,
) -> float:
    """Mean F1 of the churn pipeline with the given model parameters over stratified folds."""
    pipeline = build_sklearn_churn_pipeline().set_params(
        model__C=params["C"], model__class_weight=params["class_weight"]
    )
    scores = cross_val_score(
        pipeline,
        train.drop(columns=CHURN_LABEL),
        train[CHURN_LABEL],
        scoring="f1",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
    )
    return float(scores.mean())


def make_objective(train: pd.DataFrame, seed: int = DEFAULT_SEED) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return the objective Hyperopt minimises: the *negative* cross-validated F1."""

    # --8<-- [start:objective]
    def objective(params: dict[str, Any]) -> dict[str, Any]:
        f1 = cross_validated_f1(train, params, seed=seed)
        return {
            "loss": -f1,
            "status": STATUS_OK,
        }  # fmin minimises, so negate a score to maximise

    # --8<-- [end:objective]
    return objective


def tune_with_hyperopt(train: pd.DataFrame, max_evals: int = 20, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Bayesian (TPE) search with ``fmin``; returns the best parameter *values*."""
    # --8<-- [start:fmin]
    trials = Trials()  # runs trials one after another on the driver
    best_indices = fmin(
        fn=make_objective(train, seed),
        space=SEARCH_SPACE,
        algo=tpe.suggest,  # Bayesian; use hyperopt.rand.suggest for random search
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(seed),
        show_progressbar=False,
    )
    best = space_eval(SEARCH_SPACE, best_indices)  # hp.choice returns an index; space_eval maps it back
    # --8<-- [end:fmin]
    return dict(best)


def tune_with_spark_trials(
    train: pd.DataFrame,
    max_evals: int = 20,
    parallelism: int = 4,
    seed: int = DEFAULT_SEED,
    spark: SparkSession | None = None,
) -> tuple[dict[str, Any], SparkTrials]:
    """Parallelise single-node scikit-learn trials across the cluster with ``SparkTrials``."""
    # --8<-- [start:spark_trials]
    spark_trials = SparkTrials(parallelism=parallelism, spark_session=spark)  # trials run as Spark tasks
    best_indices = fmin(
        fn=make_objective(train, seed),
        space=SEARCH_SPACE,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=spark_trials,
        rstate=np.random.default_rng(seed),
        show_progressbar=False,
    )
    # --8<-- [end:spark_trials]
    return dict(space_eval(SEARCH_SPACE, best_indices)), spark_trials
