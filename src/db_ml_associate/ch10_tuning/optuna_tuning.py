"""Optuna: the tuning library Databricks recommends instead of Hyperopt (exam objectives S3.06, S3.07).

Optuna is preinstalled in Databricks Runtime 18 LTS ML (3.6.1). A study *maximises* or
*minimises* an objective; the sampler decides the search strategy (grid, random or Bayesian TPE).
``n_jobs`` runs trials in parallel threads on the driver; for multi-node parallelism use MLflow 3's
``MlflowSparkStudy`` (see :func:`distributed_optuna_study`) or Ray Tune.
"""

from collections.abc import Callable
from typing import Any

import optuna
import pandas as pd

from db_ml_associate.ch10_tuning.hyperopt_tuning import cross_validated_f1
from db_ml_associate.common import DEFAULT_SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)

GRID: dict[str, list[float | str | None]] = {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "class_weight": [None, "balanced"]}


def make_sampler(kind: str, seed: int = DEFAULT_SEED) -> optuna.samplers.BaseSampler:
    """Return a grid, random or Bayesian (TPE) sampler."""
    # --8<-- [start:samplers]
    samplers: dict[str, optuna.samplers.BaseSampler] = {
        "grid": optuna.samplers.GridSampler(GRID, seed=seed),  # every combination
        "random": optuna.samplers.RandomSampler(seed=seed),  # independent random draws
        "bayesian": optuna.samplers.TPESampler(seed=seed),  # learns from previous trials
    }
    # --8<-- [end:samplers]
    return samplers[kind]


def tune_with_optuna(
    train: pd.DataFrame,
    n_trials: int = 20,
    seed: int = DEFAULT_SEED,
    n_jobs: int = 1,
    sampler: optuna.samplers.BaseSampler | None = None,
) -> dict[str, Any]:
    """Maximise cross-validated F1 with an Optuna study and return the best parameters."""

    # --8<-- [start:optuna_study]
    def objective(trial: optuna.Trial) -> float:
        params = {
            "C": trial.suggest_float("C", 0.01, 100, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        return cross_validated_f1(train, params, seed=seed)

    study = optuna.create_study(direction="maximize", sampler=sampler or optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)  # n_jobs > 1: parallel trials (joblib threads)
    # --8<-- [end:optuna_study]
    return dict(study.best_params)


def distributed_optuna_study(
    experiment_id: str,
    objective: Callable[[optuna.Trial], float],
    study_name: str = "churn-tuning",
    n_trials: int = 32,
    n_jobs: int = 4,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run Optuna trials on Spark executors with MLflow 3's ``MlflowSparkStudy`` (Databricks workspace).

    ``MlflowStorage`` keeps the study state in the MLflow tracking server so that trials running on
    different executors share results; every trial is logged as an MLflow run.
    """
    # --8<-- [start:mlflow_spark_study]
    from mlflow.optuna.storage import MlflowStorage
    from mlflow.pyspark.optuna.study import MlflowSparkStudy

    storage = MlflowStorage(experiment_id=experiment_id)  # MLflow tracking server as Optuna storage
    study = MlflowSparkStudy(study_name, storage, sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)  # trials run on Spark executors
    # --8<-- [end:mlflow_spark_study]
    return dict(study.best_params)
