"""Finding the best run with the MLflow Client API (exam objective S1.11)."""

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import Run


def find_best_run(client: MlflowClient, experiment_id: str, metric: str, mode: str = "max") -> Run:
    """Return the run with the highest (``mode="max"``) or lowest (``"min"``) value of ``metric``."""
    # --8<-- [start:find_best_run]
    direction = "DESC" if mode == "max" else "ASC"  # DESC for scores (F1, AUC), ASC for losses (RMSE)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"metrics.{metric} > -1e300",  # only runs that logged the metric
        order_by=[f"metrics.{metric} {direction}"],
        max_results=1,
    )
    best_run = runs[0]  # best_run.info.run_id, best_run.data.metrics, best_run.data.params
    # --8<-- [end:find_best_run]
    return best_run


def runs_as_dataframe(experiment_name: str) -> pd.DataFrame:
    """All runs of an experiment as a pandas DataFrame (columns like ``metrics.f1``, ``params.C``)."""
    # --8<-- [start:search_runs]
    runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["start_time DESC"])
    # filter example: mlflow.search_runs(..., filter_string="metrics.test_f1 > 0.3 and params.C = '1.0'")
    # --8<-- [end:search_runs]
    return runs
