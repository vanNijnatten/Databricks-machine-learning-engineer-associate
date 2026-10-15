"""Cross-validation, search strategies, Hyperopt and Optuna (S3.05-S3.10)."""

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pytest_mock import MockerFixture

from db_ml_associate.ch10_tuning.cross_validation import (
    count_models,
    grid_search_cv,
    random_search_cv,
    spark_cross_validator,
    spark_train_validation_split,
)
from db_ml_associate.ch10_tuning.hyperopt_tuning import (
    tune_with_hyperopt,
    tune_with_spark_trials,
)
from db_ml_associate.ch10_tuning.optuna_tuning import (
    distributed_optuna_study,
    make_sampler,
    tune_with_optuna,
)
from db_ml_associate.common import CHURN_LABEL, split_frame
from db_ml_associate.datasets import load_customer_churn


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    train, _ = split_frame(load_customer_churn(), CHURN_LABEL)
    return train.sample(n=1500, random_state=0).reset_index(drop=True)


class TestCountingModels:
    def test_official_sample_convention(self) -> None:
        grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": [0.01, 0.1, 1]}
        assert count_models(grid, folds=5) == 90

    def test_including_the_refit(self) -> None:
        assert (
            count_models(
                {"max_depth": [3, 5, 7, 9], "n_estimators": [50, 100, 200]},
                folds=3,
                include_refit=True,
            )
            == 37
        )

    def test_train_validation_split_is_one_fold(self) -> None:
        assert count_models({"a": [1, 2], "b": [1, 2]}, folds=1) == 4


class TestScikitLearnSearch:
    def test_grid_search_trains_combinations_times_folds(self, train: pd.DataFrame) -> None:
        search = grid_search_cv(train, folds=3)
        n_candidates = len(search.cv_results_["params"])
        assert n_candidates == 3
        assert search.n_splits_ == 3
        assert "model__C" in search.best_params_

    def test_random_search_samples_n_iter_candidates(self, train: pd.DataFrame) -> None:
        search = random_search_cv(train, n_iter=4, folds=3)
        assert len(search.cv_results_["params"]) == 4


@pytest.mark.spark
class TestSparkTuning:
    def test_cross_validator_fits_grid_times_folds(self, spark: SparkSession, train: pd.DataFrame) -> None:
        model = spark_cross_validator(folds=3, collect_sub_models=True).fit(spark.createDataFrame(train))
        assert len(model.avgMetrics) == 4  # 2 x 2 grid
        assert sum(len(fold) for fold in model.subModels) == 12  # 4 combinations x 3 folds

    def test_train_validation_split_fits_each_combination_once(self, spark: SparkSession, train: pd.DataFrame) -> None:
        model = spark_train_validation_split(collect_sub_models=True).fit(spark.createDataFrame(train))
        assert len(model.validationMetrics) == 4
        assert len(model.subModels) == 4


class TestHyperopt:
    def test_fmin_returns_readable_best_params(self, train: pd.DataFrame) -> None:
        best = tune_with_hyperopt(train, max_evals=6, seed=42)
        assert set(best) == {"C", "class_weight"}
        assert best["class_weight"] in (
            None,
            "balanced",
        )  # space_eval turns choice indices into values
        assert 0.01 <= best["C"] <= 100

    def test_fmin_is_reproducible(self, train: pd.DataFrame) -> None:
        assert tune_with_hyperopt(train, max_evals=4, seed=7) == tune_with_hyperopt(train, max_evals=4, seed=7)

    @pytest.mark.spark
    def test_spark_trials_parallelises_single_node_models(self, spark: SparkSession, train: pd.DataFrame) -> None:
        best, trials = tune_with_spark_trials(train, max_evals=4, parallelism=2, seed=42, spark=spark)
        assert len(trials.trials) == 4
        assert set(best) == {"C", "class_weight"}


class TestOptuna:
    def test_tpe_study(self, train: pd.DataFrame) -> None:
        best = tune_with_optuna(train, n_trials=6, seed=42)
        assert set(best) == {"C", "class_weight"}

    @pytest.mark.parametrize("kind", ["grid", "random", "bayesian"])
    def test_sampler_kinds(self, kind: str) -> None:
        assert (
            type(make_sampler(kind)).__name__
            == {
                "grid": "GridSampler",
                "random": "RandomSampler",
                "bayesian": "TPESampler",
            }[kind]
        )

    def test_parallel_trials_with_joblib(self, train: pd.DataFrame) -> None:
        best = tune_with_optuna(train, n_trials=4, seed=1, n_jobs=2)
        assert set(best) == {"C", "class_weight"}


def test_distributed_optuna_study_uses_mlflow_storage(mocker: MockerFixture) -> None:
    storage = mocker.patch("mlflow.optuna.storage.MlflowStorage")
    study_class = mocker.patch("mlflow.pyspark.optuna.study.MlflowSparkStudy")
    study_class.return_value.best_params = {"C": 1.0}

    def objective(trial: object) -> float:
        return 0.5

    best = distributed_optuna_study("123", objective, n_trials=8, n_jobs=2)
    storage.assert_called_once_with(experiment_id="123")
    assert study_class.call_args.args[:2] == ("churn-tuning", storage.return_value)
    study_class.return_value.optimize.assert_called_once_with(objective, n_trials=8, n_jobs=2)
    assert best == {"C": 1.0}
