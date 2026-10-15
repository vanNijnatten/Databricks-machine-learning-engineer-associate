"""Estimators vs transformers and training pipelines (S3.03, S3.04)."""

from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from pyspark.ml import Estimator, Model, PipelineModel, Transformer
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession

from db_ml_associate.ch09_pipelines.sklearn_pipeline import build_sklearn_churn_pipeline
from db_ml_associate.ch09_pipelines.spark_pipeline import (
    build_spark_churn_pipeline,
    fit_and_evaluate,
)
from db_ml_associate.common import CHURN_LABEL, split_frame
from db_ml_associate.datasets import load_customer_churn


@pytest.fixture(scope="module")
def churn_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_frame(load_customer_churn(), CHURN_LABEL)


@pytest.mark.spark
def test_spark_transformers_and_estimators(spark: SparkSession) -> None:
    assembler = VectorAssembler(inputCols=["a"], outputCol="features")
    indexer = StringIndexer(inputCol="c", outputCol="c_index")
    assert isinstance(assembler, Transformer) and not isinstance(assembler, Estimator)
    assert isinstance(indexer, Estimator) and not hasattr(indexer, "transform")


@pytest.mark.spark
def test_pipeline_is_an_estimator_and_its_stages_are_ordered(
    spark: SparkSession,
) -> None:
    pipeline = build_spark_churn_pipeline()
    assert isinstance(pipeline, Estimator)
    names = [type(stage).__name__ for stage in pipeline.getStages()]
    assert names == [
        "Imputer",
        "StringIndexer",
        "OneHotEncoder",
        "VectorAssembler",
        "LogisticRegression",
    ]


@pytest.mark.spark
def test_spark_pipeline_fits_to_a_pipeline_model(
    spark: SparkSession, churn_split: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, test = churn_split
    model, auc = fit_and_evaluate(spark.createDataFrame(train), spark.createDataFrame(test))
    assert isinstance(model, PipelineModel)
    assert all(isinstance(stage, Transformer) for stage in model.stages)
    assert any(isinstance(stage, Model) for stage in model.stages)  # fitted estimators became models
    assert 0.7 < auc < 1.0


def test_sklearn_pipeline_round_trips_through_mlflow(
    tmp_path: Path, churn_split: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, test = churn_split
    pipeline = build_sklearn_churn_pipeline().fit(train.drop(columns=CHURN_LABEL), train[CHURN_LABEL])
    mlflow.sklearn.save_model(pipeline, tmp_path / "model")
    loaded = mlflow.sklearn.load_model(str(tmp_path / "model"))
    features = test.drop(columns=CHURN_LABEL)
    np.testing.assert_array_equal(loaded.predict(features), pipeline.predict(features))


def test_sklearn_pipeline_handles_unseen_categories(
    churn_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = churn_split
    pipeline = build_sklearn_churn_pipeline().fit(train.drop(columns=CHURN_LABEL), train[CHURN_LABEL])
    unseen = test.drop(columns=CHURN_LABEL).head(3).assign(contract_type="lifetime")
    assert len(pipeline.predict(unseen)) == 3
