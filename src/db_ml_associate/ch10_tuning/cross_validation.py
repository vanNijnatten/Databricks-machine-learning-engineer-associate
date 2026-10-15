"""Cross-validation and grid/random search (exam objectives S3.06, S3.08, S3.09, S3.10)."""

import math
from collections.abc import Mapping, Sequence
from typing import cast

import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from db_ml_associate.ch09_pipelines.sklearn_pipeline import build_sklearn_churn_pipeline
from db_ml_associate.ch09_pipelines.spark_pipeline import build_spark_churn_pipeline
from db_ml_associate.common import CHURN_LABEL, DEFAULT_SEED


def count_models(grid: Mapping[str, Sequence[object]], folds: int, include_refit: bool = False) -> int:
    """Number of models trained by a grid search with k-fold cross-validation.

    Exam convention: combinations × folds. ``include_refit`` adds the final model that
    scikit-learn's ``refit=True`` (and Spark's ``CrossValidator``) trains on all training data.
    """
    # --8<-- [start:count_models]
    combinations = math.prod(len(values) for values in grid.values())
    return combinations * folds + (1 if include_refit else 0)
    # --8<-- [end:count_models]


def grid_search_cv(train: pd.DataFrame, folds: int = 5, seed: int = DEFAULT_SEED) -> GridSearchCV:
    """Grid search over the regularisation strength with stratified k-fold cross-validation."""
    features, labels = train.drop(columns=CHURN_LABEL), train[CHURN_LABEL]
    # --8<-- [start:grid_search_cv]
    search = GridSearchCV(
        estimator=build_sklearn_churn_pipeline(),  # preprocessing is refitted inside every fold
        param_grid={"model__C": [0.1, 1.0, 10.0]},  # step name __ parameter name
        scoring="f1",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        refit=True,  # retrain the best combination on all training data
    )
    search.fit(features, labels)  # 3 combinations × 5 folds = 15 fits (+ 1 refit)
    # --8<-- [end:grid_search_cv]
    return search


def random_search_cv(
    train: pd.DataFrame, n_iter: int = 10, folds: int = 5, seed: int = DEFAULT_SEED
) -> RandomizedSearchCV:
    """Random search: sample ``n_iter`` candidates from distributions instead of trying a full grid."""
    features, labels = train.drop(columns=CHURN_LABEL), train[CHURN_LABEL]
    # --8<-- [start:random_search_cv]
    search = RandomizedSearchCV(
        estimator=build_sklearn_churn_pipeline(),
        param_distributions={"model__C": loguniform(0.01, 100)},  # continuous, log-scaled
        n_iter=n_iter,  # number of sampled candidates, independent of the space size
        scoring="f1",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        random_state=seed,
    )
    search.fit(features, labels)
    # --8<-- [end:random_search_cv]
    return search


def spark_cross_validator(folds: int = 3, collect_sub_models: bool = False, seed: int = DEFAULT_SEED) -> CrossValidator:
    """A Spark ML ``CrossValidator`` over a 2 × 2 grid for the churn pipeline."""
    pipeline = build_spark_churn_pipeline()  # chapter 9: impute → index → encode → assemble → classifier
    classifier = cast(LogisticRegression, pipeline.getStages()[-1])
    # --8<-- [start:spark_cross_validator]
    grid = (
        ParamGridBuilder()
        .addGrid(classifier.regParam, [0.0, 0.1])
        .addGrid(classifier.elasticNetParam, [0.0, 1.0])  # 0 = L2 (ridge), 1 = L1 (lasso)
        .build()
    )  # 4 parameter maps
    cross_validator = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=BinaryClassificationEvaluator(labelCol=CHURN_LABEL),  # areaUnderROC
        numFolds=folds,  # 4 maps × 3 folds = 12 fits, then 1 refit of the best map on all data
        parallelism=2,  # fit up to 2 models at the same time
        seed=seed,
        collectSubModels=collect_sub_models,
    )
    # --8<-- [end:spark_cross_validator]
    return cross_validator


def spark_train_validation_split(
    train_ratio: float = 0.8, collect_sub_models: bool = False, seed: int = DEFAULT_SEED
) -> TrainValidationSplit:
    """Spark ML ``TrainValidationSplit``: one split instead of k folds, so k times fewer fits."""
    cross_validator = spark_cross_validator(collect_sub_models=collect_sub_models, seed=seed)
    # --8<-- [start:spark_train_validation_split]
    return TrainValidationSplit(
        estimator=cross_validator.getEstimator(),
        estimatorParamMaps=cross_validator.getEstimatorParamMaps(),
        evaluator=cross_validator.getEvaluator(),
        trainRatio=train_ratio,  # 80 % train, 20 % validation: 4 maps × 1 split = 4 fits
        seed=seed,
        collectSubModels=collect_sub_models,
    )
    # --8<-- [end:spark_train_validation_split]
