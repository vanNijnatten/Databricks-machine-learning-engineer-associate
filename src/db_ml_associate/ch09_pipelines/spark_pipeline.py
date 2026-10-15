"""A Spark ML training pipeline for the churn data (exam objectives S3.03, S3.04).

In Spark ML a **Transformer** turns one DataFrame into another with ``transform()``; an
**Estimator** learns from a DataFrame with ``fit()`` and returns a fitted **Model**, which is a
Transformer. A ``Pipeline`` is itself an Estimator: ``fit()`` runs the stages in order, fitting each
estimator on the output of the previous stages, and returns a ``PipelineModel`` made only of
transformers.
"""

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame

from db_ml_associate.common import CHURN_CATEGORICAL, CHURN_LABEL, CHURN_NUMERIC

IMPUTED = [f"{column}_imputed" for column in CHURN_NUMERIC]
INDEXED = [f"{column}_index" for column in CHURN_CATEGORICAL]
ENCODED = [f"{column}_vector" for column in CHURN_CATEGORICAL]


def build_spark_churn_pipeline(label: str = CHURN_LABEL) -> Pipeline:
    """Return an unfitted Spark ML pipeline: impute → index → one-hot → assemble → logistic regression."""
    # --8<-- [start:spark_pipeline]
    from pyspark.ml.feature import (
        Imputer,
        OneHotEncoder,
        StringIndexer,
        VectorAssembler,
    )

    imputer = Imputer(strategy="median", inputCols=list(CHURN_NUMERIC), outputCols=IMPUTED)  # estimator
    indexer = StringIndexer(inputCols=list(CHURN_CATEGORICAL), outputCols=INDEXED, handleInvalid="keep")  # estimator
    encoder = OneHotEncoder(inputCols=INDEXED, outputCols=ENCODED)  # estimator
    assembler = VectorAssembler(inputCols=IMPUTED + ENCODED, outputCol="features")  # transformer
    classifier = LogisticRegression(featuresCol="features", labelCol=label)  # estimator
    return Pipeline(stages=[imputer, indexer, encoder, assembler, classifier])
    # --8<-- [end:spark_pipeline]


def fit_and_evaluate(train: DataFrame, test: DataFrame, label: str = CHURN_LABEL) -> tuple[PipelineModel, float]:
    """Fit the pipeline on ``train`` and return the fitted model and its ROC AUC on ``test``."""
    # --8<-- [start:fit_and_evaluate]
    pipeline_model = build_spark_churn_pipeline(label).fit(train)  # Pipeline (estimator) → PipelineModel
    predictions = pipeline_model.transform(test)  # PipelineModel is a transformer
    evaluator = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    # --8<-- [end:fit_and_evaluate]
    return pipeline_model, auc
