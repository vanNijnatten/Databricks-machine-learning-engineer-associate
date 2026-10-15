# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 9 tutorial: Training pipelines
# MAGIC **Objectives**: S3.03, S3.04 · **Runtime**: 18 LTS ML · **Requires**: serverless or classic compute · **Free Edition**: full
# MAGIC **Estimated time**: 30 min · **Clean-up**: none
# MAGIC
# MAGIC Build the same churn training pipeline in Spark ML and in scikit-learn and see which stages are estimators
# MAGIC and which are transformers. Run `00_setup` first.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

from db_ml_associate.common import CHURN_LABEL, DEFAULT_SEED, split_frame

churn = spark.table(uc_name("customer_churn"))
train, test = churn.randomSplit([0.8, 0.2], seed=DEFAULT_SEED)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Spark ML pipeline (S3.03, S3.04)
# MAGIC
# MAGIC Estimators have `fit()` and return a model; transformers have only `transform()`. `Pipeline.fit()` returns a
# MAGIC `PipelineModel` whose stages are all transformers.

# COMMAND ----------

from pyspark.ml import Estimator, Transformer

from db_ml_associate.ch09_pipelines.spark_pipeline import build_spark_churn_pipeline, fit_and_evaluate

for stage in build_spark_churn_pipeline().getStages():
    kind = "estimator" if isinstance(stage, Estimator) else "transformer" if isinstance(stage, Transformer) else "?"
    print(f"{type(stage).__name__:20} {kind}")

pipeline_model, auc = fit_and_evaluate(train, test)
print("fitted stages:", [type(stage).__name__ for stage in pipeline_model.stages])
print("test ROC AUC:", round(auc, 3))

# COMMAND ----------

display(pipeline_model.transform(test).select(CHURN_LABEL, "probability", "prediction").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — scikit-learn pipeline (S3.04)
# MAGIC
# MAGIC Same idea on a pandas DataFrame: a `ColumnTransformer` imputes, scales and one-hot encodes, then the
# MAGIC classifier. The whole pipeline is fitted on training data only and logged to MLflow as one model.

# COMMAND ----------

from sklearn.metrics import roc_auc_score

from db_ml_associate.ch09_pipelines.sklearn_pipeline import build_sklearn_churn_pipeline

train_pdf, test_pdf = split_frame(churn.toPandas(), CHURN_LABEL)
sk_pipeline = build_sklearn_churn_pipeline().fit(train_pdf.drop(columns=CHURN_LABEL), train_pdf[CHURN_LABEL])
probabilities = sk_pipeline.predict_proba(test_pdf.drop(columns=CHURN_LABEL))[:, 1]
print("test ROC AUC:", round(roc_auc_score(test_pdf[CHURN_LABEL], probabilities), 3))
display(sk_pipeline)

# COMMAND ----------

# MAGIC %md
# MAGIC > **⚠️ Wrong approach:** fitting a scaler on the full dataset before splitting leaks test statistics.

# COMMAND ----------

from db_ml_associate.ch09_pipelines import pitfalls

churn_pdf = churn.toPandas()
wrong_mean, _, _ = pitfalls.wrong_fit_scaler_on_full_data(churn_pdf, "tenure_months")
right_mean, _ = pitfalls.correct_fit_inside_pipeline(churn_pdf, "tenure_months")
print("scaler mean learned from all rows:  ", round(wrong_mean, 3))
print("scaler mean learned from train rows:", round(right_mean, 3))

# COMMAND ----------

# MAGIC %md
# MAGIC **Self-check**: is `VectorAssembler` an estimator or a transformer? Is a fitted `StringIndexerModel`?
# MAGIC (Both are transformers.)
