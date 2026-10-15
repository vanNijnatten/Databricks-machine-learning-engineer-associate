# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 11 tutorial: Evaluating models
# MAGIC **Objectives**: S3.11–S3.14 · **Runtime**: 18 LTS ML · **Requires**: serverless or classic compute · **Free Edition**: full
# MAGIC **Estimated time**: 30 min · **Clean-up**: none
# MAGIC
# MAGIC Compute classification and regression metrics with scikit-learn and Spark evaluators, pick the metric that
# MAGIC fits a scenario, and back-transform log-scale predictions before computing metrics. Run `00_setup` first.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

from db_ml_associate.common import CHURN_LABEL, DEFAULT_SEED, split_frame

churn = spark.table(uc_name("customer_churn"))
houses = spark.table(uc_name("house_prices"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Classification metrics (S3.11)

# COMMAND ----------

from db_ml_associate.ch08_foundations.imbalance import make_churn_classifier
from db_ml_associate.ch11_evaluation.metrics import as_float_dict, classification_metrics

train, test = split_frame(churn.toPandas(), CHURN_LABEL)
model = make_churn_classifier(class_weight="balanced").fit(train.drop(columns=CHURN_LABEL), train[CHURN_LABEL])
test_features = test.drop(columns=CHURN_LABEL)
sk_metrics = as_float_dict(
    classification_metrics(test[CHURN_LABEL], model.predict(test_features), model.predict_proba(test_features)[:, 1])
)
print(sk_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC The same metrics from Spark ML evaluators on a Spark ML pipeline's predictions:

# COMMAND ----------

from db_ml_associate.ch09_pipelines.spark_pipeline import build_spark_churn_pipeline
from db_ml_associate.ch11_evaluation.metrics import spark_classification_metrics

spark_train, spark_test = churn.randomSplit([0.8, 0.2], seed=DEFAULT_SEED)
predictions = build_spark_churn_pipeline().fit(spark_train).transform(spark_test)
print(spark_classification_metrics(predictions, label=CHURN_LABEL))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Choosing a metric (S3.13)

# COMMAND ----------

from db_ml_associate.ch11_evaluation.metric_choice import Scenario, choose_metric

display(
    spark.createDataFrame(
        [(scenario.value, choose_metric(scenario)) for scenario in Scenario], "scenario string, metric string"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Regression metrics and log-transformed targets (S3.12, S3.14)
# MAGIC
# MAGIC A model trained on `log(price)` predicts log-prices. RMSE computed on that scale is in "log units" and
# MAGIC cannot be compared with an RMSE in dollars.

# COMMAND ----------

from db_ml_associate.ch11_evaluation import pitfalls

house_pdf = houses.toPandas()
print("RMSE on the log scale (not interpretable):", round(pitfalls.wrong_rmse_on_log_scale(house_pdf), 4))
evaluation = pitfalls.correct_exponentiate_before_metrics(house_pdf)
print("RMSE after exponentiating (in price units):", round(evaluation.rmse, 0))

# COMMAND ----------

from db_ml_associate.ch11_evaluation.metrics import regression_metrics

house_train, house_test = split_frame(house_pdf, "price", stratify=False)
baseline = [house_train["price"].mean()] * len(house_test)
print(as_float_dict(regression_metrics(house_test["price"], baseline)))  # predicting the mean → R² ≈ 0

# COMMAND ----------

# MAGIC %md
# MAGIC **Self-check**: churners are 10 % of customers and missing one costs far more than a retention offer.
# MAGIC Which metric do you optimise? (Recall — or F1 if false alarms matter too.)
