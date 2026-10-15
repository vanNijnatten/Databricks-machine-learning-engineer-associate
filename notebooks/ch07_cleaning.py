# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 7 tutorial: Cleaning and transforming data
# MAGIC **Objectives**: S2.02, S2.05–S2.09 · **Runtime**: 18 LTS ML · **Requires**: serverless or classic compute · **Free Edition**: full
# MAGIC **Estimated time**: 40 min · **Clean-up**: none
# MAGIC
# MAGIC Remove outliers by standard deviation and IQR, impute missing values with mean, median and mode, one-hot
# MAGIC encode categoricals and log-transform skewed values — each with a wrong-vs-correct pitfall. Run `00_setup` first.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

from pyspark.sql import functions as F

churn = spark.table(uc_name("customer_churn"))
houses = spark.table(uc_name("house_prices"))
churn_pdf = churn.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Removing outliers (S2.02)
# MAGIC
# MAGIC Standard deviation: keep rows with |z| ≤ 3 (assumes a roughly normal distribution).
# MAGIC IQR: keep rows within [Q1 - 1.5·IQR, Q3 + 1.5·IQR] (robust to skew and to the outliers themselves).

# COMMAND ----------

from db_ml_associate.ch07_cleaning.outliers import iqr_bounds, remove_outliers_iqr, remove_outliers_stddev

by_stddev = remove_outliers_stddev(churn, "support_calls", threshold=3.0)
by_iqr = remove_outliers_iqr(churn, "support_calls", factor=1.5)
print("rows before:", churn.count())
print("after std-dev filter:", by_stddev.count())
print("after IQR filter:", by_iqr.count(), "bounds:", iqr_bounds(churn, "support_calls"))

# COMMAND ----------

# MAGIC %md
# MAGIC Compare the two counts: on skewed or discrete data the IQR rule can remove more rows than the 3-standard-deviation rule.
# MAGIC Only remove outliers from **training** data, and only when they are errors or unrepresentative.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Imputing missing values (S2.05, S2.06)
# MAGIC
# MAGIC Inspect the distribution first: mean for symmetric data, median for skewed data or data with outliers,
# MAGIC mode for categorical (or discrete) data.

# COMMAND ----------

display(churn.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in churn.columns]))
display(churn.select("monthly_charges", "data_usage_gb").summary("mean", "50%"))

# COMMAND ----------

from db_ml_associate.ch07_cleaning.impute import add_missing_indicator, impute_categorical_mode, impute_spark

with_indicator = add_missing_indicator(churn, "data_usage_gb")
imputed = impute_spark(with_indicator, ["monthly_charges"], strategy="mean")
imputed = impute_spark(imputed, ["data_usage_gb"], strategy="median")
imputed = impute_categorical_mode(imputed, "payment_method")
display(imputed.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in imputed.columns]))

# COMMAND ----------

# MAGIC %md
# MAGIC > **⚠️ Wrong approach:** mean-imputing a skewed column. The mean is pulled up by the long tail; the median is not.

# COMMAND ----------

from db_ml_associate.ch07_cleaning import pitfalls

print("mean fill value:  ", round(pitfalls.wrong_mean_impute_skewed(churn_pdf, "data_usage_gb"), 2))
print("median fill value:", round(pitfalls.correct_median_impute(churn_pdf, "data_usage_gb"), 2))

# COMMAND ----------

# MAGIC %md
# MAGIC > **⚠️ Wrong approach:** imputing before the train/test split leaks test statistics into training.

# COMMAND ----------

print("fill value learned from all rows:      ", pitfalls.wrong_impute_before_split(churn_pdf, "monthly_charges"))
print("fill value learned from training rows: ", pitfalls.correct_impute_after_split(churn_pdf, "monthly_charges"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — One-hot encoding (S2.07, S2.08)
# MAGIC
# MAGIC `StringIndexer` maps categories to indices; `OneHotEncoder` turns each index into a sparse vector with
# MAGIC `numCategories - 1` slots (`dropLast=True`). Appropriate for linear models and low-cardinality columns.

# COMMAND ----------

from db_ml_associate.ch07_cleaning.encoding import one_hot_spark

encoded = one_hot_spark(churn, ["contract_type", "payment_method"])
display(encoded.select("contract_type", "contract_type_vector", "payment_method", "payment_method_vector").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC > **⚠️ Wrong approach:** one-hot encoding the ~800-value `city` column for a tree model explodes the
# MAGIC > feature count; index (or target-encode) it instead.

# COMMAND ----------

print("columns after one-hot:", pitfalls.wrong_one_hot_high_cardinality_for_trees(churn_pdf, "city"))
print("columns after indexing:", pitfalls.correct_index_high_cardinality(churn_pdf, "city"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Log transform (S2.09)
# MAGIC
# MAGIC Right-skewed, non-negative values (prices, usage, counts) become closer to symmetric on a log scale.

# COMMAND ----------

from db_ml_associate.ch07_cleaning.transforms import log_transform_spark

logged = log_transform_spark(houses, "price")
display(logged.select(F.skewness("price").alias("skew_price"), F.skewness("price_log").alias("skew_log_price")))

# COMMAND ----------

# MAGIC %md
# MAGIC > **⚠️ Wrong approach:** `log(0)` is minus infinity. Use `log1p` (log(1 + x)) for data that contains zeros.

# COMMAND ----------

import numpy as np

wrong = pitfalls.wrong_log_of_zero(churn_pdf, "support_calls")
right = pitfalls.correct_log1p(churn_pdf, "support_calls")
print("infinite values with log:  ", int(np.isinf(wrong).sum()))
print("infinite values with log1p:", int(np.isinf(right).sum()))

# COMMAND ----------

# MAGIC %md
# MAGIC A model trained on `log(price)` predicts log-prices: exponentiate (`expm1` after `log1p`) before computing
# MAGIC metrics in dollars — see the chapter 11 tutorial.
# MAGIC
# MAGIC **Self-check**: which imputation would you choose for `lot_size_sqm` in `house_prices`, and why?
