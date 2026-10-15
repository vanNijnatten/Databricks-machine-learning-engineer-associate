# Databricks notebook source
# MAGIC %md
# MAGIC # Tutorial setup
# MAGIC
# MAGIC Run this notebook once before the chapter tutorials. It
# MAGIC
# MAGIC 1. creates the catalog and schema the tutorials use (default `ml_associate.tutorial`),
# MAGIC 2. loads the two synthetic datasets from the repository's `data/` folder into Delta tables
# MAGIC    `customer_churn` and `house_prices`.
# MAGIC
# MAGIC **Compute**: Databricks Runtime 18 LTS ML (classic, Dedicated access mode) or a serverless notebook with
# MAGIC the ML environment. **Needs**: permission to create a catalog, or an existing catalog you may create a
# MAGIC schema in (change the `catalog` widget).
# MAGIC
# MAGIC The tutorial code is imported from the repository's `src/` folder, so the chapter notebooks need no
# MAGIC package installation. Two libraries are not preinstalled in Runtime 18 LTS ML and are installed by the
# MAGIC notebooks that need them: `hyperopt==0.2.7` (chapter 10) and `databricks-automl-runtime` (chapter 2).

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

# MAGIC %md
# MAGIC ## Catalog and schema

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.files")  # checkpoints and files for chapter 12
display(spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the datasets
# MAGIC
# MAGIC The Parquet files are committed to the repository and described in `data/README.md`. They are read with
# MAGIC pandas from the Git folder (workspace files) and saved as managed Delta tables. pandas marks missing numbers
# MAGIC as NaN, which Spark treats as a value, not as missing (chapter 6 pitfall), so NaN is converted to null first.

# COMMAND ----------

from db_ml_associate.ch06_exploring.pitfalls import nan_to_null
from db_ml_associate.datasets import load_customer_churn, load_house_prices

datasets = {"customer_churn": load_customer_churn(), "house_prices": load_house_prices()}
for name, frame in datasets.items():
    floats = [column for column, dtype in frame.dtypes.items() if dtype.kind == "f"]
    table = nan_to_null(spark.createDataFrame(frame), floats)
    table.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(uc_name(name))
    print(f"{uc_name(name)}: {len(frame)} rows, NaN converted to null in {floats}")

# COMMAND ----------

churn = spark.table(uc_name("customer_churn"))
assert churn.count() == 5000, "customer_churn should have 5,000 rows"
assert spark.table(uc_name("house_prices")).count() == 3000, "house_prices should have 3,000 rows"
display(churn.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Setup complete. Continue with the chapter tutorials in the order of your study plan, and run
# MAGIC `99_cleanup` when you are finished.
