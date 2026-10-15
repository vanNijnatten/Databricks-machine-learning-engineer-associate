# Databricks notebook source
# MAGIC %md
# MAGIC # Shared tutorial settings (included by the other notebooks)
# MAGIC
# MAGIC Every tutorial notebook runs this file with `%run ./00_common`. You do not need to run it yourself.
# MAGIC It makes the repository's `src/` folder importable, defines the `catalog` and `schema` widgets and
# MAGIC sets the MLflow model registry to Unity Catalog.

# COMMAND ----------

import os
import sys

# In a Git folder the notebook's working directory is the folder that contains it (notebooks/),
# so the tested tutorial code lives one level up in src/.
REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

dbutils.widgets.text("catalog", "ml_associate", "Catalog")
dbutils.widgets.text("schema", "tutorial", "Schema")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")


def uc_name(name: str) -> str:
    """Three-level Unity Catalog name of a tutorial table or model, e.g. ml_associate.tutorial.churn_model."""
    return f"{CATALOG}.{SCHEMA}.{name}"


def has_spark_context() -> bool:
    """True on classic compute; False on serverless and other Spark Connect sessions."""
    try:
        return spark.sparkContext is not None
    except Exception:  # noqa: BLE001 - Spark Connect raises different errors per version
        return False


import mlflow

mlflow.set_registry_uri("databricks-uc")  # the default with MLflow 3; explicit for older setups

print(f"Repository: {REPO_ROOT}")
print(f"Tutorial schema: {CATALOG}.{SCHEMA}")
print(f"Databricks Runtime: {os.environ.get('DATABRICKS_RUNTIME_VERSION', 'not on Databricks')}")
