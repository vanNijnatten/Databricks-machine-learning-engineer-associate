# Databricks notebook source
# MAGIC %md
# MAGIC # Clean up the tutorial resources
# MAGIC
# MAGIC Deletes everything the tutorials may have created that costs money or clutters the workspace:
# MAGIC
# MAGIC | Resource | Created by |
# MAGIC |---|---|
# MAGIC | Serving endpoint `churn-endpoint` | chapter 13 |
# MAGIC | Online table and online store `ml-associate-online-store` | chapter 3 (optional step) |
# MAGIC | Lakeflow pipeline `ml-associate-churn-scoring` | chapter 12 |
# MAGIC | Registered models `churn_model`, `churn_risk_model`, `churn_feature_model` | chapters 3, 5, 13 |
# MAGIC | Experiments `/Users/<you>/ml-associate-ch*` | chapters 3, 4, 5, 10, 13 |
# MAGIC | Schema `ml_associate.tutorial` with all tables and the volume | `00_setup` and all chapters |
# MAGIC
# MAGIC Set the `confirm` widget to `yes` and run all cells. The catalog itself is kept unless `drop_catalog` is `yes`.
# MAGIC Remember to terminate classic clusters you created (chapter 2).

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

dbutils.widgets.dropdown("confirm", "no", ["no", "yes"], "Really delete?")
dbutils.widgets.dropdown("drop_catalog", "no", ["no", "yes"], "Also drop the catalog")
if dbutils.widgets.get("confirm") != "yes":
    dbutils.notebook.exit("Nothing deleted: set the 'confirm' widget to 'yes' and run again.")

# COMMAND ----------

from collections.abc import Callable

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from mlflow import MlflowClient

workspace = WorkspaceClient()
client = MlflowClient()
fe = FeatureEngineeringClient()
ONLINE_STORE = "ml-associate-online-store"


def attempt(description: str, action: Callable[[], object]) -> None:
    """Run one deletion; report instead of failing when the resource does not exist."""
    try:
        action()
        print(f"deleted  {description}")
    except NotFound:
        print(f"absent   {description}")
    except Exception as error:  # noqa: BLE001 - keep cleaning up the remaining resources
        print(f"skipped  {description}: {type(error).__name__}: {str(error)[:150]}")


# COMMAND ----------

attempt("serving endpoint churn-endpoint", lambda: workspace.serving_endpoints.delete(name="churn-endpoint"))
attempt(
    f"online table {uc_name('customer_features_online')}",
    lambda: fe.drop_online_table(
        name=uc_name("customer_features_online"), online_store=fe.get_online_store(name=ONLINE_STORE)
    ),
)
attempt(f"online store {ONLINE_STORE}", lambda: fe.delete_online_store(name=ONLINE_STORE))
for pipeline in workspace.pipelines.list_pipelines(filter="name LIKE 'ml-associate-churn-scoring'"):
    attempt(
        f"pipeline {pipeline.name}",
        lambda pipeline=pipeline: workspace.pipelines.delete(pipeline_id=pipeline.pipeline_id),
    )

# COMMAND ----------

for model in ("churn_model", "churn_risk_model", "churn_feature_model"):
    attempt(f"registered model {uc_name(model)}", lambda model=model: client.delete_registered_model(uc_name(model)))

user = spark.sql("SELECT current_user()").first()[0]
for experiment in client.search_experiments(filter_string=f"name LIKE '/Users/{user}/ml-associate-ch%'"):
    attempt(
        f"experiment {experiment.name}",
        lambda experiment=experiment: client.delete_experiment(experiment.experiment_id),
    )

# COMMAND ----------

attempt(f"schema {CATALOG}.{SCHEMA}", lambda: spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE"))
if dbutils.widgets.get("drop_catalog") == "yes":
    attempt(f"catalog {CATALOG}", lambda: spark.sql(f"DROP CATALOG IF EXISTS {CATALOG} CASCADE"))
print("clean-up finished")
