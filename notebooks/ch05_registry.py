# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 5 tutorial: Model lifecycle in Unity Catalog
# MAGIC **Objectives**: S1.14, S1.15, S1.17, S1.18 · **Runtime**: 18 LTS ML · **Requires**: Unity Catalog · **Free Edition**: full
# MAGIC **Estimated time**: 35 min · **Clean-up**: optional (`99_cleanup` deletes the registered model)
# MAGIC
# MAGIC Register two versions of the churn model in Unity Catalog, set and remove tags, and promote the challenger
# MAGIC to champion with aliases. Chapters 12 and 13 use the `churn_model@champion` this notebook creates.
# MAGIC Run `00_setup` first.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

from mlflow import MlflowClient

from db_ml_associate.ch04_tracking.logging import train_and_log_churn_model
from db_ml_associate.common import CHURN_LABEL, split_frame

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/ml-associate-ch05")
MODEL_NAME = uc_name("churn_model")  # three levels: catalog.schema.model
client = MlflowClient()
train, test = split_frame(spark.table(uc_name("customer_churn")).toPandas(), CHURN_LABEL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Register model versions (S1.14, S1.15)
# MAGIC
# MAGIC Unity Catalog requires a model signature; `train_and_log_churn_model` logs one. Each registration of the same
# MAGIC name creates a new version.

# COMMAND ----------

from db_ml_associate.ch05_registry.registry import register_model

weak = train_and_log_churn_model(train, test, c=0.01)
strong = train_and_log_churn_model(train, test, c=1.0)
first = register_model(weak.model_uri, MODEL_NAME)
second = register_model(strong.model_uri, MODEL_NAME)
print(
    f"{MODEL_NAME}: version {first.version} (F1 {weak.metrics['test_f1']:.3f}), version {second.version} (F1 {strong.metrics['test_f1']:.3f})"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Open **Catalog → ml_associate → tutorial → churn_model**: versions, their source runs, lineage to the
# MAGIC training data and permissions are governed by Unity Catalog — across every workspace attached to the metastore.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Set and remove tags (S1.17)

# COMMAND ----------

from db_ml_associate.ch05_registry.registry import remove_model_tag, remove_version_tag, set_model_tag, set_version_tag

set_model_tag(client, MODEL_NAME, "team", "retention")
set_model_tag(client, MODEL_NAME, "temporary", "yes")
set_version_tag(client, MODEL_NAME, second.version, "validation_status", "approved")
remove_model_tag(client, MODEL_NAME, "temporary")
print("model tags:", client.get_registered_model(MODEL_NAME).tags)
print("version tags:", client.get_model_version(MODEL_NAME, second.version).tags)
remove_version_tag(client, MODEL_NAME, second.version, "validation_status")
print("version tags after removal:", client.get_model_version(MODEL_NAME, second.version).tags)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Champion and challenger aliases (S1.18)

# COMMAND ----------

from db_ml_associate.ch05_registry.registry import load_by_alias, promote_challenger, set_alias

set_alias(client, MODEL_NAME, "champion", first.version)
set_alias(client, MODEL_NAME, "challenger", second.version)
promoted = promote_challenger(client, MODEL_NAME, "test_f1")
champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
print("challenger promoted:", promoted, "→ champion is version", champion.version)

# COMMAND ----------

champion_model = load_by_alias(MODEL_NAME, "champion")  # models:/<name>@champion — no version number in code
display(champion_model.predict(test.drop(columns=CHURN_LABEL).head(5)))

# COMMAND ----------

# MAGIC %md
# MAGIC > **⚠️ Wrong approach:** workspace-registry stages (`transition_model_version_stage`) do not exist for models
# MAGIC > in Unity Catalog — the call fails. Use aliases instead.

# COMMAND ----------

from db_ml_associate.ch05_registry.pitfalls import wrong_transition_stage

try:
    wrong_transition_stage(client, MODEL_NAME, champion.version)
except Exception as error:  # noqa: BLE001 - we want to show whatever the registry returns
    print("as expected, stages are not supported:", type(error).__name__, str(error)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC **Self-check**: a batch job loads `models:/ml_associate.tutorial.churn_model/1`. What changes after the next
# MAGIC promotion, and how would an alias avoid it?
