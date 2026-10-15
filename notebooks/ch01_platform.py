# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 1 tutorial: The Databricks ML platform
# MAGIC **Objectives**: S1.01, S1.02, S1.16 · **Runtime**: 18 LTS ML · **Requires**: Unity Catalog · **Free Edition**: full
# MAGIC **Estimated time**: 20 min · **Clean-up**: none
# MAGIC
# MAGIC You will check which runtime the notebook runs on and what it preinstalls, browse the Unity Catalog
# MAGIC namespace, and see how separate catalogs model the dev/staging/prod environments of an MLOps strategy.
# MAGIC Run `00_setup` first.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Runtime information (S1.02)
# MAGIC
# MAGIC The ML runtime preinstalls the ML libraries in tested, compatible versions. `runtime_info()` compares
# MAGIC them with the versions this book targets (`runtime.toml`).

# COMMAND ----------

from db_ml_associate.ch01_platform.runtime_info import runtime_info

info = runtime_info()
print(f"Target runtime: {info.target_runtime} (Python {info.target_python})")
print(f"Running on Databricks: {info.running_on_databricks}, runtime version: {info.runtime_version}")
for name, version in sorted(info.installed.items()):
    print(f"  {name:32} installed {version or '-':12} target {info.target_libraries[name]}")
print("Mismatches:", info.mismatches() or "none")

# COMMAND ----------

# MAGIC %md
# MAGIC Libraries showing `-` are not preinstalled (Hyperopt since 17.0 ML). A mismatch means the compute uses a
# MAGIC different runtime than the book targets; the code usually still works, but check the release notes.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — The Unity Catalog namespace (S1.01)
# MAGIC
# MAGIC Every table, volume, function and registered model has a three-level name `catalog.schema.object`.
# MAGIC Access is granted on these securables and inherited downward.

# COMMAND ----------

display(spark.sql("SHOW CATALOGS"))
display(spark.sql(f"SHOW SCHEMAS IN {CATALOG}"))
display(spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}"))

# COMMAND ----------

display(spark.sql(f"DESCRIBE TABLE EXTENDED {uc_name('customer_churn')}"))

# COMMAND ----------

# MAGIC %md
# MAGIC Open **Catalog** in the sidebar and find `customer_churn`. Look at the **Overview**, **Sample data**,
# MAGIC **Lineage** and **Permissions** tabs: lineage and permissions are tracked by Unity Catalog for every
# MAGIC asset, including the models you will register in chapter 5.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Environments as catalogs (S1.01, S1.16)
# MAGIC
# MAGIC A common MLOps layout uses one catalog per environment (`dev`, `staging`, `prod`) with the same schema
# MAGIC names, so code can be promoted unchanged and only the catalog name changes. The cell below shows how a
# MAGIC job parameter selects the environment. It only prints names; it creates nothing.

# COMMAND ----------

for environment in ("dev", "staging", "prod"):
    print(f"{environment:8} table: {environment}.churn.customer_features   model: {environment}.churn.churn_model")

# COMMAND ----------

# MAGIC %md
# MAGIC **Deploy code** (the default recommendation): the training code moves from dev to staging to prod and
# MAGIC is *re-run* in each environment, so the production model is trained on production data.
# MAGIC **Deploy models**: the model artifact trained once is copied between environments (for example with
# MAGIC `MlflowClient.copy_model_version`); prefer it when training is very expensive or must happen once.
# MAGIC
# MAGIC **Self-check**: explain in two sentences why deploy-code is the default.
