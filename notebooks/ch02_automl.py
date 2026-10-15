# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 2 tutorial: AutoML
# MAGIC **Objectives**: S1.03, S1.04 · **Runtime**: 18 LTS ML · **Requires**: classic ML compute, Dedicated access mode · **Free Edition**: read_through
# MAGIC **Estimated time**: 45 min · **Clean-up**: terminate the cluster when the experiment has finished
# MAGIC
# MAGIC Run an AutoML classification experiment on the churn data and inspect what it generates: a data exploration
# MAGIC notebook, one MLflow run per trial, and an editable notebook for the best trial.
# MAGIC
# MAGIC > **Requires:** AutoML runs only on classic compute with the ML runtime in **Dedicated** (single user) access
# MAGIC > mode — not on serverless, so not on Free Edition. **Cost**: the cluster runs for the experiment's
# MAGIC > `timeout_minutes` (10 here) plus start-up; with a small single-node cluster that is well under one hour of
# MAGIC > DBUs. **Clean-up**: terminate the cluster afterwards (or set auto-termination).
# MAGIC
# MAGIC > **Platform change:** AutoML is not built into Databricks Runtime from 18.0 ML; it is installed from the
# MAGIC > `databricks-automl-runtime` package (next cell). See Appendix A.
# MAGIC
# MAGIC **Free Edition / serverless learners**: read this notebook and the chapter's screenshots instead of running it.

# COMMAND ----------

# MAGIC %pip install databricks-automl-runtime

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

if not has_spark_context():
    raise RuntimeError(
        "AutoML needs classic compute (Dedicated access mode). Read this notebook instead of running it."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Run an AutoML experiment (S1.03, S1.04)
# MAGIC
# MAGIC The API call is the same as the **Experiments → Create AutoML experiment** form in the UI: dataset, target
# MAGIC column, primary metric, time-out and columns to exclude. `customer_id` and `city` are excluded so that AutoML
# MAGIC does not treat an identifier as a feature.

# COMMAND ----------

from databricks import automl

from db_ml_associate.ch02_automl.workspace import run_automl_classification, summarize_automl

churn = spark.table(uc_name("customer_churn"))
summary = run_automl_classification(
    automl,
    churn,
    target="churned",
    timeout_minutes=10,
    primary_metric="f1",
    exclude_cols=["customer_id", "city"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Inspect the results (S1.03, S1.04)

# COMMAND ----------

result = summarize_automl(summary)
print("experiment:", result.experiment_id)
print("trials:", result.trial_count)
print("best run:", result.best_run_id, {k: round(v, 3) for k, v in result.best_metrics.items() if "f1" in k})
print("best trial notebook:", result.best_notebook_url)
print("best model URI:", result.best_model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC Now open the experiment (link in the output of Step 1) and check off:
# MAGIC
# MAGIC - [ ] The **data exploration notebook**: column types AutoML detected, missing values, distributions and
# MAGIC       correlations — this is AutoML's semantic type detection and data profiling
# MAGIC - [ ] The **runs table**: one run per trial with the algorithm (scikit-learn, XGBoost, LightGBM),
# MAGIC       hyperparameters and validation metrics — model selection made visible
# MAGIC - [ ] The **best trial notebook**: complete, editable source code (glass-box) including preprocessing such as
# MAGIC       imputation and one-hot encoding, training and SHAP feature importance (when enabled)
# MAGIC - [ ] The registered best model is **not** created automatically; register it yourself if you want to keep it
# MAGIC
# MAGIC AutoML gives you a strong baseline and reproducible code in minutes; you still review the features, check for
# MAGIC leakage and decide whether to improve the generated notebook.

# COMMAND ----------

best_model = mlflow.pyfunc.load_model(result.best_model_uri)
display(best_model.predict(churn.limit(5).toPandas().drop(columns=["churned"])))

# COMMAND ----------

# MAGIC %md
# MAGIC **Clean-up**: terminate the cluster (Compute → your cluster → Terminate). The experiment and its notebooks
# MAGIC remain available.
# MAGIC
# MAGIC **Self-check**: name two ways AutoML helps with feature or model selection, and one reason not to ship the
# MAGIC best trial unchanged.
