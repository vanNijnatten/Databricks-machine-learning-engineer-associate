# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 4 tutorial: Experiment tracking with MLflow
# MAGIC **Objectives**: S1.11–S1.13 · **Runtime**: 18 LTS ML · **Requires**: serverless or classic compute · **Free Edition**: full
# MAGIC **Estimated time**: 40 min · **Clean-up**: none (delete the experiment if you like)
# MAGIC
# MAGIC Log parameters, metrics, artifacts and models manually and with autologging, explore the MLflow UI, and find
# MAGIC the best run with the Client API. Run `00_setup` first.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

from db_ml_associate.common import CHURN_LABEL, split_frame

user = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT = f"/Users/{user}/ml-associate-ch04"
experiment = mlflow.set_experiment(
    EXPERIMENT
)  # a workspace experiment; without this, runs go to the notebook experiment
train, test = split_frame(spark.table(uc_name("customer_churn")).toPandas(), CHURN_LABEL)
print("experiment id:", experiment.experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Manual logging (S1.12)
# MAGIC
# MAGIC Each call of `train_and_log_churn_model` opens a run and logs parameters, metrics, a confusion-matrix figure,
# MAGIC a JSON artifact and the model with its signature and input example.

# COMMAND ----------

from db_ml_associate.ch04_tracking.logging import log_learning_curve, train_and_log_churn_model

logged = [train_and_log_churn_model(train, test, c=c) for c in (0.01, 0.1, 1.0, 10.0)]
for run in logged:
    print(run.run_id, {name: round(value, 3) for name, value in run.metrics.items()}, run.model_uri)

with mlflow.start_run(run_name="metric steps"):
    log_learning_curve([0.9, 0.6, 0.45, 0.4, 0.38])  # one value per step → a line chart in the UI

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Autologging and nested runs (S1.12)

# COMMAND ----------

from db_ml_associate.ch04_tracking.logging import train_with_autolog, tune_with_nested_runs

autolog_run_id = train_with_autolog(train)
parent_run_id = tune_with_nested_runs(train, test, cs=(0.05, 0.5, 5.0))
print("autolog run:", autolog_run_id, "parent run:", parent_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Tour of the MLflow UI (S1.13)
# MAGIC
# MAGIC Click **Experiments** in the sidebar and open `ml-associate-ch04`. Check off what you find:
# MAGIC
# MAGIC - [ ] The **runs table**: run names, creation time, duration, source notebook, logged parameters and metrics as columns
# MAGIC - [ ] Sort the table by `test_f1`; the **parent run** `tune C` expands into its child runs
# MAGIC - [ ] Select two or more runs → **Compare**: parallel-coordinates and scatter plots of parameters vs metrics
# MAGIC - [ ] Open a run → **Overview**: parameters, metrics, tags, source notebook (and Git commit for a Git folder)
# MAGIC - [ ] **Model metrics** / **Charts**: `train_loss` of the `metric steps` run drawn over steps
# MAGIC - [ ] **Artifacts**: `confusion_matrix.png`, `features.json` and the `model` folder with `MLmodel`,
# MAGIC       the **signature** (input and output schema) and the input example
# MAGIC - [ ] The **Models** tab of the experiment (MLflow 3 LoggedModels) with each model's metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Finding the best run with the Client API (S1.11)

# COMMAND ----------

from mlflow import MlflowClient

from db_ml_associate.ch04_tracking.best_run import find_best_run, runs_as_dataframe

client = MlflowClient()
best = find_best_run(client, experiment.experiment_id, "test_f1", mode="max")
print("best run:", best.info.run_id, best.data.params, best.data.metrics)
display(runs_as_dataframe(EXPERIMENT)[["run_id", "tags.mlflow.runName", "params.C", "metrics.test_f1"]])

# COMMAND ----------

# MAGIC %md
# MAGIC > **⚠️ Wrong approach:** ordering by a score in ascending order returns the *worst* run.

# COMMAND ----------

from db_ml_associate.ch04_tracking import pitfalls

wrong = pitfalls.wrong_order_by_ascending_for_max_metric(client, experiment.experiment_id, "test_f1")
right = pitfalls.correct_order_by_descending_for_max_metric(client, experiment.experiment_id, "test_f1")
print("ASC picks test_f1 =", wrong.data.metrics["test_f1"], "| DESC picks test_f1 =", right.data.metrics["test_f1"])

# COMMAND ----------

# MAGIC %md
# MAGIC **Self-check**: name three things you can see in the MLflow UI for a run without writing code.
