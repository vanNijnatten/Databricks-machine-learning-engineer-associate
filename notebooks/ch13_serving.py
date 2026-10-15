# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 13 tutorial: Real-time serving
# MAGIC **Objectives**: S4.02, S4.05, S4.06 · **Runtime**: 18 LTS ML · **Requires**: Model Serving (paid) · **Free Edition**: partial
# MAGIC **Estimated time**: 45 min · **Clean-up**: delete the endpoint (last cell, or `99_cleanup`)
# MAGIC
# MAGIC Wrap the churn classifier in a custom pyfunc model, deploy it to a Model Serving endpoint, query it in real time,
# MAGIC and split traffic 80/20 between two versions.
# MAGIC
# MAGIC > **Requires:** Model Serving. A `Small` endpoint with scale-to-zero costs little while idle but is billed while
# MAGIC > it serves; creating it takes several minutes. Free Edition allows only a few small CPU endpoints.
# MAGIC > **Delete the endpoint** with the last cell when you are done.
# MAGIC
# MAGIC Run `00_setup` first.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

dbutils.widgets.dropdown("keep_endpoint", "no", ["no", "yes"], "Keep the endpoint at the end")
from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient

from db_ml_associate.ch09_pipelines.sklearn_pipeline import build_sklearn_churn_pipeline
from db_ml_associate.common import CHURN_FEATURES, CHURN_LABEL, split_frame

ENDPOINT = "churn-endpoint"
MODEL_NAME = uc_name("churn_risk_model")
FEATURES = list(CHURN_FEATURES)
workspace = WorkspaceClient()
client = MlflowClient()
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/ml-associate-ch13")
train, test = split_frame(spark.table(uc_name("customer_churn")).toPandas(), CHURN_LABEL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — A custom pyfunc model (S4.02)
# MAGIC
# MAGIC `ChurnRiskModel` returns the churn probability **and** a risk band — business logic a plain scikit-learn model
# MAGIC cannot serve. Two versions are registered: C = 1.0 and C = 0.1.

# COMMAND ----------

from db_ml_associate.ch05_registry.registry import register_model
from db_ml_associate.ch13_serving.custom_model import log_custom_model

versions = []
for c in (1.0, 0.1):
    classifier = build_sklearn_churn_pipeline().set_params(model__C=c).fit(train[FEATURES], train[CHURN_LABEL])
    model_info = log_custom_model(classifier, test[FEATURES].head(5))
    versions.append(register_model(model_info.model_uri, MODEL_NAME).version)
print(MODEL_NAME, "versions", versions)
display(mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{versions[0]}").predict(test[FEATURES].head(5)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Deploy and query for real-time inference (S4.02, S4.05)

# COMMAND ----------

from db_ml_associate.ch13_serving.workspace import ServedVersion, create_endpoint, endpoint_config

champion = ServedVersion("champion", MODEL_NAME, str(versions[0]))
challenger = ServedVersion("challenger", MODEL_NAME, str(versions[1]))
existing = {endpoint.name for endpoint in workspace.serving_endpoints.list()}
if ENDPOINT not in existing:
    create_endpoint(workspace, ENDPOINT, endpoint_config([champion], {"champion": 100}))  # several minutes
print(workspace.serving_endpoints.get(ENDPOINT).state)

# COMMAND ----------

import mlflow.deployments

from db_ml_associate.ch13_serving.workspace import dataframe_split_payload, query_endpoint

deploy_client = mlflow.deployments.get_deploy_client("databricks")
print("request body:", dataframe_split_payload(test[FEATURES].head(2)))
print("predictions:", query_endpoint(deploy_client, ENDPOINT, test[FEATURES].head(5)))

# COMMAND ----------

# MAGIC %md
# MAGIC The same request over REST: `POST https://<workspace>/serving-endpoints/churn-endpoint/invocations` with a
# MAGIC bearer token and the JSON body printed above. Open **Serving → churn-endpoint** to see the **Query endpoint**
# MAGIC button, metrics and logs.

# COMMAND ----------

# MAGIC %md
# MAGIC > **⚠️ Wrong approach:** a request body without one of the documented keys (`dataframe_split`,
# MAGIC > `dataframe_records`, `instances`, `inputs`) is rejected.

# COMMAND ----------

from db_ml_associate.ch13_serving import pitfalls

print("bare rows:", pitfalls.payload_problems(pitfalls.wrong_payload_without_input_key()))
print("dataframe_split:", pitfalls.payload_problems(pitfalls.correct_payload()) or "valid")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Split traffic between two versions (S4.06)
# MAGIC
# MAGIC Serve both versions on the same endpoint; 80 % of requests go to the champion, 20 % to the challenger.
# MAGIC Percentages must add up to 100.

# COMMAND ----------

from db_ml_associate.ch13_serving.workspace import update_traffic

update_traffic(workspace, ENDPOINT, [champion, challenger], {"champion": 80, "challenger": 20})
for route in workspace.serving_endpoints.get(ENDPOINT).config.traffic_config.routes:
    print(route.served_model_name, route.traffic_percentage, "%")

# COMMAND ----------

predictions = [query_endpoint(deploy_client, ENDPOINT, test[FEATURES].head(1))[0] for _ in range(10)]
display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC Each request is routed by the split, so repeated identical requests can get different answers. To test one
# MAGIC version directly, call `/serving-endpoints/churn-endpoint/served-models/challenger/invocations` — that bypasses
# MAGIC the traffic split.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean-up

# COMMAND ----------

from db_ml_associate.ch13_serving.workspace import delete_endpoint

if dbutils.widgets.get("keep_endpoint") == "no":
    delete_endpoint(workspace, ENDPOINT)
    print("endpoint deleted")
else:
    print("endpoint kept — remember to run 99_cleanup")

# COMMAND ----------

# MAGIC %md
# MAGIC **Self-check**: how would you move from 80/20 to 100 % challenger without downtime?
