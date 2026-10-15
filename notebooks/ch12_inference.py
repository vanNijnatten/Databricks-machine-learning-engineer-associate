# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 12 tutorial: Batch and streaming inference
# MAGIC **Objectives**: S4.01, S4.03, S4.04 · **Runtime**: 18 LTS ML · **Requires**: Unity Catalog; Lakeflow pipelines for step 4 · **Free Edition**: partial
# MAGIC **Estimated time**: 45 min · **Clean-up**: delete the pipeline when done (or run `99_cleanup`)
# MAGIC
# MAGIC Score the champion churn model with pandas, with a Spark UDF and with SQL `ai_query`, then score a stream —
# MAGIC first with Structured Streaming, then with a Lakeflow pipeline (the exam's "Delta Live Tables").
# MAGIC Run `00_setup` and the chapter 5 tutorial first (they create the tables and `churn_model@champion`).

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

from db_ml_associate.common import CHURN_FEATURES

MODEL_URI = f"models:/{uc_name('churn_model')}@champion"
churn = spark.table(uc_name("customer_churn"))
FEATURES = list(CHURN_FEATURES)
print("model:", MODEL_URI)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Batch inference with pandas (S4.03)
# MAGIC
# MAGIC Load the model **once** with `mlflow.pyfunc.load_model`, then predict on a whole DataFrame. Fine for data that
# MAGIC fits in the driver's memory.

# COMMAND ----------

from db_ml_associate.ch12_inference.batch import batch_predict_pandas

scored_pdf = batch_predict_pandas(MODEL_URI, churn.limit(1000).toPandas())
display(scored_pdf[["customer_id", *FEATURES, "prediction"]].head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC > **⚠️ Wrong approach:** loading the model inside a loop over rows loads it once per row.

# COMMAND ----------

from db_ml_associate.ch12_inference import pitfalls

sample = churn.limit(20).toPandas()
print("model loads, per-row loop:", pitfalls.wrong_load_model_per_row(MODEL_URI, sample).loads)
print("model loads, load once:   ", pitfalls.correct_load_once(MODEL_URI, sample).loads)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Batch inference at scale with a Spark UDF (S4.01)

# COMMAND ----------

from db_ml_associate.ch12_inference.batch import batch_predict_spark

scored = batch_predict_spark(spark, MODEL_URI, churn, FEATURES)
scored.select("customer_id", "prediction").write.mode("overwrite").saveAsTable(uc_name("churn_scores"))
display(spark.table(uc_name("churn_scores")).groupBy("prediction").count())

# COMMAND ----------

# MAGIC %md
# MAGIC `ai_query` scores from SQL by calling a **Model Serving endpoint** — the batch path for SQL users. It needs an
# MAGIC endpoint that returns one number per row (for example one serving `churn_model`), so it is shown here but not
# MAGIC run. The chapter 13 tutorial creates endpoints.

# COMMAND ----------

from db_ml_associate.ch12_inference.batch import AI_QUERY_SQL

print(AI_QUERY_SQL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Streaming inference with Structured Streaming (S4.01, S4.04)
# MAGIC
# MAGIC The same UDF scores a streaming DataFrame. First create a source table and append "new events" to it.

# COMMAND ----------

RAW_EVENTS = uc_name("customer_events_raw")
CHECKPOINT = f"/Volumes/{CATALOG}/{SCHEMA}/files/checkpoints/churn_stream"
dbutils.fs.rm(CHECKPOINT, True)
churn.filter("customer_id <= 2500").drop("churned").write.mode("overwrite").saveAsTable(RAW_EVENTS)

# COMMAND ----------

from db_ml_associate.ch12_inference.streaming import score_stream

stream = spark.readStream.table(RAW_EVENTS)
query = (
    score_stream(spark, stream, MODEL_URI, FEATURES)
    .writeStream.option("checkpointLocation", CHECKPOINT)
    .trigger(availableNow=True)  # process everything available, then stop
    .toTable(uc_name("churn_scores_stream"))
)
query.awaitTermination()
print("scored rows:", spark.table(uc_name("churn_scores_stream")).count())

# COMMAND ----------

churn.filter("customer_id > 2500").drop("churned").write.mode("append").saveAsTable(RAW_EVENTS)  # new events arrive
query = (
    score_stream(spark, spark.readStream.table(RAW_EVENTS), MODEL_URI, FEATURES)
    .writeStream.option("checkpointLocation", CHECKPOINT)
    .trigger(availableNow=True)
    .toTable(uc_name("churn_scores_stream"))
)
query.awaitTermination()
print(
    "scored rows after the second batch:", spark.table(uc_name("churn_scores_stream")).count()
)  # only new rows were added

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Streaming inference with a Lakeflow pipeline (S4.04)
# MAGIC
# MAGIC > **Platform change:** Delta Live Tables is now **Lakeflow pipelines**; exam answers still say
# MAGIC > "Delta Live Tables pipeline that applies the model as a Spark UDF".
# MAGIC
# MAGIC 1. Sidebar → **Jobs & Pipelines** → **Create** → **ETL pipeline**. Name it `ml-associate-churn-scoring`.
# MAGIC 2. Default catalog `ml_associate`, default schema `tutorial`; **serverless** compute.
# MAGIC 3. Source code: add the file `notebooks/ch12_pipeline_source.py` from this Git folder.
# MAGIC 4. **Configuration** (Settings → Advanced): `churn.model_uri` = the value printed below and
# MAGIC    `churn.source_table` = `ml_associate.tutorial.customer_events_raw` (adjust if you changed the widgets).
# MAGIC 5. **Start** the pipeline. The graph shows `customer_events` (streaming table) → `churn_predictions`.
# MAGIC 6. Run the next cell to append more events, start the pipeline again and see that only the new rows are scored.

# COMMAND ----------

print("churn.model_uri    =", MODEL_URI)
print("churn.source_table =", RAW_EVENTS)

# COMMAND ----------

churn.filter("customer_id <= 100").drop("churned").write.mode("append").saveAsTable(RAW_EVENTS)
print("appended 100 events; start the pipeline again, then run the next cell")

# COMMAND ----------

display(spark.table(uc_name("churn_predictions")).groupBy("churn_prediction").count())

# COMMAND ----------

# MAGIC %md
# MAGIC **Clean-up**: delete the pipeline (Jobs & Pipelines → … → Delete) or run `99_cleanup`.
# MAGIC
# MAGIC **Self-check**: why is calling a Model Serving endpoint from inside the pipeline worse than the Spark UDF?
