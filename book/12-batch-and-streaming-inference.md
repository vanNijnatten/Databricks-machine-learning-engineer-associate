---
chapter: 12
title: Batch and streaming inference
objectives: [S4.01, S4.03, S4.04]
prerequisites: [5, 9]
reading_minutes: 70
tutorial:
  notebook: notebooks/ch12_inference.py
  minutes: 45
  requires: [unity_catalog, lakeflow]
  free_edition: partial
  cleanup: false
  extra_files: [notebooks/ch12_pipeline_source.py]
  steps:
    - id: pandas-batch
      objectives: [S4.03]
    - id: spark-udf
      objectives: [S4.01]
    - id: structured-streaming
      objectives: [S4.01, S4.04]
    - id: lakeflow-pipeline
      objectives: [S4.04]
references: [model-inference, ai-query, lakeflow-pipelines, lakeflow-transform, lakeflow-python-ref, model-serving, model-lifecycle-uc]
---

# 12. Batch and streaming inference

> **Exam mapping**: Section 4 — Model Deployment · objectives S4.01, S4.03, S4.04 ·
> **Runtime**: Databricks Runtime 18 LTS ML · **Read after**: chapters 5 and 9.

A registered model creates value only when it scores new data. Databricks offers three ways to do
that — **batch**, **streaming** and **real-time** — and the exam asks you to pick the right one,
to run batch inference with pandas, and to know how streaming inference works in a Lakeflow pipeline
(formerly Delta Live Tables). Real-time serving gets its own chapter (13).

> **Tutorial:** [`notebooks/ch12_inference.py`](../notebooks/ch12_inference.py) · about 45 minutes · **Requires:** Unity Catalog; a Lakeflow pipeline for the last step; run the chapter 5 tutorial first · **Free Edition:** partly · **Clean-up:** delete the pipeline. Run [`notebooks/00_setup.py`](../notebooks/00_setup.py) once first.

## Learning objectives

- Compare batch, streaming and real-time inference by latency, throughput, cost and freshness.
- Score a DataFrame with pandas and at scale with an MLflow Spark UDF.
- Call a serving endpoint from SQL with `ai_query`.
- Explain how streaming inference is performed with a Lakeflow (Delta Live Tables) pipeline.

## Choosing a serving approach

<!-- objectives: S4.01 -->

| | Batch | Streaming | Real-time |
|---|---|---|---|
| Scores | A whole table on a schedule (hourly, nightly) | New records continuously, in micro-batches | One request at a time, on demand |
| Latency | Minutes to hours | Seconds to minutes | Milliseconds |
| Throughput | Highest (millions of rows per run) | High | Per request |
| Cost | Lowest — compute only while the job runs | Medium — a pipeline runs continuously or frequently | Highest per prediction — an endpoint is always ready (scale-to-zero can help) |
| Freshness | As of the last run | Near real time | Immediate |
| Typical tool | Spark UDF / `fe.score_batch` / `ai_query` in a job | Lakeflow pipeline or Structured Streaming with a Spark UDF | Mosaic AI Model Serving endpoint (chapter 13) |
| Example | Nightly churn scores for the marketing team | Flag suspicious sensor readings within a minute | Approve a card payment while the customer waits |

Most use cases are batch: the most common error in architecture questions is choosing real-time
serving when nobody consumes the predictions instantly. Databricks recommends AI Functions
(`ai_query`) for batch inference on data in Databricks, and Model Serving for real-time inference
([Model inference](https://docs.databricks.com/aws/en/machine-learning/model-inference/)).

> **Exam tip:** look for the latency requirement. "Daily report" → batch. "Within seconds of the event,
> high volume, compute must scale with load" → streaming pipeline. "Response inside a user request" →
> real-time endpoint.

## Batch inference with pandas

<!-- objectives: S4.03 -->

When the data fits in the driver's memory, load the registered model once with `mlflow.pyfunc` and
call `predict` on the whole DataFrame:

<!-- snippet: src/db_ml_associate/ch12_inference/batch.py#pandas_batch -->
```python
model = mlflow.pyfunc.load_model(model_uri)  # load ONCE, e.g. "models:/prod.ml.churn@champion"
scored = frame.copy()
scored["prediction"] = model.predict(frame)  # the signature selects and checks the input columns
```

- `mlflow.pyfunc.load_model` returns a generic **PyFuncModel** whatever the flavour (scikit-learn,
  XGBoost, Spark ML), so scoring code does not depend on the training library.
- The model's **signature** enforces input columns and types: missing or renamed columns raise an
  error; extra columns are ignored.
- Load by **alias** (`models:/catalog.schema.model@champion`, chapter 5) so the job uses the current
  production version automatically.
- For a Spark table: `spark.table(...).toPandas()` on a modest table, or the Spark UDF below for large
  data. pandas API on Spark (`pyspark.pandas`) offers a pandas-like syntax on distributed data.

## Batch inference at scale with a Spark UDF

<!-- objectives: S4.03, S4.01 -->

`mlflow.pyfunc.spark_udf` wraps the model in a pandas UDF; Spark sends batches of rows to every
executor, so even a scikit-learn model scores billions of rows in parallel:

<!-- snippet: src/db_ml_associate/ch12_inference/batch.py#spark_udf -->
```python
churn_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="double")
scored = frame.withColumn("prediction", churn_model(F.struct(*features)))  # struct keeps column names
# scored.write.mode("overwrite").saveAsTable("ml_associate.tutorial.churn_scores")
```

In the tests the Spark UDF and pandas produce identical predictions for the same rows. Models logged
with the feature engineering client score with `fe.score_batch` instead, which also looks up the
features (chapter 3).

### `ai_query` from SQL

`ai_query` calls a Model Serving endpoint from SQL — handy for analysts and for batch inference
pipelines written in SQL
([ai_query](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query)):

<!-- snippet: src/db_ml_associate/ch12_inference/sql/ai_query.sql#ai_query -->
```sql
SELECT customer_id,
       ai_query(
         endpoint   => 'churn-endpoint',              -- an existing Model Serving endpoint
         request    => named_struct(
                         'tenure_months', tenure_months,
                         'monthly_charges', monthly_charges,
                         'data_usage_gb', data_usage_gb,
                         'support_calls', support_calls,
                         'contract_type', contract_type,
                         'payment_method', payment_method),
         returnType => 'DOUBLE'
       ) AS churn_prediction
FROM ml_associate.tutorial.customer_churn;
```

The request's struct fields must match the model's input features. `ai_query` needs Databricks
Runtime 15.4 LTS or above (or Databricks SQL, not Classic) and an existing endpoint.

## Streaming inference with Lakeflow pipelines

<!-- objectives: S4.04 -->

> **Platform change:** **Delta Live Tables** is now **Lakeflow pipelines (Spark Declarative
> Pipelines)**. Python code imports `from pyspark import pipelines as dp` instead of `import dlt`
> (still accepted, but replaced)
> ([Lakeflow pipelines Python reference](https://docs.databricks.com/aws/en/ldp/developer/python-ref)).
> Exam answers still say "Delta Live Tables".

A **pipeline** is declarative: you define datasets as functions that return DataFrames, and the
pipeline works out the dependencies, runs them, keeps checkpoints, retries failures and **autoscales**
its compute. A function decorated with `@dp.table` that returns a *streaming* DataFrame defines a
**streaming table**, which processes each new record once.

Streaming inference applies the model as a **Spark UDF** inside a dataset definition
([Transform data with pipelines](https://docs.databricks.com/aws/en/ldp/transform)):

<!-- snippet: src/db_ml_associate/ch12_inference/churn_scoring_pipeline.py#pipeline -->
```python
import mlflow
from pyspark import pipelines as dp  # formerly: import dlt
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

spark = SparkSession.active()
MODEL_URI = spark.conf.get("churn.model_uri", "models:/ml_associate.tutorial.churn_model@champion")
SOURCE_TABLE = spark.conf.get("churn.source_table", "ml_associate.tutorial.customer_events_raw")
FEATURES = ["tenure_months", "monthly_charges", "data_usage_gb", "support_calls", "contract_type", "payment_method"]

# Load the model once as a Spark UDF; the champion alias picks up promotions on the next update.
churn_model = mlflow.pyfunc.spark_udf(spark, model_uri=MODEL_URI, result_type="double")


@dp.table(comment="Customer events arriving incrementally")
def customer_events() -> DataFrame:
    return spark.readStream.table(SOURCE_TABLE)  # streaming read → streaming table


@dp.table(comment="Churn prediction for every new customer event")
def churn_predictions() -> DataFrame:
    events = spark.readStream.table("customer_events")  # another dataset of this pipeline
    return events.withColumn("churn_prediction", churn_model(F.struct(*FEATURES)))
```

Create a pipeline in the workspace, add this file as its source code, set `churn.model_uri` and
`churn.source_table` in the pipeline configuration and start it ([Lakeflow pipelines](https://docs.databricks.com/aws/en/ldp/)). Every update scores the
records that arrived since the last one and appends them to `churn_predictions`.

Why this design answers the exam's streaming questions:

- The model runs **inside** the pipeline as a UDF — no network call per record, so it handles tens of
  thousands of events per second.
- The pipeline **autoscales** with the event volume.
- Calling a serving endpoint from the UDF instead would add latency per record and make the endpoint
  the bottleneck.

The same scoring logic also works in plain **Structured Streaming** (you manage checkpoints and the
sink yourself):

<!-- snippet: src/db_ml_associate/ch12_inference/streaming.py#structured_streaming -->
```python
churn_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="double")
scored = stream.withColumn("churn_prediction", churn_model(F.struct(*features)))
# (scored.writeStream
#     .option("checkpointLocation", "/Volumes/ml_associate/tutorial/checkpoints/churn")
#     .trigger(availableNow=True)        # or processingTime="1 minute" for continuous scoring
#     .toTable("ml_associate.tutorial.churn_scores_stream"))
```

> **Exam tip:** the official sample question's answer is "a Delta Live Tables pipeline that applies the
> algorithm as a Spark UDF". Endpoint calls from a UDF, and Structured Streaming jobs without
> autoscaling, are the distractors.

## Choosing an approach

| Requirement | Choose |
|---|---|
| Small data, notebook or simple job | pandas + `mlflow.pyfunc.load_model` |
| Large table, scheduled | `mlflow.pyfunc.spark_udf` (or `fe.score_batch` with feature tables) in a job |
| SQL users, or batch with an existing endpoint | `ai_query` |
| Continuous scoring of arriving data, managed and autoscaled | Lakeflow pipeline with a Spark UDF |
| Full control of a streaming job | Structured Streaming with a Spark UDF |
| Millisecond responses | Model Serving (chapter 13) |

## Pitfalls

**1. Loading the model per row.** Loading a model takes seconds; doing it for each of 20 rows loads
it 20 times. Load once and predict the whole batch (the tests count the loads).

> **⚠️ Wrong approach:** `for row in rows: mlflow.pyfunc.load_model(uri).predict(row)`

**2. Columns that do not match the signature.** Renaming `tenure_months` to `tenure` makes schema
enforcement raise an `MlflowException`. Keep scoring columns identical to the signature.

**3. Choosing real-time serving for batch needs.** An always-on endpoint costs more than a nightly
job and still needs a client to call it for every row.

**4. Calling a serving endpoint from a streaming UDF.** Adds a network round-trip per record and
couples the stream to the endpoint's capacity; load the model into the pipeline instead.

**5. Hard-coding a model version in a pipeline.** Use the `@champion` alias so promotions apply
automatically.

**6. Writing to tables inside a pipeline function.** Pipeline dataset functions must only return a
DataFrame; the pipeline does the writing.

## Summary

- Batch: cheapest, highest throughput, scheduled; streaming: near real-time for arriving data;
  real-time: milliseconds per request at the highest cost.
- pandas batch inference: `mlflow.pyfunc.load_model(uri).predict(df)`; at scale:
  `mlflow.pyfunc.spark_udf`; from SQL: `ai_query`.
- Streaming inference with Lakeflow (Delta Live Tables): a streaming table whose function applies the
  model as a Spark UDF; the pipeline manages state and autoscales.

## Review questions

<!-- questions: ch12 -->

### Question 1

A marketing team receives a list of customers likely to churn every Monday morning, computed from a 50-million-row customer table. Which serving approach is most appropriate?

- **A.** Batch inference in a scheduled job
- **B.** A real-time Model Serving endpoint called for each customer
- **C.** A continuous streaming pipeline
- **D.** Manual scoring in a notebook by a data scientist each week

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Weekly consumption with no latency requirement is a batch use case: cheapest and highest throughput.

Why the other options are wrong:

- **B**: An always-on endpoint costs more and adds a call per customer for no benefit.
- **C**: Continuous streaming is unnecessary for a weekly list.
- **D**: Manual scoring is not reproducible or automated.

Objective S4.01 · [Revisit](12-batch-and-streaming-inference.md#choosing-a-serving-approach)

</details>

### Question 2

Which TWO statements correctly compare inference approaches?

- **A.** Real-time serving offers the lowest latency per prediction but the highest cost per prediction
- **B.** Batch inference typically has the highest throughput and lowest cost
- **C.** Streaming inference requires a Model Serving endpoint
- **D.** Batch inference always produces fresher predictions than streaming
- **E.** Real-time serving is the right choice whenever predictions are written to a table

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — Real-time endpoints trade cost for millisecond latency; batch jobs process large volumes cheaply on a schedule.

Why the other options are wrong:

- **C**: Streaming inference usually applies the model as a UDF inside the stream, without an endpoint.
- **D**: Batch predictions are only as fresh as the last run; streaming is fresher.
- **E**: Writing predictions to a table is typical of batch or streaming.

Objective S4.01 · [Revisit](12-batch-and-streaming-inference.md#choosing-a-serving-approach)

</details>

### Question 3

A data scientist has a pandas DataFrame new_customers and a registered model prod.ml.churn with the alias champion. Which code performs batch inference?

- **A.** mlflow.pyfunc.load_model("models:/prod.ml.churn@champion").predict(new_customers)
- **B.** mlflow.register_model("models:/prod.ml.churn@champion", new_customers)
- **C.** mlflow.log_model(new_customers, "prod.ml.churn")
- **D.** MlflowClient().get_model_version_by_alias("prod.ml.churn", "champion").predict(new_customers)

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Load the model once with mlflow.pyfunc using the alias URI and call predict on the DataFrame.

Why the other options are wrong:

- **B**: register_model registers a model; it does not score data.
- **C**: log_model logs a model, not data.
- **D**: A ModelVersion entity holds metadata and has no predict method.

Objective S4.03 · [Revisit](12-batch-and-streaming-inference.md#batch-inference-with-pandas)

</details>

### Question 4

A scikit-learn model must score a 2-billion-row Delta table nightly. Collecting the table to pandas fails with out-of-memory errors. What should the data scientist do?

- **A.** Wrap the model with mlflow.pyfunc.spark_udf and apply it to the Spark DataFrame
- **B.** Use a larger driver and call toPandas() anyway
- **C.** Retrain the model with Spark ML before every scoring run
- **D.** Score the rows one at a time in a Python loop

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — A Spark UDF distributes scoring of the single-node model across the cluster's executors.

Why the other options are wrong:

- **B**: Two billion rows will not fit on one driver.
- **C**: Retraining is unnecessary; the scoring step is the problem.
- **D**: A row loop on the driver is extremely slow and still needs the data on the driver.

Objective S4.03 · [Revisit](12-batch-and-streaming-inference.md#batch-inference-at-scale-with-a-spark-udf)

</details>

### Question 5

A batch job calls model.predict(df) and fails with a schema enforcement error mentioning a missing column "tenure_months". The DataFrame has a column called "tenure". What is the fix?

- **A.** Rename the column to tenure_months so it matches the model signature
- **B.** Re-register the model without a version number
- **C.** Convert the DataFrame to a Spark DataFrame
- **D.** Drop all other columns from the DataFrame

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — MLflow checks inputs against the signature; columns must have the names (and compatible types) the model was trained with.

Why the other options are wrong:

- **B**: Registration does not change input validation.
- **C**: The same signature is enforced for Spark input.
- **D**: Extra columns are ignored; the missing one is the problem.

Objective S4.03 · [Revisit](12-batch-and-streaming-inference.md#pitfalls)

</details>

### Question 6

An IoT company must score sensor events with an anomaly model within a minute of arrival. Event volume varies strongly during the day and the compute should resize automatically. Which design fits best?

- **A.** A Delta Live Tables (Lakeflow) pipeline with a streaming table that applies the model as a Spark UDF
- **B.** A Model Serving endpoint called from a Python loop over all events every hour
- **C.** A nightly batch job with a Spark UDF
- **D.** A Structured Streaming job that calls a serving endpoint for every event

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — A declarative pipeline processes new events incrementally, applies the model in-process as a UDF and autoscales its compute with the load.

Why the other options are wrong:

- **B**: An hourly loop misses the one-minute requirement and does not scale.
- **C**: Nightly batches are far too slow.
- **D**: Per-event endpoint calls add latency and make the endpoint a bottleneck, and the job lacks pipeline autoscaling.

Objective S4.04 · [Revisit](12-batch-and-streaming-inference.md#streaming-inference-with-lakeflow-pipelines)

</details>

### Question 7

In a Lakeflow (Delta Live Tables) pipeline written in Python, how is a streaming table that scores new records defined?

- **A.** A function decorated with @dp.table (or @dlt.table) that reads with spark.readStream and adds a column computed by an MLflow Spark UDF
- **B.** A function that calls df.write.saveAsTable inside the pipeline
- **C.** A SQL warehouse query scheduled every second
- **D.** A notebook cell that loops over rows and calls model.predict

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Pipeline datasets are functions returning DataFrames; a streaming read produces a streaming table, and the model is applied with mlflow.pyfunc.spark_udf.

Why the other options are wrong:

- **B**: Dataset functions must not write tables; the pipeline does the writing.
- **C**: Scheduled warehouse queries are not streaming pipelines.
- **D**: Row loops do not scale and are not pipeline definitions.

Objective S4.04 · [Revisit](12-batch-and-streaming-inference.md#streaming-inference-with-lakeflow-pipelines)

</details>

### Question 8

A SQL analyst wants to add churn predictions from an existing custom-model serving endpoint called churn-endpoint to a query result, without writing Python. Which function can they use?

- **A.** ai_query('churn-endpoint', named_struct(...))
- **B.** predict_sql('churn-endpoint')
- **C.** mlflow.pyfunc.spark_udf in a SQL cell
- **D.** CALL MODEL churn-endpoint

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — ai_query calls a Model Serving endpoint from SQL; the struct fields must match the model's inputs.

Why the other options are wrong:

- **B**: predict_sql does not exist.
- **C**: spark_udf is a Python API.
- **D**: CALL MODEL is not Databricks SQL syntax.

Objective S4.03 · [Revisit](12-batch-and-streaming-inference.md#ai_query-from-sql)

</details>

### Question 9

What is the current name of the Databricks product that the exam guide calls Delta Live Tables?

- **A.** Lakeflow pipelines (Spark Declarative Pipelines)
- **B.** Mosaic AI Model Serving
- **C.** Unity Catalog volumes
- **D.** MLflow Projects

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Delta Live Tables was renamed Lakeflow pipelines, built on Spark Declarative Pipelines; the Python module is pyspark.pipelines.

Why the other options are wrong:

- **B**: Model Serving serves models over REST.
- **C**: Volumes store files in Unity Catalog.
- **D**: MLflow Projects package code for reproducible runs.

Objective S4.04 · [Revisit](12-batch-and-streaming-inference.md#streaming-inference-with-lakeflow-pipelines)

</details>
<!-- /questions -->
