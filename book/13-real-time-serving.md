---
chapter: 13
title: Real-time model serving
objectives: [S4.02, S4.05, S4.06]
prerequisites: [5, 12]
reading_minutes: 70
tutorial:
  notebook: notebooks/ch13_serving.py
  minutes: 45
  requires: [unity_catalog, model_serving]
  free_edition: partial
  cleanup: true
  extra_files: []
  steps:
    - id: custom-model
      objectives: [S4.02]
    - id: deploy-and-query
      objectives: [S4.02, S4.05]
    - id: traffic-split
      objectives: [S4.06]
references: [model-serving, custom-models, create-serving-endpoints, query-custom-endpoints, serve-multiple-models, ai-query, free-edition-limitations, model-serving-intro]
---

# 13. Real-time model serving

> **Exam mapping**: Section 4 — Model Deployment · objectives S4.02, S4.05, S4.06 ·
> **Runtime**: any (Model Serving is serverless) · **Read after**: chapters 5 and 12.

When a prediction must come back while a user waits — approve a payment, personalise a page — the
model runs behind a **REST endpoint**. **Mosaic AI Model Serving** turns a registered MLflow model
into such an endpoint on serverless compute that scales with demand
([Model Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/)). The exam asks you
to deploy a custom model, query it, and split traffic between model versions.

> **Requires:** Model Serving (capped on Free Edition: few endpoints, CPU only). An endpoint costs
> money while it is running; enable scale-to-zero for learning and **delete the endpoint** at the end
> of the tutorial (`notebooks/99_cleanup.py`).

> **Tutorial:** [`notebooks/ch13_serving.py`](../notebooks/ch13_serving.py) · about 45 minutes · **Requires:** Model Serving (billed while serving; scale-to-zero enabled) · **Free Edition:** partly · **Clean-up:** delete the endpoint. Run [`notebooks/00_setup.py`](../notebooks/00_setup.py) once first.

## Learning objectives

- Wrap custom logic in an MLflow `PythonModel` and deploy it to a serving endpoint.
- Create an endpoint in the UI, with the REST API, the MLflow deployments client or the Databricks
  SDK.
- Query an endpoint with `dataframe_split` and the other input formats.
- Split traffic between served model versions for A/B tests and canary releases.

## Custom models

<!-- objectives: S4.02 -->

Model Serving serves any MLflow model. A **custom model** is a `mlflow.pyfunc.PythonModel` that adds
your own logic — preprocessing, post-processing, several models, business rules — around a trained
model ([Deploy custom models](https://docs.databricks.com/aws/en/machine-learning/model-serving/custom-models)):

<!-- snippet: src/db_ml_associate/ch13_serving/custom_model.py#python_model -->
```python
class ChurnRiskModel(mlflow.pyfunc.PythonModel):
    """Return the churn probability and a risk band the retention team can act on."""

    def load_context(self, context: Any) -> None:
        # Runs once when the endpoint (or load_model) starts: load artifacts here, not in predict.
        self.classifier = mlflow.sklearn.load_model(context.artifacts["classifier"])

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        probability = self.classifier.predict_proba(model_input)[:, 1]
        risk_band = np.select([probability >= 0.7, probability >= 0.4], ["high", "medium"], default="low")
        return pd.DataFrame({"churn_probability": probability, "risk_band": risk_band})
```

- `load_context(context)` runs **once** when the model is loaded (endpoint start): load artifacts
  there. `context.artifacts` maps names to the files logged with the model.
- `predict(context, model_input, params)` runs for every request and returns predictions — here a
  DataFrame with a probability and a risk band.

Log it with its artifacts, a **signature** and an input example — required to register it in Unity
Catalog and used by the endpoint to validate requests:

<!-- snippet: src/db_ml_associate/ch13_serving/custom_model.py#log_pyfunc -->
```python
with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        name="churn_risk_model",
        python_model=ChurnRiskModel(),
        artifacts={"classifier": str(classifier_path)},  # copied into the logged model
        signature=infer_signature(input_example, example_output),  # required for Unity Catalog
        input_example=input_example,
    )
```

Register it (chapter 5), then serve it. Dependencies are captured from the model's environment files;
add extras with `pip_requirements` / `extra_pip_requirements`, and code files with `code_paths`.

## Creating an endpoint

<!-- objectives: S4.02, S4.05 -->

([Create custom model serving endpoints](https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints);
[Tutorial: Deploy and query a custom model](https://docs.databricks.com/aws/en/machine-learning/model-serving/model-serving-intro))

**UI:** **Serving → Create serving endpoint** → name → served entity **My models – Unity Catalog** →
model and version → compute type (CPU or GPU) → size → scale to zero → **Create**. The endpoint shows
*Not Ready* until the container is built and *Ready* afterwards.

**Python (Databricks SDK):**

<!-- snippet: src/db_ml_associate/ch13_serving/workspace.py#endpoint_config -->
```python
return EndpointCoreConfigInput(
    served_entities=_served_entities(versions),  # e.g. champion v1 and challenger v2
    traffic_config=_traffic(versions, traffic),  # e.g. {"champion": 80, "challenger": 20}
)
```

<!-- snippet: src/db_ml_associate/ch13_serving/workspace.py#create_endpoint -->
```python
# from databricks.sdk import WorkspaceClient; workspace = WorkspaceClient()
return workspace.serving_endpoints.create_and_wait(name=name, config=config)
```

**REST:** `POST /api/2.0/serving-endpoints` with the same `config`; **MLflow deployments client:**
`get_deploy_client("databricks").create_endpoint(name=..., config=...)`.

| Setting | Meaning |
|---|---|
| `entity_name`, `entity_version` | The Unity Catalog model and version to serve |
| `workload_size` | Concurrency: **Small** (0–4 requests), **Medium** (8–16), **Large** (16–64) |
| `workload_type` | CPU, CPU_MEDIUM, CPU_LARGE or a GPU type |
| `scale_to_zero_enabled` | Scales to zero after 30 minutes without traffic; the next request has a cold start — not for latency-critical production |

The creator needs `USE CATALOG`, `USE SCHEMA` and `EXECUTE` on the model; use a long-lived service
principal for production endpoints.

## Querying an endpoint

<!-- objectives: S4.05 -->

An endpoint accepts `POST https://<workspace>/serving-endpoints/<name>/invocations` with a JSON
body containing **one** of these keys
([Query serving endpoints for custom models](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints)):

| Key | Shape | Use |
|---|---|---|
| **`dataframe_split`** (recommended) | `{"columns": [...], "data": [[...], ...]}` | Tabular input; preserves column order |
| `dataframe_records` | `[{"col": value, ...}, ...]` | Tabular input as a list of records |
| `instances` | Row-format tensors | Tensor models, same first dimension |
| `inputs` | Column-format tensors | Tensor models |

<!-- snippet: src/db_ml_associate/ch13_serving/workspace.py#dataframe_split -->
```python
return {"dataframe_split": {"columns": list(frame.columns), "data": frame.to_numpy().tolist()}}
```

With curl:

<!-- illustrative -->
```bash
curl -X POST "https://<workspace>/serving-endpoints/churn-endpoint/invocations" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["tenure_months", "monthly_charges", "data_usage_gb", "support_calls", "contract_type", "payment_method"], "data": [[12, 70.5, 3.1, 2, "one-year", "card"]]}}'
```

With the MLflow deployments client:

<!-- snippet: src/db_ml_associate/ch13_serving/workspace.py#query_endpoint -->
```python
# import mlflow.deployments; deploy_client = mlflow.deployments.get_deploy_client("databricks")
response = deploy_client.predict(endpoint=name, inputs=dataframe_split_payload(frame))
return list(response["predictions"])
```

The response is JSON with a `predictions` key. You can also test an endpoint from the **Query
endpoint** button in the Serving UI, and call it from SQL with `ai_query` (chapter 12). For
production clients use OAuth tokens of a service principal.

## Splitting traffic between models

<!-- objectives: S4.06 -->

One endpoint can serve **several model versions** and route a percentage of requests to each
([Serve multiple models to a model serving endpoint](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-multiple-models-to-serving-endpoint)):

- `served_entities` lists each version under a name (e.g. `champion`, `challenger`);
- `traffic_config.routes` assigns each name a `traffic_percentage`; **the percentages must add up to
  100**.

Change the split on a running endpoint without downtime — the old configuration keeps serving until the
new one is ready:

<!-- snippet: src/db_ml_associate/ch13_serving/workspace.py#update_traffic -->
```python
return workspace.serving_endpoints.update_config_and_wait(
    name=name,
    served_entities=_served_entities(versions),
    traffic_config=_traffic(versions, traffic),
)
```

Typical uses:

- **Canary release**: send 5–10 % to a new version, watch errors and latency, then raise it to 100 %.
- **A/B test / online champion–challenger**: 80/20 or 50/50 and compare business metrics (chapter 5).
- **Rollback**: set the old version back to 100 %.

To query **one** served model directly, bypassing the split, POST to
`/serving-endpoints/<endpoint>/served-models/<served-model-name>/invocations` — traffic settings are
ignored for such calls.

> **Exam tip:** "split data between endpoints for real-time inference" → one endpoint with several
> `served_entities` and `traffic_config.routes` whose percentages sum to 100.

## Choosing an approach

| Need | Use |
|---|---|
| Responses in milliseconds per request | Model Serving endpoint |
| Pre/post-processing or several models behind one API | Custom `PythonModel` |
| Roll out a new version gradually | Traffic split (canary) |
| Compare two versions on live traffic | Traffic split (A/B), then promote with aliases |
| Scoring whole tables | Batch inference instead (chapter 12) |
| Query from SQL | `ai_query` |
| Lowest cost, tolerate cold starts | Scale to zero |

## Pitfalls

**1. Traffic that does not add up to 100 %.**

> **⚠️ Wrong approach:** routes of 80 % and 30 %. The configuration is rejected (the chapter's
> validator raises the same error before calling the API).

**2. Request bodies without an input key.** `{"data": [...]}` is not a valid body; use
`dataframe_split` (or one of the other documented keys).

**3. Loading artifacts in `predict`.** Every request would reload them; load once in
`load_context`.

**4. Scale to zero in production.** The first request after idle time waits for a cold start.

**5. Models without a signature.** Cannot be registered in Unity Catalog and give poor request
validation.

**6. Forgetting to delete tutorial endpoints.** They keep running (and billing) unless they scale to
zero or are deleted.

## Summary

- Custom models are `mlflow.pyfunc.PythonModel`s (`load_context` once, `predict` per request), logged
  with a signature and registered in Unity Catalog.
- Create endpoints in the Serving UI, with the REST API, the MLflow deployments client or the
  Databricks SDK; size them Small/Medium/Large; scale to zero for non-critical use.
- Query `…/serving-endpoints/<name>/invocations` with `dataframe_split` (recommended), records, or
  tensor inputs; responses contain `predictions`.
- Split traffic with `served_entities` + `traffic_config.routes` summing to 100 %.

## Review questions

<!-- questions: ch13 -->

### Question 1

A team wants its real-time endpoint to return not only a churn probability but also a risk band computed from business rules, using the existing scikit-learn model. What should they deploy?

- **A.** An mlflow.pyfunc.PythonModel that loads the classifier in load_context and applies the rules in predict, registered in Unity Catalog
- **B.** The scikit-learn model plus a separate notebook that post-processes every request
- **C.** A Lakeflow pipeline that calls the endpoint
- **D.** Two endpoints, one per output column

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — A custom pyfunc model packages preprocessing, post-processing and models as one MLflow model that Model Serving deploys behind a single endpoint.

Why the other options are wrong:

- **B**: Notebooks are not part of the request path of an endpoint.
- **C**: Pipelines process tables; they do not add logic to endpoint responses.
- **D**: Splitting outputs across endpoints doubles cost and still lacks the rules.

Objective S4.02 · [Revisit](13-real-time-serving.md#custom-models)

</details>

### Question 2

A custom PythonModel downloads a 2 GB lookup file inside predict(), and endpoint latency is very high. What should the developer change?

- **A.** Log the file as a model artifact and load it once in load_context
- **B.** Increase the workload size to Large
- **C.** Enable scale to zero
- **D.** Query the endpoint with dataframe_records instead of dataframe_split

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — load_context runs once when the model starts; artifacts logged with the model are available there, so requests no longer pay the loading cost.

Why the other options are wrong:

- **B**: More concurrency does not remove the per-request download.
- **C**: Scale to zero adds cold starts and does not fix per-request work.
- **D**: The input format is not the bottleneck.

Objective S4.02 · [Revisit](13-real-time-serving.md#custom-models)

</details>

### Question 3

Which request body is the recommended format for sending tabular input to a custom model serving endpoint?

- **A.** {"dataframe_split": {"columns": ["age", "plan"], "data": [[34, "basic"]]}}
- **B.** {"rows": [[34, "basic"]]}
- **C.** {"sql": "SELECT * FROM customers"}
- **D.** {"model": "prod.ml.churn", "version": 3}

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — dataframe_split sends the column names once and the rows as lists; it is the recommended tabular format.

Why the other options are wrong:

- **B**: rows is not one of the accepted input keys.
- **C**: Endpoints receive data, not SQL queries.
- **D**: The endpoint already knows which model it serves; the body carries input data.

Objective S4.05 · [Revisit](13-real-time-serving.md#querying-an-endpoint)

</details>

### Question 4

Which TWO are valid ways to get real-time predictions from a Databricks Model Serving endpoint named churn-endpoint?

- **A.** POST to /serving-endpoints/churn-endpoint/invocations with a dataframe_split body
- **B.** mlflow.deployments.get_deploy_client("databricks").predict(endpoint="churn-endpoint", inputs=payload)
- **C.** mlflow.register_model("churn-endpoint", "prod.ml.churn")
- **D.** spark.read.table("churn-endpoint")
- **E.** dbutils.data.summarize("churn-endpoint")

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — Endpoints are queried over REST or with the MLflow deployments client (and from SQL with ai_query).

Why the other options are wrong:

- **C**: register_model registers models; it does not query endpoints.
- **D**: Endpoints are not tables.
- **E**: summarize profiles DataFrames.

Objective S4.05 · [Revisit](13-real-time-serving.md#querying-an-endpoint)

</details>

### Question 5

A team deploys a fraud model to an endpoint for card authorisations that must always answer within 100 ms, at any time of day. Which setting should they avoid?

- **A.** Scale to zero
- **B.** A served entity from Unity Catalog
- **C.** A model signature
- **D.** A Small or Medium workload size sized for the load

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — An endpoint that scaled to zero needs a cold start for the next request; Databricks does not recommend it for production endpoints that need guaranteed capacity.

Why the other options are wrong:

- **B**: Unity Catalog is the recommended registry for served models.
- **C**: Signatures improve request validation and are required for Unity Catalog.
- **D**: Sizing for the expected load is appropriate.

Objective S4.05 · [Revisit](13-real-time-serving.md#creating-an-endpoint)

</details>

### Question 6

A new model version 8 should receive 10% of live requests while version 7 keeps 90%, behind the same endpoint. How is this configured?

- **A.** Serve both versions as served_entities and set traffic_config.routes to 90 and 10 percent
- **B.** Create a second endpoint and ask 10% of clients to call it
- **C.** Move the champion alias to version 8 for 10% of the day
- **D.** Set workload_size to 10 for version 8

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — One endpoint can serve several versions; traffic_config routes split requests by percentage, which must add up to 100.

Why the other options are wrong:

- **B**: Splitting clients manually is error-prone and not a traffic split.
- **C**: An alias points to one version at a time; it does not split traffic.
- **D**: workload_size controls concurrency, not routing.

Objective S4.06 · [Revisit](13-real-time-serving.md#splitting-traffic-between-models)

</details>

### Question 7

An engineer updates an endpoint with routes champion=70 and challenger=40. What happens?

- **A.** The update is rejected because the traffic percentages must add up to 100
- **B.** Requests are scaled proportionally to 70/40
- **C.** The challenger receives the remaining 30% only
- **D.** Both models receive every request

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Traffic routes must sum to exactly 100 percent.

Why the other options are wrong:

- **B**: Percentages are not normalised automatically.
- **C**: The configuration is invalid; nothing is inferred.
- **D**: Each request goes to one served model according to the split.

Objective S4.06 · [Revisit](13-real-time-serving.md#pitfalls)

</details>

### Question 8

During an A/B test with an 80/20 split, a tester needs to send requests only to the challenger served model. How can they do that without changing the split?

- **A.** POST to /serving-endpoints/<endpoint>/served-models/challenger/invocations
- **B.** Add the header X-Traffic: 100
- **C.** Temporarily set the challenger's route to 100 percent
- **D.** It is impossible while a traffic split is configured

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Querying a specific served model through its served-models path ignores the endpoint's traffic settings.

Why the other options are wrong:

- **B**: No such header controls routing.
- **C**: Changing the split affects all clients.
- **D**: Direct served-model invocation exists for this purpose.

Objective S4.06 · [Revisit](13-real-time-serving.md#splitting-traffic-between-models)

</details>

### Question 9

Which Databricks SDK call creates a serving endpoint for version 3 of the Unity Catalog model prod.ml.churn and waits until it is ready?

- **A.** w.serving_endpoints.create_and_wait(name="churn", config=EndpointCoreConfigInput(served_entities=[ServedEntityInput(entity_name="prod.ml.churn", entity_version="3", workload_size="Small", scale_to_zero_enabled=True)]))
- **B.** w.jobs.create(name="churn", model="prod.ml.churn")
- **C.** mlflow.register_model("prod.ml.churn", "churn-endpoint")
- **D.** w.clusters.create(spark_version="18.0.x-cpu-ml-scala2.13")

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — serving_endpoints.create_and_wait takes an endpoint name and a config listing the served entities.

Why the other options are wrong:

- **B**: Jobs run workflows; they do not create endpoints.
- **C**: register_model registers models.
- **D**: Clusters are classic compute; Model Serving is serverless.

Objective S4.02 · [Revisit](13-real-time-serving.md#creating-an-endpoint)

</details>
<!-- /questions -->
