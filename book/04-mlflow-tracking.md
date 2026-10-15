---
chapter: 4
title: Experiment tracking with MLflow
objectives: [S1.11, S1.12, S1.13]
prerequisites: [9, 11]
reading_minutes: 80
tutorial:
  notebook: notebooks/ch04_tracking.py
  minutes: 40
  requires: [serverless]
  free_edition: full
  cleanup: false
  extra_files: []
  steps:
    - id: manual-logging
      objectives: [S1.12]
    - id: autolog-and-nested-runs
      objectives: [S1.12]
    - id: mlflow-ui-tour
      objectives: [S1.13]
    - id: best-run
      objectives: [S1.11]
references: [mlflow-tracking, mlflow-experiments, mlflow-runs, mlflow-3, hyperparameter-tuning, mlflow-logged-model]
---

# 4. Experiment tracking with MLflow

> **Exam mapping**: Section 1 — Databricks Machine Learning · objectives S1.11, S1.12, S1.13 ·
> **Runtime**: Databricks Runtime 18 LTS ML (MLflow 3.8) · **Read after**: chapters 9 and 11.

Which hyperparameters produced the model in production? On which data was it trained, and how good
was it? **MLflow** answers these questions by recording every training run. Databricks hosts the
MLflow tracking server for you. The exam expects you to log metrics, artifacts and models manually,
find the best run with the client API, and know what the MLflow UI shows.

> **Tutorial:** [`notebooks/ch04_tracking.py`](../notebooks/ch04_tracking.py) · about 40 minutes · **Requires:** serverless or classic compute · **Free Edition:** yes · **Clean-up:** none. Run [`notebooks/00_setup.py`](../notebooks/00_setup.py) once first.

## Learning objectives

- Explain experiments, runs and logged models, and where Databricks stores them.
- Log parameters, metrics (with steps), artifacts and models manually; use autologging.
- Find the best run programmatically with `MlflowClient.search_runs` and `mlflow.search_runs`.
- Identify the information available in the MLflow UI for experiments, runs and models.

## Experiments, runs and models

<!-- objectives: S1.12, S1.13 -->

| Concept | What it is |
|---|---|
| **Run** | One execution of training code. Holds **parameters** (inputs), **metrics** (numeric results, optionally over steps), **tags** (key–value labels), **artifacts** (files) and the source (notebook revision, Git commit) |
| **Experiment** | A collection of related runs that you compare and search |
| **Logged model** (MLflow 3) | A model as a first-class object with its own ID, linked to the run that created it and to later evaluation runs ([Logged Models](https://docs.databricks.com/aws/en/mlflow/logged-model)) |
| **Registered model** | A named, versioned, governed model in Unity Catalog (chapter 5) |

On Databricks there are two kinds of experiments
([Organize training runs with MLflow experiments](https://docs.databricks.com/aws/en/mlflow/experiments)):

- **Notebook experiment** — created automatically when you start a run without an active
  experiment; it carries the notebook's name and is deleted with it.
- **Workspace experiment** — created in the Workspace browser or with
  `mlflow.set_experiment("/Users/you@example.com/churn")`; any notebook or job can log to it.

> **Platform change:** in MLflow 3, `log_model(name="model")` replaces the deprecated
> `artifact_path=`, `log_model` returns a `models:/<model_id>` URI, a model can be logged without an
> active run, and models are stored under their own path instead of inside the run
> ([Get started with MLflow 3](https://docs.databricks.com/aws/en/mlflow/mlflow-3-install)).

## Logging manually

<!-- objectives: S1.12 -->

Wrap training in `mlflow.start_run()` and log what you want to remember:

<!-- snippet: src/db_ml_associate/ch04_tracking/logging.py#manual_logging -->
```python
with mlflow.start_run(run_name=f"logreg C={c}") as run:
    mlflow.log_params({"C": c, "class_weight": "balanced", "train_rows": len(train)})  # inputs
    model = build_sklearn_churn_pipeline().set_params(model__C=c).fit(features, labels)

    predictions = model.predict(test_features)
    metrics = {
        "test_f1": f1_score(test_labels, predictions),
        "test_roc_auc": roc_auc_score(test_labels, model.predict_proba(test_features)[:, 1]),
    }
    mlflow.log_metrics(metrics)  # results

    figure = ConfusionMatrixDisplay.from_predictions(test_labels, predictions).figure_
    mlflow.log_figure(figure, "confusion_matrix.png")  # any file: plots, reports, data samples
    mlflow.log_dict({"numeric": list(CHURN_NUMERIC), "categorical": list(CHURN_CATEGORICAL)}, "features.json")

    model_info = mlflow.sklearn.log_model(
        model,
        name="model",  # MLflow 3; replaces the deprecated artifact_path
        signature=infer_signature(features, predictions),  # input/output schema
        input_example=features.head(3),
    )
```

| Call | Logs | Example |
|---|---|---|
| `mlflow.log_param(key, value)` / `log_params(dict)` | Inputs: hyperparameters, data versions | `C`, `train_rows` |
| `mlflow.log_metric(key, value, step=None)` / `log_metrics(dict)` | Numeric results; with `step`, a series | `test_f1`, `train_loss` per epoch |
| `mlflow.set_tag(key, value)` / `set_tags(dict)` | Labels for searching and organising | `team=churn`, `stage=baseline` |
| `mlflow.log_artifact(local_path)` / `log_artifacts(dir)` | Any existing file or folder | a CSV of errors |
| `mlflow.log_figure(fig, "x.png")`, `log_dict`, `log_text`, `log_table` | Objects saved straight to an artifact file | confusion matrix, feature list |
| `mlflow.<flavor>.log_model(model, name=..., signature=..., input_example=...)` | The model in MLflow format | `mlflow.sklearn`, `mlflow.spark`, `mlflow.xgboost`, `mlflow.pyfunc` |

A metric logged with steps becomes a line chart in the UI:

<!-- snippet: src/db_ml_associate/ch04_tracking/logging.py#metric_steps -->
```python
for step, loss in enumerate(losses):
    mlflow.log_metric("train_loss", loss, step=step)  # the UI draws a line chart over steps
```

The **signature** records the input columns and types and the output type. Unity Catalog requires
it to register a model, and Model Serving uses it to validate requests (chapters 5 and 13). The
**input example** shows users what a request looks like.

### Autologging

`mlflow.autolog()` (or a flavour-specific `mlflow.sklearn.autolog()`) patches the training library so
that parameters, training metrics and the model are logged automatically
([Track model development using MLflow](https://docs.databricks.com/aws/en/mlflow/tracking)):

<!-- snippet: src/db_ml_associate/ch04_tracking/logging.py#autolog -->
```python
mlflow.sklearn.autolog(log_models=True)  # or mlflow.autolog() for all supported libraries
with mlflow.start_run() as run:
    build_sklearn_churn_pipeline().fit(features, labels)  # params, training_* metrics, model logged
mlflow.sklearn.autolog(disable=True)
```

For scikit-learn, autologged training metrics are prefixed `training_` (e.g. `training_f1_score`).
For Spark ML use `mlflow.pyspark.ml.autolog()`.

### Nested runs

Tuning creates many runs. Group them under a parent with `nested=True`; the child runs carry a
`mlflow.parentRunId` tag and appear indented under the parent in the UI:

<!-- snippet: src/db_ml_associate/ch04_tracking/logging.py#nested_runs -->
```python
with mlflow.start_run(run_name="tune C") as parent:
    for c in cs:
        with mlflow.start_run(run_name=f"C={c}", nested=True):  # child run, linked to the parent
            model = build_sklearn_churn_pipeline().set_params(model__C=c).fit(features, labels)
            mlflow.log_param("C", c)
            mlflow.log_metric("test_f1", f1_score(test_labels, model.predict(test_features)))
```

## Finding the best run with the Client API

<!-- objectives: S1.11 -->

`MlflowClient.search_runs` returns `Run` objects; `order_by` and `max_results=1` give the best run:

<!-- snippet: src/db_ml_associate/ch04_tracking/best_run.py#find_best_run -->
```python
direction = "DESC" if mode == "max" else "ASC"  # DESC for scores (F1, AUC), ASC for losses (RMSE)
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"metrics.{metric} > -1e300",  # only runs that logged the metric
    order_by=[f"metrics.{metric} {direction}"],
    max_results=1,
)
best_run = runs[0]  # best_run.info.run_id, best_run.data.metrics, best_run.data.params
```

- Sort **scores** (F1, AUC, R²) `DESC` and **losses** (RMSE, log loss) `ASC` — the default order is
  ascending.
- `filter_string` uses a SQL-like syntax: `metrics.test_f1 > 0.3`, `params.C = '1.0'` (parameters are
  strings), `tags.team = 'churn'`, `attributes.status = 'FINISHED'`.
- The result exposes `run.info.run_id`, `run.data.metrics`, `run.data.params` and `run.data.tags`.

`mlflow.search_runs` returns the same information as a **pandas DataFrame**, convenient for analysis:

<!-- snippet: src/db_ml_associate/ch04_tracking/best_run.py#search_runs -->
```python
runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["start_time DESC"])
# filter example: mlflow.search_runs(..., filter_string="metrics.test_f1 > 0.3 and params.C = '1.0'")
```

With the best run's ID you can load its model, register it (chapter 5) or compare it with others.

> **Exam tip:** "identify the best run programmatically" → `MlflowClient().search_runs(experiment_ids=[...], order_by=["metrics.<metric> DESC"], max_results=1)`.

## The MLflow UI

<!-- objectives: S1.13 -->

Open an experiment from the notebook's **Experiment** side panel or from **Experiments** in the
sidebar ([MLflow experiments](https://docs.databricks.com/aws/en/mlflow/experiments),
[MLflow runs](https://docs.databricks.com/aws/en/mlflow/runs)).

**Experiment page (runs table)**

- One row per run: run name, creation time, the user, the **source** (the notebook revision, or job,
  that created the run), logged models, and columns for any parameters, metrics and tags you add.
- Search and filter runs by metric, parameter and tag values (the same filter syntax as
  `search_runs`); sort by any column; group nested runs under their parent.
- A **chart view** plots metrics across runs.
- Select runs and click **Compare** to open the comparison page.

**Run page**

- Run ID, status, start time and duration, the user and the **source** with a link to the exact
  notebook revision that produced the run.
- **Parameters**, **metrics** (with charts for metrics logged over steps) and **tags** (editable).
- **Artifacts** tab: every logged file — plots, JSON, data samples — and logged models with their
  `MLmodel` file, environment files (`conda.yaml`, `requirements.txt`), signature and ready-to-use
  code snippets for loading and predicting.

**Comparing runs**

- Tables of parameters and metrics for the selected runs side by side, with an option to hide rows
  that are identical across runs, plus charts of metrics against parameters.

**Logged and registered models**

- A logged model links back to its run and shows its metrics; a registered model in Unity Catalog
  shows versions, aliases, tags and lineage (chapter 5).

## Choosing an approach

| Situation | Prefer | Why |
|---|---|---|
| Quick experiments with a supported library | Autologging | Zero effort, consistent names |
| Custom metrics, plots, data versions, business KPIs | Manual logging (alone or on top of autolog) | Full control |
| Tuning | Nested runs (parent + children) | Keeps the experiment readable |
| Pick the best run in a notebook or job | `MlflowClient.search_runs` | Programmatic, reproducible |
| Explore and compare by eye | MLflow UI | Charts, compare view |
| Ad-hoc analysis of many runs | `mlflow.search_runs` → pandas | Familiar DataFrame operations |

## Pitfalls

**1. Logging a model without a signature.** Registration in Unity Catalog fails and serving cannot
validate input. Infer the signature from training inputs and predictions:

> **⚠️ Wrong approach:** `mlflow.sklearn.log_model(model, name="model")` — `model_info.signature` is
> `None`.

**2. Sorting the wrong way.** `order_by=["metrics.f1"]` is ascending: `max_results=1` returns the
*worst* run. Add `DESC` for scores.

**3. Comparing parameters as numbers.** Parameters are stored as strings:
`params.max_depth > 5` compares text. Filter on metrics, or quote parameter values exactly.

**4. Runs in the wrong experiment.** Without `mlflow.set_experiment`, runs go to the notebook
experiment — jobs running the same code from another notebook scatter runs across experiments.

**5. Forgetting to end runs.** Use `with mlflow.start_run():` so the run is closed (and marked failed
on exceptions); otherwise later logging lands in an old run.

**6. `artifact_path` in new code.** It still works but is deprecated in MLflow 3; use `name`.

## Summary

- Experiments group runs; runs hold parameters, metrics, tags, artifacts and the source; MLflow 3
  treats models as first-class logged models.
- Log manually with `log_param(s)`, `log_metric(s)` (with `step`), `log_artifact`/`log_figure`/
  `log_dict` and `<flavor>.log_model(name=..., signature=...)`; or use autologging.
- Find the best run with `MlflowClient.search_runs(..., order_by=["metrics.x DESC"], max_results=1)`.
- The UI shows runs tables, run details (params, metrics, tags, source notebook revision, artifacts,
  models) and side-by-side comparisons.

## Review questions

<!-- questions: ch04 -->

### Question 1

A data scientist ran 40 training runs in an experiment, each logging the metric val_auc. Which code returns the single run with the highest val_auc?

- **A.** MlflowClient().search_runs([experiment_id], order_by=["metrics.val_auc DESC"], max_results=1)
- **B.** MlflowClient().search_runs([experiment_id], order_by=["metrics.val_auc"], max_results=1)
- **C.** MlflowClient().get_run(experiment_id)
- **D.** mlflow.log_metric("val_auc", max_results=1)

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Sorting the metric descending and keeping one result returns the best-scoring run.

Why the other options are wrong:

- **B**: The default order is ascending, which returns the lowest AUC.
- **C**: get_run takes a run ID and returns that run, not the best one.
- **D**: log_metric records a value; it does not search.

Objective S1.11 · [Revisit](04-mlflow-tracking.md#finding-the-best-run-with-the-client-api)

</details>

### Question 2

An analyst wants all runs of an experiment with test_rmse below 12 as a pandas DataFrame to plot them. Which call is the most direct?

- **A.** mlflow.search_runs(experiment_names=["/Shared/prices"], filter_string="metrics.test_rmse < 12")
- **B.** mlflow.log_metrics({"test_rmse": 12})
- **C.** MlflowClient().list_artifacts(run_id)
- **D.** mlflow.set_experiment("/Shared/prices")

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — mlflow.search_runs returns matching runs as a pandas DataFrame, and filter_string restricts them.

Why the other options are wrong:

- **B**: This logs a metric instead of querying runs.
- **C**: list_artifacts lists the files of one run.
- **D**: set_experiment selects the active experiment but returns no runs.

Objective S1.11 · [Revisit](04-mlflow-tracking.md#finding-the-best-run-with-the-client-api)

</details>

### Question 3

Inside a with mlflow.start_run(): block, a data scientist wants to record the regularisation strength used, the validation F1 score and a PNG of the ROC curve. Which TWO calls are needed in addition to logging the F1 with mlflow.log_metric?

- **A.** mlflow.log_param("C", 0.5)
- **B.** mlflow.log_figure(roc_figure, "roc_curve.png")
- **C.** mlflow.set_experiment("roc_curve.png")
- **D.** mlflow.register_model("roc_curve.png", "prod.ml.roc")
- **E.** mlflow.log_metric("C", "0.5")

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — Hyperparameters are parameters (log_param), and a matplotlib figure is saved as an artifact with log_figure (or saved to a file and logged with log_artifact).

Why the other options are wrong:

- **C**: set_experiment selects an experiment; it does not log a file.
- **D**: register_model registers models, not images.
- **E**: Metrics must be numeric results; a hyperparameter belongs in params.

Objective S1.12 · [Revisit](04-mlflow-tracking.md#logging-manually)

</details>

### Question 4

A team trains a neural network for 20 epochs and wants the MLflow UI to show the training loss as a line over the epochs. How should they log it?

- **A.** Call mlflow.log_metric("train_loss", loss, step=epoch) once per epoch
- **B.** Call mlflow.log_param("train_loss", losses) once with the list of losses
- **C.** Call mlflow.set_tag("train_loss", loss) once per epoch
- **D.** Log only the final loss with mlflow.log_metric

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Logging a metric repeatedly with a step builds a metric history that the UI charts.

Why the other options are wrong:

- **B**: Parameters are single strings, not time series.
- **C**: Tags are labels; repeated calls overwrite the value.
- **D**: The final value alone cannot be plotted over epochs.

Objective S1.12 · [Revisit](04-mlflow-tracking.md#logging-manually)

</details>

### Question 5

A model logged with mlflow.sklearn.log_model(model, name="model") cannot be registered in Unity Catalog, and the error mentions a missing signature. What should the data scientist change?

- **A.** Pass signature=infer_signature(X_train, model.predict(X_train)) (and an input example) to log_model
- **B.** Replace name="model" with artifact_path="model"
- **C.** Register the model in the workspace registry instead
- **D.** Call mlflow.end_run() before logging the model

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Unity Catalog requires models to have a signature describing inputs and outputs; infer it from sample inputs and predictions.

Why the other options are wrong:

- **B**: artifact_path is the deprecated name of the same argument and adds no signature.
- **C**: The workspace registry is legacy and avoids rather than fixes the problem.
- **D**: Ending the run does not add a signature.

Objective S1.12 · [Revisit](04-mlflow-tracking.md#pitfalls)

</details>

### Question 6

A data scientist wants scikit-learn training runs to record parameters, training metrics and the fitted model without writing logging calls. What should they do?

- **A.** Call mlflow.sklearn.autolog() (or mlflow.autolog()) before training
- **B.** Call mlflow.log_artifacts() after training
- **C.** Use mlflow.search_runs() during training
- **D.** Set the tag mlflow.autolog=true on the experiment

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Autologging patches supported libraries so fit() logs parameters, metrics and the model automatically.

Why the other options are wrong:

- **B**: log_artifacts logs files you already created; it does not capture parameters or metrics.
- **C**: search_runs queries runs; it logs nothing.
- **D**: No such tag enables autologging.

Objective S1.12 · [Revisit](04-mlflow-tracking.md#autologging)

</details>

### Question 7

Which TWO kinds of information can a data scientist find on an MLflow run page in Databricks?

- **A.** A link to the notebook revision that produced the run
- **B.** The logged artifacts, including the model with code snippets to load it
- **C.** The real-time request latency of the model serving endpoint
- **D.** The Spark executor logs of every job in the workspace
- **E.** The list of users who own the Unity Catalog metastore

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — The run page shows parameters, metrics, tags, the source notebook revision and an Artifacts tab with files and logged models.

Why the other options are wrong:

- **C**: Endpoint latency is shown on the serving endpoint page, not the run page.
- **D**: Cluster logs are on the compute page.
- **E**: Metastore ownership is managed in Catalog Explorer and the account console.

Objective S1.13 · [Revisit](04-mlflow-tracking.md#the-mlflow-ui)

</details>

### Question 8

Two runs with different max_depth values have similar metrics, and the data scientist wants to see their parameters and metrics side by side in the UI. What should they use?

- **A.** Select both runs on the experiment page and click Compare
- **B.** Open the Catalog Explorer and select the experiment
- **C.** Open the cluster's Spark UI
- **D.** Export the notebook as HTML

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — The Compare view shows parameter and metric tables (and charts) for the selected runs side by side.

Why the other options are wrong:

- **B**: Catalog Explorer shows Unity Catalog assets, not run comparisons.
- **C**: The Spark UI shows job execution details.
- **D**: An HTML export contains the notebook, not a comparison of runs.

Objective S1.13 · [Revisit](04-mlflow-tracking.md#the-mlflow-ui)

</details>

### Question 9

A notebook runs a hyperparameter search that creates 30 runs, and the experiment page becomes hard to read. How can the runs be grouped under one tuning session?

- **A.** Start a parent run and create each trial with mlflow.start_run(nested=True) inside it
- **B.** Log all 30 models into one run with the same name
- **C.** Create 30 separate experiments
- **D.** Delete all runs except the best one

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Nested runs are linked to the parent run and shown grouped under it in the UI.

Why the other options are wrong:

- **B**: Overwriting artifacts loses the comparison between trials.
- **C**: Separate experiments make comparison harder.
- **D**: Deleting runs destroys the record of the search.

Objective S1.12 · [Revisit](04-mlflow-tracking.md#nested-runs)

</details>
<!-- /questions -->
