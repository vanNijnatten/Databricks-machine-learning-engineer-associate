---
chapter: 2
title: AutoML
objectives: [S1.03, S1.04]
prerequisites: [4, 8, 11]
reading_minutes: 50
references: [automl, mlflow-experiments, how-automl-works, automl-api-reference, automl-classification-data-prep, dbr-18-0-ml, free-edition-limitations]
---

# 2. AutoML

> **Exam mapping**: Section 1 — Databricks Machine Learning · objectives S1.03, S1.04 ·
> **Runtime**: Databricks Runtime 18 LTS ML (AutoML installed separately) · **Read after**:
> chapters 4, 8 and 11.

**Databricks AutoML** takes a table and a target column and, within a time budget, prepares the
data, trains and tunes many models with several libraries, logs every trial to MLflow and hands you
the **source code** of each trial. The exam asks how AutoML helps with model and feature selection
and what it brings to model development.

## Learning objectives

- Describe what AutoML does from data preparation to the generated notebooks.
- Explain how AutoML facilitates model selection and feature selection.
- List the advantages AutoML brings to the model development process, and its limits.
- Start an AutoML experiment from the UI or the Python API and read its results.

> **Platform change:** AutoML is **no longer a built-in library from Databricks Runtime 18.0 ML**;
> it is provided by the `databricks-automl-runtime` package
> ([AutoML](https://docs.databricks.com/aws/en/machine-learning/automl/),
> [18.0 ML release notes](https://docs.databricks.com/aws/en/release-notes/runtime/18.0ml)). The
> concepts the exam tests are unchanged.

> **Requires:** dedicated (classic) compute with the ML runtime — AutoML does not run on standard
> access mode, so it is not available on serverless-only Free Edition. Cost: a classic cluster runs
> for the whole `timeout_minutes` (default 120); set a short timeout for learning and terminate the
> cluster afterwards.

## What AutoML does

<!-- objectives: S1.03, S1.04 -->

AutoML supports **classification**, **regression** and **forecasting**. For classification and
regression it ([How Databricks AutoML works](https://docs.databricks.com/aws/en/machine-learning/automl/how-automl-works),
[Data preparation for classification](https://docs.databricks.com/aws/en/machine-learning/automl/classification-data-prep)):

1. **Prepares the data**
   - **Semantic type detection**: date strings become timestamps, numeric strings become numbers,
     numeric IDs become categoricals, free text becomes text.
   - **Imputation**: chooses a method per column type (override per column with `imputers`).
   - **Class imbalance**: balances only the *training* set by downsampling the majority class and
     adding class weights — validation and test data keep the real distribution.
   - **Sampling**: samples very large data so each trial fits on one worker (stratified for
     classification).
2. **Splits** the data 60 % train / 20 % validation / 20 % test (stratified for classification), or
   chronologically with a time column, or with your own split column.
3. **Searches models**: runs trials in parallel with scikit-learn (decision trees, random forests,
   logistic/linear regression), **XGBoost** and **LightGBM**, tuning hyperparameters; forecasting
   uses Prophet and Auto-ARIMA.
4. **Tracks everything in MLflow**: every trial is a run with parameters, metrics and the model.
5. **Generates notebooks**:
   - a **data exploration notebook** with an overview of the data;
   - a **best trial notebook** — the full, editable training code of the best model;
   - notebooks for the other trials, stored as MLflow artifacts and importable on demand.
   The notebooks include **SHAP** feature-importance code, disabled by default because it is
   memory-intensive (`shap_enabled = True` turns it on).

The best model is the trial with the best **primary metric** on the **validation** set; its test
score is reported for an unbiased estimate.

## How AutoML facilitates model and feature selection

<!-- objectives: S1.03 -->

**Model selection**

- It trains and tunes *many* candidate algorithms and hyperparameter settings automatically and ranks
  them by the primary metric you choose (`f1` by default for classification; also `log_loss`,
  `precision`, `accuracy`, `roc_auc`; `r2` by default for regression; also `mae`, `rmse`, `mse`).
- Results land in one MLflow experiment, so you compare trials side by side (chapter 4) and register
  the winner (chapter 5).
- You can restrict the search with `exclude_frameworks`.

**Feature selection**

- **Column selection**: untick columns in the UI or pass `exclude_cols` (e.g. identifiers or columns
  that leak the label). The target and time columns cannot be excluded.
- **Semantic type detection** and **imputation** turn raw columns into usable features.
- **Feature importance**: SHAP code in the trial notebooks shows which features drive predictions,
  guiding which features to keep, drop or engineer.
- **Feature Store integration**: `feature_store_lookups` adds features from feature tables
  (chapter 3).

> **Exam tip:** AutoML does **not** promise a production-ready final model or remove the need for
> domain knowledge. It *accelerates* model and feature selection by producing a ranked set of tracked
> trials, a data overview, feature importances and editable code.

## Advantages for the model development process

<!-- objectives: S1.04 -->

| Advantage | Why it matters |
|---|---|
| **Fast baseline** | A strong benchmark in minutes; tells you quickly whether the data has signal |
| **Glass-box code** | Generated notebooks are ordinary, editable Python — no black box; reuse and customise them |
| **Built-in best practices** | Stratified splits, imbalance handling, imputation, a held-out test set |
| **Full MLflow tracking** | Every trial is logged with parameters, metrics and the model; reproducible and comparable |
| **Low code** | Domain experts can start from the UI; experts save boilerplate |
| **Explainability** | SHAP code included |
| **Parallel tuning** | Trials run in parallel on the cluster |

**Limits**: sampling on very large data; tabular data only (no images); not a substitute for feature
engineering and domain knowledge; and it consumes cluster time for the whole budget.

## Running AutoML

<!-- objectives: S1.03, S1.04 -->

**UI:** open **Experiments** in the sidebar and pick the classification, regression or forecasting
AutoML option ([MLflow experiments](https://docs.databricks.com/aws/en/mlflow/experiments)): choose the
compute, problem type, table, prediction target, evaluation metric and timeout, and optionally
untick columns or set imputation per column. Click **Start AutoML**.

**Python API** ([AutoML Python API reference](https://docs.databricks.com/aws/en/machine-learning/automl/automl-api-reference)):

<!-- snippet: src/db_ml_associate/ch02_automl/workspace.py#classify -->
```python
summary = automl.classify(
    dataset=dataset,  # Spark/pandas DataFrame or a Unity Catalog table name
    target_col=target,
    primary_metric=primary_metric,  # f1 (default), log_loss, precision, accuracy, roc_auc
    timeout_minutes=timeout_minutes,  # default 120, minimum 5
    exclude_cols=list(exclude_cols),  # e.g. identifiers that must not be features
)
```

The call blocks until the timeout or completion and returns an `AutoMLSummary`:

<!-- snippet: src/db_ml_associate/ch02_automl/workspace.py#summary -->
```python
best = summary.best_trial  # TrialInfo with the best primary-metric score on validation data
result = AutoMLResult(
    experiment_id=summary.experiment.experiment_id,  # every trial is an MLflow run here
    best_run_id=best.mlflow_run_id,
    best_metrics=dict(best.metrics),
    best_model_uri=f"runs:/{best.mlflow_run_id}/model",  # or best.load_model()
    best_notebook_url=best.notebook_url,  # generated, editable source code
    trial_count=len(summary.trials),
)
```

Other trial notebooks can be imported from their MLflow artifact:

<!-- snippet: src/db_ml_associate/ch02_automl/workspace.py#import_notebook -->
```python
result = automl.import_notebook(trial.artifact_uri, path, overwrite=True)
```

## Choosing an approach

| Situation | Prefer |
|---|---|
| New tabular problem, need a baseline quickly | AutoML |
| Domain experts without much ML coding experience | AutoML UI, then hand the best notebook to an ML engineer |
| Custom architecture, custom loss, unusual data (images, text models) | Manual training |
| Full control over preprocessing and the search space | Manual training + tuning (chapter 10), possibly starting from the AutoML notebook |
| Production pipeline | Start from the generated best-trial notebook, refactor into reviewed, tested code (chapter 1) |

## Pitfalls

1. **Leaky columns left in** — an ID or a column derived from the label makes AutoML look brilliant.
   Use `exclude_cols`.
2. **Treating the best trial as production-ready** — review the generated code, validate on fresh
   data, and productionise it through your MLOps process.
3. **Forgetting the time column for temporal data** — a random split lets models train on the
   future; set `time_col`.
4. **Ignoring sampling** — on very large tables AutoML may train on a sample; check the notebook.
5. **Running on unsupported compute** — standard access mode and serverless are not supported, and
   from Runtime 18.0 ML the package must be installed.
6. **Expecting SHAP plots by default** — enable `shap_enabled` in the notebook and re-run it.

## Summary

- AutoML prepares data, splits 60/20/20, trains and tunes scikit-learn, XGBoost and LightGBM models
  (Prophet/ARIMA for forecasting), logs every trial to MLflow and generates editable notebooks.
- It facilitates model selection by ranking tracked trials on a chosen metric, and feature selection
  through column exclusion, semantic types, imputation, feature-store lookups and SHAP importances.
- Its advantages are speed to a baseline, glass-box code, built-in best practices, reproducibility
  and accessibility.

## Review questions

<!-- questions: ch02 -->

### Question 1

A business analyst with limited coding experience needs a reasonable classification baseline for a new dataset within an hour, and an ML engineer will later refine it. Which Databricks capability is the best starting point?

- **A.** AutoML, which trains and tunes many models and produces editable notebooks
- **B.** Model Serving with a traffic split
- **C.** A Lakeflow pipeline
- **D.** Unity Catalog lineage

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — AutoML quickly produces a ranked set of tracked models and the source code of the best trial, which an engineer can then refine.

Why the other options are wrong:

- **B**: Serving deploys an existing model; it does not train one.
- **C**: Pipelines process data; they do not search for models.
- **D**: Lineage tracks data and model dependencies.

Objective S1.04 · [Revisit](02-automl.md#advantages-for-the-model-development-process)

</details>

### Question 2

Which TWO are advantages AutoML brings to the model development process?

- **A.** It generates editable notebooks with the code of the trials ("glass-box" AutoML)
- **B.** It logs every trial to MLflow with parameters, metrics and the model
- **C.** It guarantees that the best trial needs no further validation before production
- **D.** It removes the need to understand the business problem
- **E.** It trains deep-learning models on image data automatically

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — AutoML's code is reviewable and reusable, and its trials are fully tracked, which makes results reproducible and comparable.

Why the other options are wrong:

- **C**: The best trial still needs review, validation and an MLOps process.
- **D**: Domain knowledge is still needed, for example to exclude leaky columns.
- **E**: AutoML works on tabular data; image data is not supported.

Objective S1.04 · [Revisit](02-automl.md#advantages-for-the-model-development-process)

</details>

### Question 3

An AutoML classification experiment on loan data reports a suspiciously perfect F1 score. The table contains a column default_flag_updated that is only filled after a loan defaults. How should the data scientist rerun AutoML?

- **A.** Exclude the column with exclude_cols (or untick it in the UI)
- **B.** Increase timeout_minutes
- **C.** Change primary_metric to accuracy
- **D.** Set exclude_frameworks to ["xgboost"]

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — The column leaks the label; removing it from the features is part of feature selection, done with exclude_cols or the UI column selection.

Why the other options are wrong:

- **B**: More time only finds more models that exploit the leak.
- **C**: The metric is not the problem; the leaked feature is.
- **D**: Every framework can exploit the leaked column.

Objective S1.03 · [Revisit](02-automl.md#how-automl-facilitates-model-and-feature-selection)

</details>

### Question 4

How does AutoML decide which trial is the best model of a classification experiment?

- **A.** The trial with the best value of the chosen primary metric on the validation set
- **B.** The trial that finished first
- **C.** The trial with the most features
- **D.** The trial with the best training accuracy

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — AutoML ranks trials by the primary metric (f1 by default) evaluated on the validation split; the test split gives an unbiased estimate afterwards.

Why the other options are wrong:

- **B**: Training order says nothing about quality.
- **C**: More features do not mean a better model.
- **D**: Training accuracy rewards overfitting.

Objective S1.03 · [Revisit](02-automl.md#what-automl-does)

</details>

### Question 5

A data scientist wants to know which features drove the predictions of the best AutoML model. Where do they find this?

- **A.** In the SHAP feature-importance code of the generated trial notebook, after enabling shap_enabled and re-running it
- **B.** In the cluster's Spark UI
- **C.** In the Unity Catalog audit log
- **D.** AutoML does not provide any way to inspect feature importance

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Regression and classification trial notebooks include SHAP code; it is disabled by default because it is memory-intensive.

Why the other options are wrong:

- **B**: The Spark UI shows job execution, not model explanations.
- **C**: Audit logs record access, not feature effects.
- **D**: SHAP code is included in the notebooks.

Objective S1.03 · [Revisit](02-automl.md#what-automl-does)

</details>

### Question 6

An AutoML classification experiment runs on a dataset in which 4% of rows are positive. What does AutoML do about the imbalance?

- **A.** It balances only the training split by downsampling the majority class and adding class weights
- **B.** It oversamples the minority class in the training, validation and test splits
- **C.** It removes the minority class rows that look like outliers
- **D.** Nothing; imbalance must be handled before calling AutoML

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — AutoML downsamples the majority class(es) and adds class weights to the training set only, so validation and test metrics reflect the real distribution.

Why the other options are wrong:

- **B**: Resampling validation or test data would distort the evaluation.
- **C**: AutoML does not drop minority rows.
- **D**: AutoML handles imbalance for classification on recent runtimes.

Objective S1.03 · [Revisit](02-automl.md#what-automl-does)

</details>

### Question 7

After an AutoML run, the team wants to reproduce and modify the training code of the third-best trial, whose notebook was not imported into the workspace. What should they do?

- **A.** Import the notebook from the trial's MLflow artifact with databricks.automl.import_notebook
- **B.** Rerun AutoML with a different seed until that trial becomes the best
- **C.** Reverse-engineer the model from its predictions
- **D.** Nothing; only the best trial's code is ever kept

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Notebooks of non-best trials are stored as MLflow artifacts and can be imported with import_notebook (or from the UI).

Why the other options are wrong:

- **B**: Rerunning is wasteful and not guaranteed to reproduce the trial.
- **C**: The source code already exists as an artifact.
- **D**: Every trial's notebook is saved as an artifact.

Objective S1.04 · [Revisit](02-automl.md#running-automl)

</details>

### Question 8

Which statement about AutoML in Databricks Runtime 18 LTS ML is correct?

- **A.** AutoML is no longer a built-in library and must be installed, and it runs on dedicated (classic) compute
- **B.** AutoML runs only on SQL warehouses
- **C.** AutoML replaced MLflow tracking in Runtime 18
- **D.** AutoML is available only for image classification

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — From Databricks Runtime 18.0 ML AutoML is not built in; it is provided by a separate package and requires dedicated compute.

Why the other options are wrong:

- **B**: SQL warehouses run SQL; AutoML runs on ML compute.
- **C**: AutoML logs its trials to MLflow.
- **D**: AutoML handles tabular classification, regression and forecasting.

Objective S1.04 · [Revisit](02-automl.md#2-automl)

</details>
<!-- /questions -->
