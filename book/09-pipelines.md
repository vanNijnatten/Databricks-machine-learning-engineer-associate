---
chapter: 9
title: "Pipelines: estimators and transformers"
objectives: [S3.03, S3.04]
prerequisites: [8]
reading_minutes: 70
tutorial:
  notebook: notebooks/ch09_pipelines.py
  minutes: 30
  requires: [serverless]
  free_edition: full
  cleanup: false
  extra_files: []
  steps:
    - id: spark-pipeline
      objectives: [S3.03, S3.04]
    - id: sklearn-pipeline
      objectives: [S3.04]
references: [spark-mllib]
---

# 9. Pipelines: estimators and transformers

> **Exam mapping**: Section 3 — Model Development · objectives S3.03, S3.04 ·
> **Runtime**: Databricks Runtime 18 LTS ML · **Read after**: chapter 8.

A model is rarely a single algorithm. Before a logistic regression can learn, missing values must be
filled, categories turned into numbers and features gathered into one vector. A **pipeline** chains
these steps so they are fitted together, applied together and saved together. The exam asks you to
tell estimators from transformers and to build such a pipeline.

> **Tutorial:** [`notebooks/ch09_pipelines.py`](../notebooks/ch09_pipelines.py) · about 30 minutes · **Requires:** serverless or classic compute · **Free Edition:** yes · **Clean-up:** none. Run [`notebooks/00_setup.py`](../notebooks/00_setup.py) once first.

## Learning objectives

- Define estimator, transformer and model in Spark ML, and the equivalent terms in scikit-learn.
- Predict what `fit()` and `transform()` return for each kind of stage.
- Build a Spark ML training pipeline and a scikit-learn pipeline for the same problem.
- Explain why pipelines prevent data leakage and make deployment safer.

## Estimators, transformers and models

<!-- objectives: S3.03 -->

Spark ML (the DataFrame-based `pyspark.ml` package, also called MLlib) defines two kinds of
pipeline stages:

| Concept | Method | Input → output | Examples |
|---|---|---|---|
| **Transformer** | `transform(df)` | DataFrame → DataFrame (adds columns) | `VectorAssembler`, `SQLTransformer`, every fitted model |
| **Estimator** | `fit(df)` | DataFrame → **Model** (a Transformer) | `StringIndexer`, `OneHotEncoder`, `Imputer`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`, `Pipeline` |
| **Model** | `transform(df)` | DataFrame → DataFrame | `StringIndexerModel`, `LogisticRegressionModel`, `PipelineModel` |

An estimator must *learn something from data* before it can transform anything: `StringIndexer`
learns which categories exist, `Imputer` learns the median, `StandardScaler` learns mean and standard
deviation, `LogisticRegression` learns coefficients. A transformer needs nothing from data:
`VectorAssembler` just concatenates columns into a vector.

- An estimator has **no** `transform()` method. Call `fit()` first; it returns a separate model object.
- A fitted model is a transformer: `model.transform(df)` adds the prediction columns
  (`rawPrediction`, `probability`, `prediction`).
- **Evaluators** (`BinaryClassificationEvaluator`, `RegressionEvaluator`, …) are a third kind of
  object: `evaluate(predictions)` returns a single number (chapter 11).

scikit-learn uses the same idea with slightly different vocabulary:

| | Spark ML | scikit-learn |
|---|---|---|
| Learns from data | Estimator: `fit()` returns a **new** Model object | `fit()` updates the **same** object in place and returns it |
| Changes features | Transformer: `transform()` | Transformer: `fit()` + `transform()` (or `fit_transform()`) |
| Predicts | Model: `transform()` adds a `prediction` column | Estimator/predictor: `predict()`, `predict_proba()` |
| Chain of steps | `Pipeline` → `PipelineModel` | `Pipeline`, `ColumnTransformer` |

> **Exam tip:** "Which of these is an estimator?" — anything that must be *fitted*: indexers,
> encoders, imputers, scalers, every learning algorithm and `Pipeline` itself. `VectorAssembler`
> is the classic pure transformer. A *fitted* model is a transformer.

## Building a Spark ML training pipeline

<!-- objectives: S3.04, S3.03 -->

The churn pipeline imputes missing numbers, indexes and one-hot encodes the categorical columns,
assembles a feature vector and trains a logistic regression:

<!-- snippet: src/db_ml_associate/ch09_pipelines/spark_pipeline.py#spark_pipeline -->
```python
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)

imputer = Imputer(strategy="median", inputCols=list(CHURN_NUMERIC), outputCols=IMPUTED)  # estimator
indexer = StringIndexer(inputCols=list(CHURN_CATEGORICAL), outputCols=INDEXED, handleInvalid="keep")  # estimator
encoder = OneHotEncoder(inputCols=INDEXED, outputCols=ENCODED)  # estimator
assembler = VectorAssembler(inputCols=IMPUTED + ENCODED, outputCol="features")  # transformer
classifier = LogisticRegression(featuresCol="features", labelCol=label)  # estimator
return Pipeline(stages=[imputer, indexer, encoder, assembler, classifier])
```

`Pipeline` is an estimator. Fitting it runs every stage in order: each estimator is fitted on the
output of the stages before it and replaced by its model. The result is a `PipelineModel` that
contains **only transformers**:

<!-- snippet: src/db_ml_associate/ch09_pipelines/spark_pipeline.py#fit_and_evaluate -->
```python
pipeline_model = build_spark_churn_pipeline(label).fit(train)  # Pipeline (estimator) → PipelineModel
predictions = pipeline_model.transform(test)  # PipelineModel is a transformer
evaluator = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
```

On the churn data the fitted stages are `ImputerModel`, `StringIndexerModel`,
`OneHotEncoderModel`, `VectorAssembler` (unchanged — it was already a transformer) and
`LogisticRegressionModel`, with a test ROC AUC of 0.76.

Things worth knowing about the Spark stages used here:

- **`StringIndexer`** maps categories to indices ordered by frequency (most frequent → 0).
  `handleInvalid="keep"` gives unseen categories at prediction time their own index instead of
  failing.
- **`OneHotEncoder`** turns indices into sparse vectors. By default `dropLast=True` drops one
  slot: *k* categories give vectors of length *k − 1* — or *k* when the indexer uses
  `handleInvalid="keep"`, which reserves an extra index for unseen categories (chapter 7).
- **`VectorAssembler`** is required: Spark ML algorithms read a single vector column
  (`featuresCol`, default `features`), not separate columns.
- **`Imputer`** only works on numeric columns and supports `mean`, `median` and `mode` (chapter 7).

`pyspark.ml` is supported on serverless, standard and dedicated compute
([Use Apache Spark MLlib on Databricks](https://docs.databricks.com/aws/en/machine-learning/train-model/mllib)).

## Building a scikit-learn pipeline

<!-- objectives: S3.04 -->

When the data fits in memory, the same pipeline in scikit-learn trains in seconds on one node. A
`ColumnTransformer` applies different preprocessing to different columns:

<!-- snippet: src/db_ml_associate/ch09_pipelines/sklearn_pipeline.py#sklearn_pipeline -->
```python
numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
categorical = OneHotEncoder(handle_unknown="ignore")  # unseen categories become all-zero vectors
prepare = ColumnTransformer(
    [
        ("numeric", numeric, list(CHURN_NUMERIC)),
        ("categorical", categorical, list(CHURN_CATEGORICAL)),
    ]
)
return Pipeline(
    [
        ("prepare", prepare),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ]
)
```

`pipeline.fit(X_train, y_train)` fits every step on the training data only;
`pipeline.predict(X_test)` applies the fitted steps to new rows. Logged to MLflow as one model
(chapter 4), the preprocessing travels with the algorithm, so the serving endpoint (chapter 13)
receives raw columns and does the same preparation as during training.

## Why pipelines matter

<!-- objectives: S3.04 -->

- **No leakage**: every learned statistic (medians, category lists, scaling) comes from the training
  data, because the whole pipeline is fitted on training data only.
- **No training–serving skew**: the exact same transformations run at training and at inference.
- **Tuning the whole chain**: cross-validation and hyperparameter search (chapter 10) can wrap a
  pipeline, so preprocessing is refitted inside every fold — and you can tune preprocessing
  parameters too.
- **One artefact**: a single `PipelineModel` or scikit-learn `Pipeline` is saved, logged and
  registered.

## Choosing an approach

| Situation | Prefer | Why |
|---|---|---|
| Data fits in memory on one machine | scikit-learn `Pipeline` (+ `ColumnTransformer`) | Fastest training, richest algorithm choice |
| Data does not fit on one machine, or already lives in Spark | Spark ML `Pipeline` | Distributed fitting and scoring |
| Need to score a huge table with a scikit-learn model | Train with scikit-learn, score with `mlflow.pyfunc.spark_udf` (chapter 12) | Single-node training, distributed inference |
| A transformation needs no learning (e.g. a ratio of two columns) | `SQLTransformer` / plain DataFrame code | No fitting needed |

## Pitfalls

**1. Fitting preprocessing before the split.** A scaler fitted on all rows learns the test rows'
statistics.

> **⚠️ Wrong approach:** fit, then split.

<!-- snippet: src/db_ml_associate/ch09_pipelines/pitfalls.py#wrong_fit_scaler_on_full_data -->
```python
scaler = StandardScaler().fit(churn[[column]])  # learns from rows that will become test data
train, test = split_frame(churn, CHURN_LABEL, seed=seed)
train_scaled, test_scaled = (
    scaler.transform(train[[column]]),
    scaler.transform(test[[column]]),
)
```

The leaked scaler's mean is 36.41 (all rows); the training rows' mean is 36.52. Small here, but with
target-dependent steps such as target encoding or feature selection the leak can make a useless
model look excellent. Split first and let a pipeline fit the scaler:

<!-- snippet: src/db_ml_associate/ch09_pipelines/pitfalls.py#correct_fit_inside_pipeline -->
```python
train, test = split_frame(churn, CHURN_LABEL, seed=seed)
pipeline = Pipeline([("scale", StandardScaler())]).fit(train[[column]])  # training statistics only
test_scaled = pipeline.transform(test[[column]])  # reuses the training mean and std
```

**2. Calling `transform()` on an estimator.** A Spark `StringIndexer` has no `transform()`; an
unfitted scikit-learn `StandardScaler().transform(...)` raises `NotFittedError`. Fit first — or let
the pipeline do it.

**3. Forgetting `VectorAssembler`.** Spark ML algorithms fail with "column features does not exist"
when columns are not assembled into one vector.

**4. Unseen categories at prediction time.** Without `handleInvalid="keep"` (Spark) or
`handle_unknown="ignore"` (scikit-learn), a new category in production makes scoring fail.

**5. Preprocessing outside the logged model.** If imputation happens in a notebook cell before
`fit()`, the logged model expects already-imputed data and serving receives raw data: skew.

## Summary

- Transformers transform; estimators fit and return models; fitted models are transformers.
- A Spark `Pipeline` is an estimator whose `fit()` returns a `PipelineModel` of transformers;
  scikit-learn's `Pipeline` fits in place.
- Pipelines prevent leakage and training–serving skew and are the unit you tune, log and deploy.

## Review questions

<!-- questions: ch09 -->

### Question 1

A data scientist builds a Spark ML pipeline with StringIndexer, OneHotEncoder, VectorAssembler and RandomForestClassifier. Which stage is a transformer that does not need to be fitted?

- **A.** StringIndexer
- **B.** OneHotEncoder
- **C.** VectorAssembler
- **D.** RandomForestClassifier

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: C** — VectorAssembler only concatenates existing columns into a vector; it learns nothing from the data, so it is a pure transformer.

Why the other options are wrong:

- **A**: StringIndexer must learn the set of categories and their order, so it is an estimator.
- **B**: In Spark 3+ OneHotEncoder learns the number of categories, so it is an estimator.
- **D**: Every learning algorithm is an estimator; its fitted model is the transformer.

Objective S3.03 · [Revisit](09-pipelines.md#estimators-transformers-and-models)

</details>

### Question 2

In Spark ML, what does calling fit() on a LogisticRegression estimator with a training DataFrame return?

- **A.** A DataFrame with a prediction column
- **B.** A LogisticRegressionModel, which is a transformer
- **C.** A single float with the training accuracy
- **D.** Nothing; the estimator object is updated in place

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — Spark ML estimators return a new Model object from fit(); the model is a transformer whose transform() adds prediction columns.

Why the other options are wrong:

- **A**: Predictions are produced by the fitted model's transform(), not by fit().
- **C**: Metrics come from an evaluator applied to predictions.
- **D**: Updating in place is scikit-learn's convention, not Spark ML's.

Objective S3.03 · [Revisit](09-pipelines.md#estimators-transformers-and-models)

</details>

### Question 3

Which TWO statements about Spark ML pipelines are correct?

- **A.** A Pipeline is an estimator
- **B.** A PipelineModel contains only transformers
- **C.** A Pipeline can only contain transformers
- **D.** fit() on a Pipeline fits every stage on the original input DataFrame, ignoring earlier stages
- **E.** A PipelineModel must be refitted before every call to transform()

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — Pipeline.fit() runs the stages in order, replacing each estimator with its fitted model, and returns a PipelineModel made only of transformers.

Why the other options are wrong:

- **C**: Pipelines routinely contain estimators such as indexers and algorithms.
- **D**: Each stage is fitted on the output of the previous stages.
- **E**: A fitted PipelineModel can transform any number of DataFrames.

Objective S3.03 · [Revisit](09-pipelines.md#building-a-spark-ml-training-pipeline)

</details>

### Question 4

A Spark ML training job fails with an error saying that the column "features" does not exist. The DataFrame has numeric columns age, income and tenure, and the code fits LogisticRegression(labelCol="label") directly on it. What is missing?

- **A.** A StringIndexer on the label column
- **B.** A VectorAssembler that combines the numeric columns into a features vector column
- **C.** A call to cache() on the DataFrame
- **D.** A StandardScaler with withMean=True

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — Spark ML algorithms read a single vector column (featuresCol, default "features"); a VectorAssembler must create it from the individual columns.

Why the other options are wrong:

- **A**: A numeric label is already valid; indexing it would not create the features column.
- **C**: Caching affects performance, not the schema.
- **D**: A scaler also expects a vector column as input and does not create "features" from separate columns.

Objective S3.04 · [Revisit](09-pipelines.md#building-a-spark-ml-training-pipeline)

</details>

### Question 5

A model in production starts failing whenever a customer has a payment method that did not exist in the training data. The model is a Spark PipelineModel that uses StringIndexer. Which change prevents the failure?

- **A.** Set handleInvalid="keep" on the StringIndexer and retrain
- **B.** Replace the StringIndexer with a VectorAssembler
- **C.** Increase the number of partitions of the scoring DataFrame
- **D.** Drop the payment method column at scoring time

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — With handleInvalid="keep", unseen categories get their own index instead of raising an error at transform time.

Why the other options are wrong:

- **B**: VectorAssembler cannot encode string categories.
- **C**: Partitioning does not change how unseen labels are handled.
- **D**: The fitted pipeline expects the column; removing it makes scoring fail differently.

Objective S3.04 · [Revisit](09-pipelines.md#pitfalls)

</details>

### Question 6

A data scientist imputes missing values with the median of the whole dataset in a notebook cell, then splits the data and trains a scikit-learn model on the imputed training rows. Which statement best describes the problems with this approach?

- **A.** The test rows influenced the imputation statistic, and the logged model will not impute raw data at serving time
- **B.** Medians cannot be computed on numeric data, and scikit-learn cannot train on imputed data
- **C.** The model will be too simple, and the training set will be too small
- **D.** There is no problem as long as the split is stratified

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Imputing before the split leaks test information into training, and because imputation is not part of the logged model, serving receives raw data the model never saw: training-serving skew. Putting SimpleImputer inside a Pipeline fixes both.

Why the other options are wrong:

- **B**: Both statements are false.
- **C**: Imputation does not change model complexity or the number of rows.
- **D**: Stratification controls the class ratio, not leakage of statistics.

Objective S3.04 · [Revisit](09-pipelines.md#why-pipelines-matter)

</details>

### Question 7

Which TWO are benefits of putting preprocessing and the model into one pipeline before logging it to MLflow?

- **A.** The same transformations are applied at training and at inference time
- **B.** Cross-validation refits the preprocessing inside every fold, avoiding leakage
- **C.** The model no longer needs a signature
- **D.** Training becomes distributed automatically
- **E.** Hyperparameters no longer need tuning

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — Pipelines remove training-serving skew and let tuning and cross-validation fit preprocessing on the training folds only.

Why the other options are wrong:

- **C**: Logging a signature is still recommended and required for Unity Catalog registration.
- **D**: A scikit-learn pipeline still trains on a single node.
- **E**: Pipelines make tuning easier, not unnecessary.

Objective S3.04 · [Revisit](09-pipelines.md#why-pipelines-matter)

</details>

### Question 8

In scikit-learn, which method call is a typical way to apply a fitted StandardScaler to new validation data?

- **A.** scaler.fit_transform(validation)
- **B.** scaler.transform(validation)
- **C.** scaler.predict(validation)
- **D.** scaler.evaluate(validation)

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — A fitted transformer is applied with transform(); it reuses the statistics learned from the training data.

Why the other options are wrong:

- **A**: fit_transform would relearn the statistics from validation data, which leaks information.
- **C**: predict() belongs to predictors such as classifiers, not to transformers.
- **D**: Transformers have no evaluate() method.

Objective S3.03 · [Revisit](09-pipelines.md#estimators-transformers-and-models)

</details>

### Question 9

A team trains a gradient-boosted model on 2 GB of tabular data and must score a 4 TB Delta table nightly. Which approach is most appropriate?

- **A.** Train with Spark ML because the scoring data is large
- **B.** Train a single-node scikit-learn or XGBoost pipeline and score it at scale with an MLflow Spark UDF
- **C.** Collect the 4 TB table to the driver with toPandas() and call predict()
- **D.** Retrain the model on every nightly batch before scoring

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — Training data fits on one machine, so single-node training is simplest and fastest; mlflow.pyfunc.spark_udf distributes scoring across the cluster.

Why the other options are wrong:

- **A**: Training data size, not scoring data size, decides the training library.
- **C**: 4 TB will not fit in driver memory.
- **D**: Retraining nightly is unrelated to the scoring problem and wastes compute.

Objective S3.04 · [Revisit](09-pipelines.md#choosing-an-approach)

</details>
<!-- /questions -->
