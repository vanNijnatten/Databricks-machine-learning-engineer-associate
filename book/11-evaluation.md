---
chapter: 11
title: Evaluation metrics
objectives: [S3.11, S3.12, S3.13, S3.14]
prerequisites: [8]
reading_minutes: 80
tutorial:
  notebook: notebooks/ch11_evaluation.py
  minutes: 30
  requires: [serverless]
  free_edition: full
  cleanup: false
  extra_files: []
  steps:
    - id: classification-metrics
      objectives: [S3.11, S3.13]
    - id: regression-metrics
      objectives: [S3.12]
    - id: back-transform
      objectives: [S3.14]
references: [spark-mllib]
---

# 11. Evaluation metrics

> **Exam mapping**: Section 3 — Model Development · objectives S3.11, S3.12, S3.13, S3.14 ·
> **Runtime**: Databricks Runtime 18 LTS ML · **Read after**: chapter 8 (read this chapter
> *before* chapter 10, which uses these metrics to compare models).

A metric turns "is this model any good?" into a number. The exam expects you to compute the common
classification and regression metrics, to pick the one that matches a business objective, and to
remember that a model trained on `log(price)` must be judged in euros, not in logs.

> **Tutorial:** [`notebooks/ch11_evaluation.py`](../notebooks/ch11_evaluation.py) · about 30 minutes · **Requires:** serverless or classic compute · **Free Edition:** yes · **Clean-up:** none. Run [`notebooks/00_setup.py`](../notebooks/00_setup.py) once first.

## Learning objectives

- Compute precision, recall, F1, accuracy, log loss and ROC AUC for a classifier.
- Compute RMSE, MAE and R² for a regression model.
- Choose the metric that matches a scenario's cost of errors.
- Exponentiate predictions of a log-transformed target before computing metrics or interpreting
  them.
- Compute the same metrics with Spark ML evaluators.

## Classification metrics

<!-- objectives: S3.11 -->

Every binary prediction lands in one cell of the **confusion matrix**:

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | True positive (TP) | False negative (FN) — a *missed* case |
| **Actually negative** | False positive (FP) — a *false alarm* | True negative (TN) |

| Metric | Formula | Question it answers | Range, best |
|---|---|---|---|
| **Accuracy** | (TP + TN) / all | How often is the model right? | 0–1, 1 |
| **Precision** | TP / (TP + FP) | When it says "positive", how often is it right? | 0–1, 1 |
| **Recall** (sensitivity, TPR) | TP / (TP + FN) | Of all real positives, how many did it find? | 0–1, 1 |
| **F1** | 2 · P · R / (P + R) | Balance of precision and recall (harmonic mean) | 0–1, 1 |
| **Log loss** (cross-entropy) | −mean(y·log p + (1−y)·log(1−p)) | Are the *probabilities* trustworthy? Punishes confident mistakes | 0–∞, 0 |
| **ROC AUC** | Area under TPR vs FPR over all thresholds | Does the model rank positives above negatives? 0.5 = random | 0–1, 1 |
| **PR AUC** (average precision) | Area under precision vs recall | Like ROC AUC, but focused on a rare positive class | 0–1, 1 |

Accuracy, precision, recall and F1 depend on the **decision threshold** (by default 0.5) because they
use hard predictions. Log loss, ROC AUC and PR AUC use the predicted **probabilities** and are
threshold-independent.

<!-- snippet: src/db_ml_associate/ch11_evaluation/metrics.py#classification_metrics -->
```python
return {
    "accuracy": accuracy_score(y_true, y_pred),  # share of correct predictions
    "precision": precision_score(y_true, y_pred),  # of predicted positives, how many are right
    "recall": recall_score(y_true, y_pred),  # of real positives, how many we found
    "f1": f1_score(y_true, y_pred),  # harmonic mean of precision and recall
    "log_loss": log_loss(y_true, y_proba),  # penalises confident wrong probabilities
    "roc_auc": roc_auc_score(y_true, y_proba),  # ranking quality over all thresholds
    "pr_auc": average_precision_score(y_true, y_proba),  # precision-recall area, for rare positives
}
```

For multiclass problems, precision, recall and F1 are computed per class and averaged:
**macro** (every class counts equally) or **weighted** (by class frequency).

## Regression metrics

<!-- objectives: S3.12 -->

| Metric | Formula | Unit | Behaviour |
|---|---|---|---|
| **RMSE** | √mean((y − ŷ)²) | Target unit | Squares errors, so large errors dominate |
| **MAE** | mean(\|y − ŷ\|) | Target unit | Every euro of error counts the same; robust to outliers |
| **R²** | 1 − Σ(y − ŷ)² / Σ(y − ȳ)² | Unitless | Share of variance explained; 1 perfect, 0 no better than predicting the mean, negative worse |

RMSE ≥ MAE always; a large gap between them means a few big errors.

<!-- snippet: src/db_ml_associate/ch11_evaluation/metrics.py#regression_metrics -->
```python
return {
    "rmse": root_mean_squared_error(y_true, y_pred),  # same unit as the target; punishes big errors
    "mae": mean_absolute_error(y_true, y_pred),  # same unit as the target; robust to outliers
    "r2": r2_score(y_true, y_pred),  # share of variance explained; 1 is perfect, 0 = predicting the mean
}
```

## The same metrics in Spark ML

<!-- objectives: S3.11, S3.12 -->

Spark ML computes metrics with **evaluators**, which take a predictions DataFrame (the output of a
fitted model's `transform()`) and return one number per call
([Use Apache Spark MLlib on Databricks](https://docs.databricks.com/aws/en/machine-learning/train-model/mllib)):

| Evaluator | `metricName` values |
|---|---|
| `BinaryClassificationEvaluator` | `areaUnderROC` (default), `areaUnderPR` |
| `MulticlassClassificationEvaluator` | `f1` (default, weighted), `accuracy`, `weightedPrecision`, `weightedRecall`, `fMeasureByLabel`, `precisionByLabel`, `recallByLabel`, `logLoss` |
| `RegressionEvaluator` | `rmse` (default), `mse`, `mae`, `r2`, `var` |

<!-- snippet: src/db_ml_associate/ch11_evaluation/metrics.py#spark_classification -->
```python
binary = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol="probability")
multiclass = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction")
return {
    "roc_auc": binary.evaluate(predictions, {binary.metricName: "areaUnderROC"}),
    "pr_auc": binary.evaluate(predictions, {binary.metricName: "areaUnderPR"}),
    "accuracy": multiclass.evaluate(predictions, {multiclass.metricName: "accuracy"}),
    # F1 of the positive class; plain "f1" is the weighted average over both classes
    "f1": multiclass.evaluate(
        predictions,
        {multiclass.metricName: "fMeasureByLabel", multiclass.metricLabel: 1.0},
    ),
    "log_loss": multiclass.evaluate(predictions, {multiclass.metricName: "logLoss"}),
}
```

<!-- snippet: src/db_ml_associate/ch11_evaluation/metrics.py#spark_regression -->
```python
evaluator = RegressionEvaluator(labelCol=label, predictionCol="prediction")
return {name: evaluator.evaluate(predictions, {evaluator.metricName: name}) for name in ("rmse", "mae", "r2")}
```

> **Exam tip:** `MulticlassClassificationEvaluator`'s default `f1` is a *weighted average over both
> classes*, not the F1 of the positive class. For binary problems use
> `BinaryClassificationEvaluator` for AUC, or `fMeasureByLabel` with `metricLabel=1.0` for the
> positive class's F1.

## Choosing the right metric

<!-- objectives: S3.13 -->

Start from the **cost of each kind of error** in the scenario:

<!-- snippet: src/db_ml_associate/ch11_evaluation/metric_choice.py#metric_choice -->
```python
METRIC_FOR_SCENARIO = {
    Scenario.MISSED_POSITIVES_COSTLY: "recall",
    Scenario.FALSE_ALARMS_COSTLY: "precision",
    Scenario.IMBALANCED_BOTH_ERRORS_MATTER: "F1",
    Scenario.WELL_CALIBRATED_PROBABILITIES: "log loss",
    Scenario.RANKING_ACROSS_THRESHOLDS: "ROC AUC",
    Scenario.BALANCED_CLASSES_EQUAL_COSTS: "accuracy",
    Scenario.LARGE_ERRORS_ESPECIALLY_BAD: "RMSE",
    Scenario.ROBUST_TO_OUTLIERS: "MAE",
    Scenario.SHARE_OF_VARIANCE_EXPLAINED: "R-squared",
}
```

| Scenario wording | Metric |
|---|---|
| "Missing a fraudulent transaction / a sick patient is very costly" | **Recall** |
| "Every flagged case triggers an expensive manual review" / "do not block good customers" | **Precision** |
| "Positives are rare; we care about both kinds of error" | **F1** (or PR AUC) |
| "We use the probabilities to compute expected cost / need calibrated probabilities" | **Log loss** |
| "Compare models regardless of the threshold we will pick later" | **ROC AUC** |
| "Classes are balanced and every error costs the same" | **Accuracy** |
| "A few very wrong predictions are disastrous" | **RMSE** |
| "Outliers in the data should not dominate evaluation" / "report average error in euros" | **MAE** |
| "How much of the variation in price does the model explain?" | **R²** |

> **Exam tip:** with an imbalanced label, "accuracy" is almost always the *wrong* answer
> (chapter 8 shows a useless model with 90 % accuracy).

## Log-transformed targets: exponentiate first

<!-- objectives: S3.14 -->

Right-skewed targets such as prices are often modelled on the log scale (chapter 7). The model
then predicts `log(price)`. Two consequences:

1. **Metrics must be computed in the original unit.** An RMSE computed on logs is not in euros and
   cannot be compared with a model trained on raw prices.
2. **Predictions must be back-transformed before anyone reads them.** A predicted value of 11.9 is
   not a price; `exp(11.9) ≈ €147,000` is.

Undo the exact transform you applied: `np.exp` undoes `np.log`, `np.expm1` undoes `np.log1p`
(Spark: `F.exp`, `F.expm1`).

> **⚠️ Wrong approach:** RMSE on the log scale.

<!-- snippet: src/db_ml_associate/ch11_evaluation/pitfalls.py#wrong_rmse_on_log_scale -->
```python
log_predictions = model.predict(x_test)
rmse = root_mean_squared_error(np.log(test[HOUSE_TARGET]), log_predictions)  # ≈ 0.37 "log euros"
```

<!-- snippet: src/db_ml_associate/ch11_evaluation/pitfalls.py#correct_exponentiate_before_metrics -->
```python
predicted_prices = np.exp(model.predict(x_test))  # undo np.log (use np.expm1 to undo np.log1p)
rmse = root_mean_squared_error(test[HOUSE_TARGET], predicted_prices)  # in euros
```

On the house data the "log" RMSE is **0.37**; the real RMSE after exponentiating is **€58,455**.
Only the second number can be compared with other models or explained to the business.

> **Beyond the exam:** exponentiating a prediction of the *mean* of `log(y)` gives roughly the
> *median* of `y`, which slightly underestimates the mean for skewed data. Smearing corrections
> exist, but the exam only expects you to exponentiate.

## Choosing an approach

| Need | scikit-learn | Spark ML |
|---|---|---|
| Classification metrics on in-memory predictions | `sklearn.metrics.*` | — |
| Metrics on a predictions DataFrame from a Spark model | — | `BinaryClassificationEvaluator`, `MulticlassClassificationEvaluator` |
| Regression metrics | `root_mean_squared_error`, `mean_absolute_error`, `r2_score` | `RegressionEvaluator` |
| Metric during tuning | `scoring=` in `GridSearchCV` (chapter 10) | `evaluator=` in `CrossValidator` (chapter 10) |
| Many metrics logged automatically | `mlflow.sklearn.autolog()` (chapter 4) | `mlflow.pyspark.ml.autolog()` (chapter 4) |

## Pitfalls

1. **Accuracy on imbalanced data** — hides a model that never predicts the rare class.
2. **Metrics on the log scale** — not in business units, not comparable (S3.14).
3. **Forgetting to back-transform predictions** before showing them to users or writing them to a
   table.
4. **Using the default Spark `f1`** for a binary problem when the question asks for the positive
   class's F1.
5. **Comparing models on different test sets** — metric differences then reflect the data, not the
   models.
6. **ROC AUC for very rare positives** — can look excellent while precision is poor; check PR AUC
   too.

## Summary

- Precision answers "how many alarms are real", recall "how many real cases we caught", F1 balances
  them; log loss judges probabilities; ROC AUC judges ranking across thresholds.
- RMSE punishes large errors, MAE treats all errors equally, R² reports explained variance.
- Choose the metric from the cost of errors in the scenario.
- Exponentiate predictions of a log-transformed target before computing metrics or interpreting them.

## Review questions

<!-- questions: ch11 -->

### Question 1

A churn model flags 200 customers as likely to churn. 150 of them actually churn. In total 300 customers churned. What are the model's precision and recall?

- **A.** Precision 0.75, recall 0.50
- **B.** Precision 0.50, recall 0.75
- **C.** Precision 0.75, recall 0.75
- **D.** Precision 0.67, recall 0.50

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Precision = TP / predicted positives = 150 / 200 = 0.75. Recall = TP / actual positives = 150 / 300 = 0.50.

Why the other options are wrong:

- **B**: Swaps the two definitions.
- **C**: Uses the predicted positives as the denominator for both.
- **D**: 0.67 is 200 / 300, which is neither metric.

Objective S3.11 · [Revisit](11-evaluation.md#classification-metrics)

</details>

### Question 2

Which TWO classification metrics are computed from predicted probabilities and therefore do not depend on the chosen decision threshold?

- **A.** ROC AUC
- **B.** Log loss
- **C.** F1 score
- **D.** Precision
- **E.** Accuracy

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — ROC AUC summarises ranking quality over all thresholds and log loss scores the probabilities directly; neither needs a threshold.

Why the other options are wrong:

- **C**: F1 is computed from hard predictions, which depend on the threshold.
- **D**: Precision is computed from hard predictions.
- **E**: Accuracy is computed from hard predictions.

Objective S3.11 · [Revisit](11-evaluation.md#classification-metrics)

</details>

### Question 3

A data scientist evaluates a binary Spark ML classifier with MulticlassClassificationEvaluator(metricName="f1") and reports the result as "the F1 score of the fraud class". Why is that statement inaccurate?

- **A.** The default f1 metric is a weighted average of the F1 scores of both classes
- **B.** MulticlassClassificationEvaluator cannot be used for binary labels
- **C.** f1 in Spark is computed from probabilities, not predictions
- **D.** Spark always reports F1 for label 0 only

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — The default "f1" is the class-frequency-weighted average; for the positive class use metricName="fMeasureByLabel" with metricLabel=1.0.

Why the other options are wrong:

- **B**: The evaluator works for any number of classes, including two.
- **C**: F1 uses the prediction column, not probabilities.
- **D**: It averages over all labels unless a specific label metric is requested.

Objective S3.11 · [Revisit](11-evaluation.md#the-same-metrics-in-spark-ml)

</details>

### Question 4

Two house-price models have the same MAE of 20,000 euros. Model A has an RMSE of 25,000 and model B an RMSE of 60,000. What does this tell you?

- **A.** Model B makes a few very large errors that model A avoids
- **B.** Model B is better because its RMSE is higher
- **C.** Both models are identical because MAE is the main metric
- **D.** Model A must have a negative R-squared

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — RMSE squares errors, so a much larger RMSE at equal MAE means some predictions of model B are far off.

Why the other options are wrong:

- **B**: Lower RMSE is better.
- **C**: Equal MAE does not mean equal error distributions.
- **D**: Nothing here implies model A is worse than predicting the mean.

Objective S3.12 · [Revisit](11-evaluation.md#regression-metrics)

</details>

### Question 5

A regression model has an R-squared of 0 on the test set. What does this mean?

- **A.** The model predicts perfectly
- **B.** The model explains none of the variance; it is no better than always predicting the mean
- **C.** The RMSE is zero
- **D.** The model is overfitting

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — R-squared compares the model's squared error with that of predicting the mean; 0 means no improvement.

Why the other options are wrong:

- **A**: A perfect model has R-squared 1.
- **C**: RMSE zero corresponds to R-squared 1.
- **D**: R-squared on one test set alone cannot show overfitting; it needs a comparison with training performance.

Objective S3.12 · [Revisit](11-evaluation.md#regression-metrics)

</details>

### Question 6

A hospital wants a model that flags patients who may have a rare but treatable disease. Flagged patients get a cheap follow-up test; a missed case can be fatal. Which metric should drive model selection?

- **A.** Precision
- **B.** Recall
- **C.** Accuracy
- **D.** R-squared

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — Missing a positive case is the costly error, so the model must find as many real cases as possible: maximise recall.

Why the other options are wrong:

- **A**: Precision matters when false alarms are expensive, but follow-up tests are cheap here.
- **C**: Accuracy is dominated by the many healthy patients.
- **D**: R-squared is a regression metric.

Objective S3.13 · [Revisit](11-evaluation.md#choosing-the-right-metric)

</details>

### Question 7

A delivery company predicts delivery times. A few deliveries take days because of customs incidents, and management wants the evaluation to reflect typical errors rather than these extreme cases. Which metric fits?

- **A.** RMSE
- **B.** MAE
- **C.** Log loss
- **D.** ROC AUC

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — MAE weighs every error linearly, so a few extreme errors do not dominate the score.

Why the other options are wrong:

- **A**: RMSE squares errors and is dominated by the extreme cases.
- **C**: Log loss is a classification metric.
- **D**: ROC AUC is a classification metric.

Objective S3.13 · [Revisit](11-evaluation.md#choosing-the-right-metric)

</details>

### Question 8

An insurer uses a model's predicted claim probability to compute an expected cost per customer and price the policy. Two models have the same ROC AUC. Which metric best tells them apart for this use?

- **A.** Accuracy at threshold 0.5
- **B.** Log loss
- **C.** Recall
- **D.** MAE of the predicted class labels

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — The probabilities themselves are used, so they must be well calibrated; log loss directly penalises poorly calibrated, overconfident probabilities.

Why the other options are wrong:

- **A**: Accuracy ignores the probability values.
- **C**: Recall depends on one threshold and ignores calibration.
- **D**: MAE of class labels is not a standard classification metric.

Objective S3.13 · [Revisit](11-evaluation.md#choosing-the-right-metric)

</details>

### Question 9

A model was trained on np.log1p(price). Its RMSE on the test set, computed directly from the model's output and np.log1p(test_price), is 0.28. The team wants to compare it with an earlier model whose RMSE is 41,000 euros. What should they do?

- **A.** Compare 0.28 with 41,000 directly; the new model is far better
- **B.** Apply np.expm1 to the predictions, then compute RMSE against the original prices
- **C.** Apply np.log1p to the earlier model's RMSE
- **D.** Multiply 0.28 by the average price

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — The predictions are on the log1p scale; back-transform them with expm1 and compute the metric in euros so both models are measured in the same unit.

Why the other options are wrong:

- **A**: The numbers are in different units and cannot be compared.
- **C**: Transforming an aggregated metric does not give a comparable metric.
- **D**: This is not a valid conversion of RMSE between scales.

Objective S3.14 · [Revisit](11-evaluation.md#log-transformed-targets-exponentiate-first)

</details>

### Question 10

A Spark ML model predicts log(salary). A dashboard shows the prediction column directly, and users see values such as 10.8. What is the fix?

- **A.** Train the model on salary instead of log(salary)
- **B.** Write exp(prediction) to the table that feeds the dashboard
- **C.** Round the predictions to whole numbers
- **D.** Divide the predictions by the mean salary

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — Predictions of a log-transformed target must be exponentiated before they are interpreted.

Why the other options are wrong:

- **A**: Retraining is unnecessary; the log model may be the better model.
- **C**: Rounding a log value still leaves it in log units.
- **D**: Scaling does not undo a logarithm.

Objective S3.14 · [Revisit](11-evaluation.md#log-transformed-targets-exponentiate-first)

</details>
<!-- /questions -->
