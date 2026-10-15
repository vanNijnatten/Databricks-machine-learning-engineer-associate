---
chapter: 8
title: ML foundations
objectives: [S3.01, S3.02, S3.15]
prerequisites: []
reading_minutes: 90
tutorial:
  notebook: notebooks/ch08_foundations.py
  minutes: 30
  requires: [serverless]
  free_edition: full
  cleanup: false
  extra_files: []
  steps:
    - id: algorithm-choice
      objectives: [S3.01]
    - id: imbalance
      objectives: [S3.02]
    - id: bias-variance
      objectives: [S3.15]
references: [train-models, dbr-18-lts-ml]
---

# 8. ML foundations

> **Exam mapping**: Section 3 — Model Development · objectives S3.01, S3.02, S3.15 ·
> **Runtime**: Databricks Runtime 18 LTS ML (see [`runtime.toml`](../runtime.toml)) ·
> **Read this chapter first**: it has no prerequisites and later chapters build on it.

The exam assumes you can reason about machine learning without a Databricks feature in sight:
which algorithm suits a scenario, what to do when one class is rare, and why a model that is
perfect on its training data can be useless. This chapter covers exactly those foundations, with
code you can run on the tutorial datasets described in [`data/README.md`](../data/README.md).

> **Tutorial:** [`notebooks/ch08_foundations.py`](../notebooks/ch08_foundations.py) · about 30 minutes · **Requires:** serverless or classic compute · **Free Edition:** yes · **Clean-up:** none. Run [`notebooks/00_setup.py`](../notebooks/00_setup.py) once first.

## Learning objectives

- Pick a suitable algorithm family for a described business scenario and justify the choice.
- Recognise class imbalance and choose between cost-sensitive learning, resampling, threshold
  tuning and better metrics.
- Explain bias, variance and model complexity, and diagnose under- and overfitting from training
  and validation errors.
- Avoid the two classic imbalance traps: resampling before the split and judging by accuracy.

## Supervised learning in one page

<!-- objectives: S3.01 -->

In **supervised learning** every training row has known *features* (inputs) and a *label*
(the answer). The model learns a mapping from features to label and is then applied to rows whose
label is unknown.

| Problem type | Label | Example from this book | Typical metrics (chapter 11) |
|---|---|---|---|
| **Binary classification** | One of two classes | Will this customer churn? (`churned`) | F1, ROC AUC, log loss |
| **Multiclass classification** | One of *k* classes | Which plan will the customer pick? | Accuracy, macro F1 |
| **Regression** | A number | What will this house sell for? (`price`) | RMSE, MAE, R² |
| **Forecasting** | Future values of a time series | Daily sales for the next 30 days | MAPE, RMSE |

**Unsupervised learning** has no label: clustering (k-means) groups similar rows, and
dimensionality reduction (PCA) compresses features. **Recommendation** (collaborative filtering
with ALS in Spark ML) learns from user–item interactions.

Every honest evaluation uses data the model did not see while learning. You split the data into
**training** data (to fit the model), **validation** data (to choose between models and
hyperparameters, chapter 10) and **test** data (to report final performance once).

## Choosing an algorithm for a scenario

<!-- objectives: S3.01 -->

The exam describes a scenario and asks which algorithm (family) fits best. Work through four
questions:

1. **What is the label?** A category → classification. A number → regression. A future value of a
   time series → forecasting. No label → clustering or anomaly detection.
2. **Must the model be explained?** Regulators or business users who need to see *why* push you to
   linear/logistic regression or a shallow decision tree.
3. **Are relationships non-linear, with interactions?** Tree ensembles capture them without manual
   feature engineering; linear models do not.
4. **How big is the data?** Data that fits in one machine's memory trains fastest with
   scikit-learn, XGBoost or LightGBM on a single node. Data that does not fit needs a distributed
   library such as Spark ML (`pyspark.ml`).

| Algorithm family | Good when | Watch out for |
|---|---|---|
| **Linear regression / logistic regression** | You need a fast, interpretable baseline; relationships are roughly linear; many sparse features (e.g. one-hot encoded) | Needs scaled numeric features and encoded categoricals; misses non-linear effects unless you engineer them |
| **Decision tree** | You need rules a person can read; mixed feature types | A single deep tree overfits badly (see below) |
| **Random forest** (bagging of trees) | Strong default for tabular data with little tuning; robust to outliers and unscaled features | Larger models, slower predictions, less interpretable |
| **Gradient-boosted trees** (XGBoost, LightGBM, Spark `GBTClassifier`) | Usually the most accurate choice for tabular data | More hyperparameters to tune; can overfit without early stopping or regularisation |
| **k-means** | Segmenting customers without labels | You must choose *k*; sensitive to feature scale |
| **ALS** (Spark ML) | Recommendations from user–item ratings or clicks | Cold start for new users/items |
| **Prophet / ARIMA** | Forecasting a time series (AutoML uses these, chapter 2) | Needs a time column and regular frequency |

On Databricks, the ML runtime already includes these libraries — scikit-learn, XGBoost,
LightGBM and Spark ML — so you can switch between them without installing anything
([Train AI and ML models](https://docs.databricks.com/aws/en/machine-learning/train-model/),
[Runtime 18 LTS ML release notes](https://docs.databricks.com/aws/en/release-notes/runtime/18ml)).

> **Exam tip:** read the scenario for the *constraint* that decides the answer — "must explain each
> prediction to auditors" (linear model or shallow tree), "highest accuracy on tabular data"
> (gradient-boosted trees), "no labels" (clustering), "data does not fit on one machine"
> (Spark ML). If nothing constrains the choice, a tree ensemble is the usual answer for tabular data.

## Class imbalance

<!-- objectives: S3.02 -->

A dataset is **imbalanced** when one class is much rarer than the other. In `customer_churn`
only 10 % of customers churn. A model trained naively learns that "nobody churns" is right 90 %
of the time and barely looks at the minority class — the one the business cares about.

The table shows a logistic regression on the churn data, evaluated on a stratified test split
(numbers from the chapter's code, seed 42):

| Training approach | Accuracy | Recall (churners found) | F1 (churn) |
|---|---|---|---|
| Plain logistic regression | 0.902 | 0.03 | 0.06 |
| `class_weight="balanced"` | 0.682 | 0.75 | 0.32 |
| Random oversampling of the training split | 0.683 | 0.75 | 0.32 |
| Random undersampling of the training split (800 rows left) | 0.678 | 0.75 | 0.32 |

Accuracy *drops* as the model becomes useful — which is exactly why accuracy is the wrong metric
here (see Pitfalls).

### Cost-sensitive learning (class weights)

Tell the algorithm that a mistake on a churner costs more than a mistake on a loyal customer. The
data is unchanged; the loss function weighs minority errors more. scikit-learn's `"balanced"`
weights are inversely proportional to class frequency:

<!-- snippet: src/db_ml_associate/ch08_foundations/imbalance.py#balanced_weights -->
```python
counts = labels.value_counts()
n_samples, n_classes = len(labels), len(counts)
return {int(label): n_samples / (n_classes * count) for label, count in counts.items()}
```

Most scikit-learn classifiers accept the weights directly:

<!-- snippet: src/db_ml_associate/ch08_foundations/imbalance.py#class_weight -->
```python
model = estimator or LogisticRegression(class_weight=class_weight, max_iter=1000)
```

Spark ML estimators have no `class_weight` argument. Instead you add a weight column and pass its
name as `weightCol` (supported by `LogisticRegression`, `RandomForestClassifier`,
`GBTClassifier` and others):

<!-- snippet: src/db_ml_associate/ch08_foundations/imbalance.py#spark_weight_col -->
```python
n_rows = F.count(F.lit(1)).over(Window.partitionBy())
n_in_class = F.count(F.lit(1)).over(Window.partitionBy(label))
n_classes = frame.select(label).distinct().count()
weighted = frame.withColumn(weight_col, n_rows / (F.lit(n_classes) * n_in_class))
# Then: LogisticRegression(labelCol=label, weightCol=weight_col)
```

> **Exam tip:** "assign a higher misclassification cost to the minority class" is the textbook
> answer for *directly* reducing a model's bias towards the majority class. Normalising features,
> using a simpler model or collecting more *majority* data do not address imbalance.

### Resampling

Resampling changes the training data instead of the loss:

- **Random oversampling** duplicates minority rows until the classes are balanced.
- **Random undersampling** drops majority rows; fast, but throws information away.
- **SMOTE** synthesises new minority rows by interpolating between neighbours; it is available in
  the `imbalanced-learn` package (preinstalled in the ML runtime).

<!-- snippet: src/db_ml_associate/ch08_foundations/imbalance.py#oversample -->
```python
counts = train[label].value_counts()
minority = train[train[label] == counts.idxmin()]
extra = minority.sample(n=counts.max() - counts.min(), replace=True, random_state=seed)
return pd.concat([train, extra], ignore_index=True)
```

Resample **only the training split**, after splitting. Validation and test data must keep the real
class ratio, otherwise your scores describe a world that does not exist.

### Other levers

- **Threshold tuning**: a classifier outputs a probability; lowering the decision threshold from
  0.5 trades precision for recall.
- **Better metrics**: judge models by recall, precision, F1 or PR AUC of the minority class
  (chapter 11), never by accuracy alone.
- **Stratified splits and cross-validation** keep the class ratio equal in every split
  (chapter 10).
- **More minority data**, if it can be collected, is the best fix of all.

## Bias, variance and model complexity

<!-- objectives: S3.15 -->

A model's error on new data has two learnable sources:

- **Bias** — error from wrong assumptions. A model that is too simple *underfits*: it cannot
  represent the real pattern, so it is inaccurate on training *and* validation data.
- **Variance** — error from sensitivity to the particular training sample. A model that is too
  flexible *overfits*: it memorises noise, scores almost perfectly on training data and much worse
  on validation data.

Increasing **model complexity** (deeper trees, more features, higher polynomial degree, weaker
regularisation) lowers bias and raises variance. The best model sits in between. The chapter's code
fits decision trees of growing depth to predict `log(price)`:

<!-- snippet: src/db_ml_associate/ch08_foundations/bias_variance.py#sweep -->
```python
for depth in depths:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=seed).fit(x_train, y_train)
    rows.append(
        {
            "depth": depth,
            "train_rmse": root_mean_squared_error(y_train, tree.predict(x_train)),
            "validation_rmse": root_mean_squared_error(y_validation, tree.predict(x_validation)),
        }
    )
```

| `max_depth` | Train RMSE | Validation RMSE | Diagnosis |
|---|---|---|---|
| 1 | 0.388 | 0.394 | High bias: both errors high and close |
| 3 | 0.360 | 0.376 | Good balance |
| 4 | 0.353 | 0.376 | Good balance (lowest validation error) |
| 8 | 0.300 | 0.422 | Variance growing: gap widens |
| 20 | 0.032 | 0.531 | High variance: memorised training data |

| Symptom | Diagnosis | Remedies |
|---|---|---|
| High training error, validation error similar | Underfitting (high bias) | More complex model, more/better features, less regularisation, train longer |
| Low training error, much higher validation error | Overfitting (high variance) | Simpler model, stronger regularisation (L1/L2, smaller trees), more training data, fewer features, early stopping, bagging |

Ensembles attack the two sources differently: **bagging** (random forest) averages many deep trees
to *reduce variance*; **boosting** (gradient-boosted trees) adds shallow trees one by one to
*reduce bias*.

> **Exam tip:** "training accuracy 99 %, validation accuracy 70 %" is overfitting — the answer is
> never "make the model more complex". "Both scores are poor and similar" is underfitting — adding
> training data alone will not help.

## Choosing an approach

| Situation | Prefer | Why |
|---|---|---|
| Imbalanced labels, algorithm supports weights | **Class weights** (`class_weight`, `weightCol`) | No data duplication, no information loss, one line of code |
| Algorithm has no weight support | **Oversampling** (or SMOTE) of the training split | Balances what the algorithm sees |
| Huge majority class, training is slow | **Undersampling** of the training split | Smaller, faster training set; accept some information loss |
| You need a specific precision/recall trade-off | **Threshold tuning** on validation data | Keeps the model, moves the decision point |
| Overfitting | Regularise / simplify / more data | Reduces variance |
| Underfitting | More complex model / better features | Reduces bias |

## Pitfalls

**1. Resampling before the split.** Oversampling the full dataset puts copies of the same churner
in both training and validation data. A decision tree memorises them and the validation score
becomes meaningless.

> **⚠️ Wrong approach:** oversample everything, then split.

<!-- snippet: src/db_ml_associate/ch08_foundations/pitfalls.py#wrong_resample_before_split -->
```python
balanced = random_oversample(churn, CHURN_LABEL, seed)  # duplicates before the split ...
train, validation = split_frame(balanced, CHURN_LABEL, seed=seed)  # ... leak into validation
```

Observed consequence on the churn data: validation F1 **0.95** — a fantasy.

<!-- snippet: src/db_ml_associate/ch08_foundations/pitfalls.py#correct_resample_after_split -->
```python
train, validation = split_frame(churn, CHURN_LABEL, seed=seed)
train = random_oversample(train, CHURN_LABEL, seed)  # validation keeps the real class ratio
```

Split first: the honest validation F1 of the same model is **0.24**.

**2. Judging an imbalanced problem by accuracy.** A "model" that predicts *no churn* for everyone
scores 90 % accuracy and finds zero churners (recall 0.0). Report minority-class recall,
precision, F1 or PR AUC instead.

**3. Fitting preprocessing on all data.** Imputation medians, scaling statistics and encoders must be
learned from training data only — the same leakage as pitfall 1 (chapter 9 shows how pipelines
prevent it).

**4. Treating the test set as a validation set.** If you pick models by their test score, the test
score is no longer an unbiased estimate. Choose with validation data or cross-validation
(chapter 10); touch the test set once.

**5. Mixing up the remedies.** More data helps variance, not bias. A more complex model helps bias
and hurts variance.

## Summary

- Choose algorithms by label type, interpretability needs, non-linearity and data size; tree
  ensembles are the default for tabular data, linear models for interpretability, Spark ML for data
  that does not fit on one machine.
- Imbalance is mitigated by cost-sensitive learning (class weights / `weightCol`), resampling the
  training split, threshold tuning and minority-class metrics — not by accuracy.
- Bias (underfitting) and variance (overfitting) trade off through model complexity; compare training
  and validation errors to tell which one you have.

## Review questions

<!-- questions: ch08 -->

### Question 1

A bank must predict whether a loan applicant will default. Regulators require that every decision can be explained to the applicant in terms of how each input pushed the prediction up or down. Which algorithm family is the best starting point?

- **A.** Gradient-boosted trees with 2,000 trees
- **B.** Logistic regression
- **C.** k-means clustering
- **D.** A deep neural network

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — Logistic regression is a classification model whose coefficients directly show how each feature moves the log-odds of default, which satisfies the explainability requirement.

Why the other options are wrong:

- **A**: Large boosted ensembles are accurate but hard to explain decision by decision.
- **C**: k-means is unsupervised clustering; it does not predict a labelled outcome.
- **D**: Neural networks are the least interpretable option here.

Objective S3.01 · [Revisit](08-ml-foundations.md#choosing-an-algorithm-for-a-scenario)

</details>

### Question 2

A marketing team has three years of customer purchase data but no labels. They want to discover groups of customers with similar behaviour to design targeted campaigns. Which approach fits?

- **A.** Linear regression on total spend
- **B.** A random forest classifier
- **C.** k-means clustering
- **D.** An ALS recommender

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: C** — Without a label the task is unsupervised; k-means groups similar customers into segments.

Why the other options are wrong:

- **A**: Regression needs a numeric label to predict and does not create groups.
- **B**: A classifier needs labelled classes, which do not exist yet.
- **D**: ALS recommends items from user-item interactions; it does not segment customers.

Objective S3.01 · [Revisit](08-ml-foundations.md#choosing-an-algorithm-for-a-scenario)

</details>

### Question 3

A data scientist must predict house prices from 40 tabular features, some with strong non-linear effects and interactions. Accuracy matters most and the data fits comfortably in memory on one node. Which TWO choices are most appropriate?

- **A.** Gradient-boosted trees such as XGBoost or LightGBM
- **B.** A random forest regressor
- **C.** Logistic regression
- **D.** k-means with k equal to the number of price bands
- **E.** A single decision tree with no depth limit

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — Tree ensembles capture non-linear effects and interactions without manual feature engineering and are the usual strongest choices for tabular regression.

Why the other options are wrong:

- **C**: Logistic regression is a classification model, and linear models miss non-linear effects.
- **D**: Clustering does not learn to predict a numeric label.
- **E**: An unlimited single tree overfits (high variance) and generalises poorly.

Objective S3.01 · [Revisit](08-ml-foundations.md#choosing-an-algorithm-for-a-scenario)

</details>

### Question 4

A fraud model is trained on transactions of which 0.5% are fraudulent. The data scientist uses scikit-learn's RandomForestClassifier and wants the model to stop favouring the legitimate class without changing the training data. What should they do?

- **A.** Set class_weight="balanced" on the classifier
- **B.** Standardise all numeric features before training
- **C.** Remove the fraudulent transactions that look like outliers
- **D.** Report accuracy instead of recall

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Class weights make errors on the rare class cost more in the training loss (cost-sensitive learning) without duplicating or deleting rows.

Why the other options are wrong:

- **B**: Scaling does not change the class ratio, and trees do not need scaled features.
- **C**: Removing minority rows makes the imbalance worse.
- **D**: Accuracy hides poor minority performance; it is a reporting choice, not a mitigation.

Objective S3.02 · [Revisit](08-ml-foundations.md#cost-sensitive-learning-class-weights)

</details>

### Question 5

A data scientist trains a Spark ML LogisticRegression on a DataFrame in which only 3% of rows have label 1. How can they apply cost-sensitive learning with Spark ML?

- **A.** Pass class_weight="balanced" to LogisticRegression
- **B.** Add a column with a larger weight for label-1 rows and pass its name as weightCol
- **C.** Call df.repartition(2, "label") before fitting
- **D.** Set the threshold parameter to 0.97 before fitting

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — Spark ML estimators accept per-row weights through weightCol; giving minority rows a larger weight is the Spark equivalent of class weights.

Why the other options are wrong:

- **A**: class_weight is a scikit-learn argument; Spark ML has no such parameter.
- **C**: Repartitioning changes data layout across executors, not the loss function.
- **D**: Raising the threshold makes the model predict label 1 even less often.

Objective S3.02 · [Revisit](08-ml-foundations.md#cost-sensitive-learning-class-weights)

</details>

### Question 6

To fix class imbalance, an engineer randomly oversamples the minority class of the full dataset and then splits it 80/20 into training and validation data. A decision tree reaches F1 0.95 on validation data but performs poorly in production. What is the most likely cause?

- **A.** Decision trees cannot be used for imbalanced data
- **B.** Duplicated minority rows ended up in both splits, so validation rows were memorised during training
- **C.** The validation set was too small
- **D.** F1 is not defined for imbalanced data

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — Resampling before splitting leaks copies of training rows into validation data, inflating the score. Split first and resample only the training split.

Why the other options are wrong:

- **A**: Trees work fine on imbalanced data, especially with class weights.
- **C**: 20% of the data is a normal validation size; the problem is leakage, not size.
- **D**: F1 is a common metric precisely for imbalanced classification.

Objective S3.02 · [Revisit](08-ml-foundations.md#pitfalls)

</details>

### Question 7

A churn dataset has 8% churners. Which TWO actions directly help a classifier detect more churners?

- **A.** Random undersampling of the non-churners in the training split
- **B.** Collecting more examples of customers who did not churn
- **C.** Lowering the decision threshold applied to the predicted churn probability
- **D.** Evaluating the model with accuracy on the test set
- **E.** Converting all categorical features to strings

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, C** — Undersampling the majority in the training split balances what the model sees, and a lower threshold labels more customers as churners, raising recall.

Why the other options are wrong:

- **B**: More majority data increases the imbalance.
- **D**: Accuracy rewards ignoring the minority class; it does not improve detection.
- **E**: Changing data types has no effect on imbalance.

Objective S3.02 · [Revisit](08-ml-foundations.md#other-levers)

</details>

### Question 8

A gradient-boosted model reaches an RMSE of 2.1 on training data and 9.8 on validation data. What best describes the model and a suitable remedy?

- **A.** Underfitting; add more trees and deeper trees
- **B.** Overfitting; increase regularisation or reduce tree depth
- **C.** Underfitting; remove features
- **D.** The model is well balanced; deploy it

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — A large gap between low training error and high validation error indicates high variance (overfitting); a simpler or more regularised model reduces it.

Why the other options are wrong:

- **A**: More and deeper trees increase complexity and worsen overfitting.
- **C**: Underfitting shows high error on both splits, which is not the case here.
- **D**: The validation error is nearly five times the training error.

Objective S3.15 · [Revisit](08-ml-foundations.md#bias-variance-and-model-complexity)

</details>

### Question 9

A linear regression predicting delivery time has an R-squared of 0.31 on training data and 0.30 on validation data. Which change is most likely to improve it?

- **A.** Collect three times more training rows of the same features
- **B.** Add stronger L2 regularisation
- **C.** Use a more flexible model or engineer features that capture non-linear effects
- **D.** Reduce the number of features

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: C** — Low and similar scores on both splits indicate high bias (underfitting); the model needs more capacity or better features.

Why the other options are wrong:

- **A**: More data mainly reduces variance; it does not fix a model that is too simple.
- **B**: Regularisation reduces complexity and would increase bias further.
- **D**: Fewer features makes the model even simpler.

Objective S3.15 · [Revisit](08-ml-foundations.md#bias-variance-and-model-complexity)

</details>

### Question 10

Which statement about ensembles and the bias-variance trade-off is correct?

- **A.** Bagging (as in random forests) mainly reduces variance by averaging many trees
- **B.** Boosting mainly reduces variance by training trees independently
- **C.** Both bagging and boosting increase bias to reduce variance
- **D.** Ensembles cannot overfit

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Averaging many high-variance trees trained on bootstrap samples (bagging) cancels out their individual noise, lowering variance.

Why the other options are wrong:

- **B**: Boosting trains trees sequentially to correct previous errors, mainly reducing bias.
- **C**: Neither technique is designed to increase bias.
- **D**: Boosted ensembles in particular overfit without regularisation or early stopping.

Objective S3.15 · [Revisit](08-ml-foundations.md#bias-variance-and-model-complexity)

</details>
<!-- /questions -->
