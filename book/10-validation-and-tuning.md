---
chapter: 10
title: Validation and hyperparameter tuning
objectives: [S3.05, S3.06, S3.07, S3.08, S3.09, S3.10]
prerequisites: [8, 9, 11]
reading_minutes: 120
references: [hyperparameter-tuning, hyperopt-concepts, optuna, serverless-limitations, spark-mllib, dbr-18-lts-ml]
---

# 10. Validation and hyperparameter tuning

> **Exam mapping**: Section 3 — Model Development · objectives S3.05–S3.10 ·
> **Runtime**: Databricks Runtime 18 LTS ML · **Read after**: chapters 8, 9 and 11.

**Hyperparameters** are the settings you choose before training — tree depth, regularisation
strength, number of trees. **Tuning** tries many settings and keeps the one that generalises best.
Doing that honestly needs **validation**: a way to score each setting on data the model did not
train on. This chapter covers cross-validation, the three search strategies the exam names, Hyperopt
(still on the exam) next to Optuna (what Databricks recommends today), parallel tuning, and how to
count the models a search trains.

## Learning objectives

- Explain the benefits and downsides of k-fold cross-validation compared with a single
  train–validation split.
- Run cross-validation as part of model fitting in scikit-learn and Spark ML.
- Count the models trained by a grid search with cross-validation.
- Tune hyperparameters with grid, random and Bayesian search.
- Use Hyperopt's `fmin` (exam) and Optuna (current platform), and parallelise single-node trials.

## Train, validation and test data

<!-- objectives: S3.08 -->

| Split | Used for | Touched |
|---|---|---|
| **Training** | Fitting model parameters (coefficients, tree splits) | Every fit |
| **Validation** | Comparing hyperparameter settings and models | Every candidate |
| **Test** | Reporting the final performance of the chosen model | **Once**, at the end |

Choosing a model by its validation score makes that score slightly optimistic — you picked the
winner of a contest. That is fine, because the untouched test set gives the final, unbiased estimate.
If you choose on the *test* set, nothing unbiased is left (see Pitfalls).

## Cross-validation vs a train–validation split

<!-- objectives: S3.08, S3.09 -->

A **train–validation split** holds out one fixed part of the training data (say 20 %) for
validation. **k-fold cross-validation** splits the training data into *k* folds; each fold is the
validation set once while the model trains on the other *k − 1* folds, and the *k* scores are
averaged.

| | Train–validation split | k-fold cross-validation |
|---|---|---|
| Fits per hyperparameter setting | 1 | *k* |
| Compute cost | Low | *k* times higher |
| Variance of the estimate | High: depends on which rows landed in validation | Lower: every row is validated exactly once |
| Use of data | 20 % never trains during tuning | Every row trains (in *k − 1* folds) and validates (in 1) |
| Best for | Large datasets, expensive models | Small and medium datasets, when a reliable estimate matters |

**Benefits of cross-validation:** a more reliable, lower-variance estimate; uses all training data
for both training and validation; reveals how stable the model is across folds.
**Downsides:** *k* times the compute; not directly suitable for time series (use time-ordered splits);
still needs a separate test set for the final estimate.

Use **stratified** folds for classification (`StratifiedKFold`) so that every fold keeps the class
ratio — essential for imbalanced labels (chapter 8).

## Counting the models a search trains

<!-- objectives: S3.10 -->

<a id="counting-models"></a>

A grid search tries every combination of the listed values. With *k*-fold cross-validation every
combination is trained *k* times:

> **models trained = (number of values of hyperparameter 1) × (… of hyperparameter 2) × … × k**

<!-- snippet: src/db_ml_associate/ch10_tuning/cross_validation.py#count_models -->
```python
combinations = math.prod(len(values) for values in grid.values())
return combinations * folds + (1 if include_refit else 0)
```

| Grid | Folds | Models |
|---|---|---|
| `C`: 3 values, `kernel`: 2, `gamma`: 3 | 5 | 3 × 2 × 3 × 5 = **90** |
| `max_depth`: 4 values, `n_estimators`: 3 | 3 | 4 × 3 × 3 = **36** |
| `regParam`: 2, `elasticNetParam`: 2 (`ParamGridBuilder`) | 3 (`CrossValidator`) | 2 × 2 × 3 = **12** |
| same 2 × 2 grid | `TrainValidationSplit` | 2 × 2 × 1 = **4** |
| Random search, `n_iter=20` | 5 | 20 × 5 = **100** (the space size does not matter) |

> **Exam tip:** the official sample question counts **combinations × folds** (90 in the first row).
> Use that convention on the exam.

> **Beyond the exam:** both scikit-learn (`refit=True`, the default) and Spark's `CrossValidator`
> afterwards train the best combination once more on all training data, so strictly 91 and 37 models
> are fitted. The exam answer does not count this final refit.

## Search strategies: grid, random and Bayesian

<!-- objectives: S3.06 -->

| Strategy | How it picks the next candidate | Strengths | Weaknesses |
|---|---|---|---|
| **Grid search** | Every combination of listed values | Exhaustive, reproducible, embarrassingly parallel | Cost explodes with each extra hyperparameter; wastes trials on unimportant ones |
| **Random search** | Independent random draws from distributions | Covers important hyperparameters better for the same budget; budget fixed by `n_iter` | May miss the optimum; no learning from earlier trials |
| **Bayesian search** (e.g. TPE) | A model of past results proposes promising candidates | Finds good settings in fewer trials | Sequential by nature; less parallel-friendly |

Grid and random search in scikit-learn — note that preprocessing lives *inside* the estimator, so it is
refitted in every fold (chapter 9):

<!-- snippet: src/db_ml_associate/ch10_tuning/cross_validation.py#grid_search_cv -->
```python
search = GridSearchCV(
    estimator=build_sklearn_churn_pipeline(),  # preprocessing is refitted inside every fold
    param_grid={"model__C": [0.1, 1.0, 10.0]},  # step name __ parameter name
    scoring="f1",
    cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
    refit=True,  # retrain the best combination on all training data
)
search.fit(features, labels)  # 3 combinations × 5 folds = 15 fits (+ 1 refit)
```

<!-- snippet: src/db_ml_associate/ch10_tuning/cross_validation.py#random_search_cv -->
```python
search = RandomizedSearchCV(
    estimator=build_sklearn_churn_pipeline(),
    param_distributions={"model__C": loguniform(0.01, 100)},  # continuous, log-scaled
    n_iter=n_iter,  # number of sampled candidates, independent of the space size
    scoring="f1",
    cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
    random_state=seed,
)
search.fit(features, labels)
```

### Cross-validation in Spark ML

<!-- objectives: S3.09, S3.06 -->

Spark ML wraps a whole pipeline in `CrossValidator` (k folds) or `TrainValidationSplit` (one
split). Both are estimators: `fit()` searches the `ParamGridBuilder` grid and returns a model that
uses the best parameter map, refitted on all the data
([Use Apache Spark MLlib on Databricks](https://docs.databricks.com/aws/en/machine-learning/train-model/mllib)).
Below, `pipeline` is the churn pipeline from chapter 9 and `classifier` its final
`LogisticRegression` stage, whose parameters the grid varies.

<!-- snippet: src/db_ml_associate/ch10_tuning/cross_validation.py#spark_cross_validator -->
```python
grid = (
    ParamGridBuilder()
    .addGrid(classifier.regParam, [0.0, 0.1])
    .addGrid(classifier.elasticNetParam, [0.0, 1.0])  # 0 = L2 (ridge), 1 = L1 (lasso)
    .build()
)  # 4 parameter maps
cross_validator = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=grid,
    evaluator=BinaryClassificationEvaluator(labelCol=CHURN_LABEL),  # areaUnderROC
    numFolds=folds,  # 4 maps × 3 folds = 12 fits, then 1 refit of the best map on all data
    parallelism=2,  # fit up to 2 models at the same time
    seed=seed,
    collectSubModels=collect_sub_models,
)
```

<!-- snippet: src/db_ml_associate/ch10_tuning/cross_validation.py#spark_train_validation_split -->
```python
return TrainValidationSplit(
    estimator=cross_validator.getEstimator(),
    estimatorParamMaps=cross_validator.getEstimatorParamMaps(),
    evaluator=cross_validator.getEvaluator(),
    trainRatio=train_ratio,  # 80 % train, 20 % validation: 4 maps × 1 split = 4 fits
    seed=seed,
    collectSubModels=collect_sub_models,
)
```

`CrossValidatorModel.avgMetrics` holds the average metric per parameter map;
`TrainValidationSplitModel.validationMetrics` holds one metric per map. `parallelism` fits several
models at once. MLflow's automatic tracking of these classes is deprecated; call
`mlflow.pyspark.ml.autolog()` instead
([Hyperparameter tuning](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/)).

## Hyperopt `fmin` (on the exam)

<!-- objectives: S3.05, S3.06 -->

> **Platform change:** Hyperopt is no longer maintained and is **not included in Databricks Runtime
> ML after 16.4 LTS ML**; Databricks recommends Optuna or Ray Tune instead
> ([Hyperparameter tuning](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/)).
> The exam still asks about `fmin`. To run it on 18 LTS ML: `%pip install hyperopt==0.2.7`
> (it imports `pkg_resources`, which the runtime's preinstalled setuptools 78.1.1 provides).
> See [Appendix A](appendix-a-exam-vs-platform.md).

Hyperopt needs three things: a **search space**, an **objective function** that returns a loss to
**minimise**, and a search **algorithm**.

<!-- snippet: src/db_ml_associate/ch10_tuning/hyperopt_tuning.py#search_space -->
```python
SEARCH_SPACE = {
    "C": hp.loguniform("C", np.log(0.01), np.log(100)),  # samples exp(uniform(log 0.01, log 100))
    "class_weight": hp.choice("class_weight", [None, "balanced"]),
}
```

| Expression | Samples |
|---|---|
| `hp.uniform(label, low, high)` | A float uniformly between low and high |
| `hp.loguniform(label, log(low), log(high))` | A float whose logarithm is uniform — for learning rates and regularisation |
| `hp.quniform(label, low, high, q)` | A uniform value rounded to multiples of q (e.g. tree depth) |
| `hp.choice(label, options)` | One of the options (categorical) |

The objective trains a model with the proposed parameters and returns a dictionary with `loss` and
`status`. **`fmin` minimises the loss**, so return the *negative* of any score you want to maximise:

<!-- snippet: src/db_ml_associate/ch10_tuning/hyperopt_tuning.py#objective -->
```python
def objective(params: dict[str, Any]) -> dict[str, Any]:
    f1 = cross_validated_f1(train, params, seed=seed)
    return {
        "loss": -f1,
        "status": STATUS_OK,
    }  # fmin minimises, so negate a score to maximise
```

<!-- snippet: src/db_ml_associate/ch10_tuning/hyperopt_tuning.py#fmin -->
```python
trials = Trials()  # runs trials one after another on the driver
best_indices = fmin(
    fn=make_objective(train, seed),
    space=SEARCH_SPACE,
    algo=tpe.suggest,  # Bayesian; use hyperopt.rand.suggest for random search
    max_evals=max_evals,
    trials=trials,
    rstate=np.random.default_rng(seed),
    show_progressbar=False,
)
best = space_eval(SEARCH_SPACE, best_indices)  # hp.choice returns an index; space_eval maps it back
```

The `fmin` arguments ([Hyperopt concepts](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/hyperopt-concepts)):
`fn` (objective), `space`, `algo` (`tpe.suggest` for Bayesian TPE, `rand.suggest` for random search),
`max_evals` (number of settings to try), `trials` (`Trials` or `SparkTrials`), and optionally
`early_stop_fn`. `fmin` returns the best point with **indices** for `hp.choice` parameters —
`space_eval(space, best)` turns them back into values.

> **Exam tip:** "Bayesian" in a Hyperopt question means `algo=tpe.suggest`; "random" means
> `algo=rand.suggest`. Hyperopt has no built-in grid search.

## Parallelising single-node models

<!-- objectives: S3.07 -->

A scikit-learn model trains on one machine. Tuning it is still parallel at the level of *trials*: each
trial is independent, so many can run at once on different executors.

**Hyperopt `SparkTrials`** runs each trial as a Spark task. Use it for single-machine algorithms
such as scikit-learn; use plain `Trials` when the objective itself trains a *distributed* model
(Spark ML), because that already uses the whole cluster
([Hyperopt concepts](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/hyperopt-concepts)).

<!-- snippet: src/db_ml_associate/ch10_tuning/hyperopt_tuning.py#spark_trials -->
```python
spark_trials = SparkTrials(parallelism=parallelism, spark_session=spark)  # trials run as Spark tasks
best_indices = fmin(
    fn=make_objective(train, seed),
    space=SEARCH_SPACE,
    algo=tpe.suggest,
    max_evals=max_evals,
    trials=spark_trials,
    rstate=np.random.default_rng(seed),
    show_progressbar=False,
)
```

`parallelism` is the maximum number of concurrent trials: by default the number of executors,
at most 128. More parallelism finishes sooner; less parallelism lets each Bayesian proposal learn from
more completed trials.

> **Requires:** `SparkTrials` drives Spark through the classic SparkContext. Serverless compute
> supports only Spark Connect APIs
> ([Serverless compute limitations](https://docs.databricks.com/aws/en/compute/serverless/limitations)),
> so run `SparkTrials` on **dedicated (classic) compute** with the ML runtime. On serverless or Free
> Edition, use `Trials` or Optuna with `n_jobs`.

**Optuna** is the current recommendation. A study states whether it maximises or minimises — no sign
trick — and the sampler selects the strategy:

<!-- snippet: src/db_ml_associate/ch10_tuning/optuna_tuning.py#samplers -->
```python
samplers: dict[str, optuna.samplers.BaseSampler] = {
    "grid": optuna.samplers.GridSampler(GRID, seed=seed),  # every combination
    "random": optuna.samplers.RandomSampler(seed=seed),  # independent random draws
    "bayesian": optuna.samplers.TPESampler(seed=seed),  # learns from previous trials
}
```

<!-- snippet: src/db_ml_associate/ch10_tuning/optuna_tuning.py#optuna_study -->
```python
def objective(trial: optuna.Trial) -> float:
    params = {
        "C": trial.suggest_float("C", 0.01, 100, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }
    return cross_validated_f1(train, params, seed=seed)

study = optuna.create_study(direction="maximize", sampler=sampler or optuna.samplers.TPESampler(seed=seed))
study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)  # n_jobs > 1: parallel trials (joblib threads)
```

`n_jobs` runs trials in parallel threads on one machine. To spread trials over the cluster, MLflow 3
(preinstalled from Runtime 17.0 ML) provides `MlflowSparkStudy`, which runs trials on PySpark
executors and stores the study in the MLflow tracking server through `MlflowStorage`
([Hyperparameter tuning with Optuna](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/optuna)):

<!-- snippet: src/db_ml_associate/ch10_tuning/optuna_tuning.py#mlflow_spark_study -->
```python
from mlflow.optuna.storage import MlflowStorage
from mlflow.pyspark.optuna.study import MlflowSparkStudy

storage = MlflowStorage(experiment_id=experiment_id)  # MLflow tracking server as Optuna storage
study = MlflowSparkStudy(study_name, storage, sampler=optuna.samplers.TPESampler(seed=seed))
study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)  # trials run on Spark executors
```

| Hyperopt (exam) | Optuna (current) |
|---|---|
| `hp.loguniform("C", log(0.01), log(100))` | `trial.suggest_float("C", 0.01, 100, log=True)` |
| `hp.choice("cw", [None, "balanced"])` | `trial.suggest_categorical("cw", [None, "balanced"])` |
| Objective returns `{"loss": -score, ...}` | `create_study(direction="maximize")`, objective returns the score |
| `algo=tpe.suggest` / `rand.suggest` | `TPESampler` / `RandomSampler` / `GridSampler` |
| `fmin(..., max_evals=20)` | `study.optimize(objective, n_trials=20)` |
| `SparkTrials(parallelism=4)` | `n_jobs=4` (threads) or `MlflowSparkStudy` (executors) |
| `space_eval(space, best)` | `study.best_params` |

## Choosing an approach

| Situation | Prefer |
|---|---|
| Small search space (a handful of combinations) | Grid search |
| Many hyperparameters, fixed budget | Random search or Bayesian (TPE) |
| Each trial is expensive | Bayesian search: fewer trials to a good result |
| Large dataset, expensive model | Train–validation split (`TrainValidationSplit`) |
| Small/medium dataset, reliable estimate needed | k-fold cross-validation (`CrossValidator`, `GridSearchCV`) |
| scikit-learn model, many trials, classic compute | Hyperopt `SparkTrials` (exam) or `MlflowSparkStudy` (current) |
| Spark ML model | `CrossValidator` with `parallelism`, or Hyperopt with plain `Trials` |
| New project on Runtime 18 LTS ML | Optuna |

## Pitfalls

**1. Tuning on the test set.** Picking the best of many models by their test score, then reporting
that score, overstates performance — the same selection bias as in a contest with many entrants.

> **⚠️ Wrong approach:** let the test set choose, then grade its own choice.

<!-- snippet: src/db_ml_associate/ch10_tuning/pitfalls.py#wrong_tune_on_test_set -->
```python
candidates = [  # 8 x 3 x 3 = 72 tree configurations
    _tree(depth, leaf, fraction, seed).fit(features, labels)
    for depth, leaf, fraction in product(DEPTHS, LEAF_SIZES, FEATURE_FRACTIONS)
]
best = max(candidates, key=lambda model: _f1(model, test))  # the test set chooses the model ...
reported = _f1(best, test)  # ... and then grades its own choice: optimistic
```

<!-- snippet: src/db_ml_associate/ch10_tuning/pitfalls.py#correct_tune_with_cv_then_test -->
```python
search = GridSearchCV(
    _tree(3, 1, 1.0, seed),
    param_grid={
        "model__max_depth": DEPTHS,
        "model__min_samples_leaf": LEAF_SIZES,
        "model__max_features": FEATURE_FRACTIONS,
    },
    scoring="f1",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
).fit(train.drop(columns=CHURN_LABEL), train[CHURN_LABEL])
reported = _f1(search.best_estimator_, test)  # first and only look at the test set
```

Averaged over eight random splits of the churn data (72 tree configurations, a 500-row test set):

| Workflow | Reported F1 | F1 on new data | Optimism |
|---|---|---|---|
| Choose on the test set | 0.345 | 0.318 | +0.027 |
| Choose with 5-fold CV, test once | 0.328 | 0.320 | +0.008 |

The bias grows with the number of candidates and shrinks with the size of the test set.

**2. Maximising with a minimiser.** An objective that returns `{"loss": f1}` makes `fmin` search
for the *lowest* F1. On the churn data it confidently selects `C=0.0001` without class weights —
F1 **0.00** — instead of `C=1.0` with balanced weights (F1 0.32). Return `-f1`, or use Optuna with
`direction="maximize"`.

**3. Reading `hp.choice` results literally.** `fmin` returns `{"class_weight": 1}` — an index. Use
`space_eval`.

**4. Preprocessing outside the cross-validation loop.** Fitting an imputer or scaler on all training
rows before `GridSearchCV` leaks validation-fold statistics into training (chapter 9). Put
preprocessing in the pipeline you tune.

**5. `SparkTrials` with Spark ML models.** The objective already runs distributed jobs; nesting them
inside Spark tasks oversubscribes the cluster. Use `Trials`.

**6. Plain k-fold on time series.** Random folds let the model train on the future. Use time-ordered
splits.

## Summary

- Cross-validation trains *k* models per setting for a lower-variance estimate; a train–validation
  split trains one and is cheaper.
- Models trained by grid search with CV = product of the value counts × folds (exam convention).
- Grid search is exhaustive, random search samples a fixed budget, Bayesian search (TPE) learns from
  previous trials.
- Hyperopt: `fmin(fn, space, algo=tpe.suggest | rand.suggest, max_evals, trials)` minimises;
  `SparkTrials` parallelises single-node models. Optuna is today's recommended equivalent.

## Review questions

<!-- questions: ch10 -->

### Question 1

A data scientist tunes a gradient-boosted classifier with GridSearchCV using 4-fold cross-validation. The grid has learning_rate in [0.01, 0.1], max_depth in [3, 5, 7] and n_estimators in [100, 300]. Following the convention used by the certification exam, how many models are trained during the search?

- **A.** 12
- **B.** 48
- **C.** 49
- **D.** 7

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — 2 × 3 × 2 = 12 combinations, each trained once per fold: 12 × 4 = 48. The exam does not count the optional final refit.

Why the other options are wrong:

- **A**: Counts combinations but ignores the 4 folds.
- **C**: Adds the final refit on all training data, which the exam convention does not count.
- **D**: Adds the value counts (2 + 3 + 2) instead of multiplying them, and ignores the folds.

Objective S3.10 · [Revisit](10-validation-and-tuning.md#counting-models)

</details>

### Question 2

A Spark ML pipeline is tuned with TrainValidationSplit(trainRatio=0.75) over a ParamGridBuilder grid with 3 values of regParam and 3 values of maxIter. How many models are fitted during the search, excluding any final refit?

- **A.** 9
- **B.** 27
- **C.** 36
- **D.** 6

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — TrainValidationSplit evaluates each of the 3 × 3 = 9 parameter maps on a single split, so 9 models are fitted.

Why the other options are wrong:

- **B**: Would be the count for 3-fold cross-validation.
- **C**: Treats trainRatio=0.75 as 4 folds; TrainValidationSplit uses one split.
- **D**: Adds the value counts instead of multiplying them.

Objective S3.10 · [Revisit](10-validation-and-tuning.md#counting-models)

</details>

### Question 3

Compared with a single train-validation split, which TWO statements about k-fold cross-validation are correct?

- **A.** It gives a lower-variance estimate of model performance
- **B.** It requires roughly k times more training compute per hyperparameter setting
- **C.** It removes the need for a separate test set
- **D.** It always produces a better final model
- **E.** It is the preferred choice for time-ordered data

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — Every row is validated once and the k scores are averaged, which stabilises the estimate, at the cost of k fits per setting.

Why the other options are wrong:

- **C**: Selecting the best setting by its CV score makes that score optimistic; a test set is still needed.
- **D**: Cross-validation improves the estimate and the choice, not necessarily the model itself.
- **E**: Random folds leak future information in time series; time-ordered splits are needed.

Objective S3.08 · [Revisit](10-validation-and-tuning.md#cross-validation-vs-a-trainvalidation-split)

</details>

### Question 4

A data scientist wants Spark ML to choose regParam for a pipeline using 5-fold cross-validation and then return a model refitted with the best value. Which object should wrap the pipeline?

- **A.** CrossValidator with a ParamGridBuilder grid and an evaluator
- **B.** TrainValidationSplit with numFolds=5
- **C.** VectorAssembler with handleInvalid="keep"
- **D.** BinaryClassificationEvaluator with numFolds=5

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — CrossValidator is an estimator that evaluates every parameter map with k folds using the given evaluator and refits the best map on all the data.

Why the other options are wrong:

- **B**: TrainValidationSplit uses a single split and has no numFolds parameter.
- **C**: VectorAssembler builds feature vectors; it does not tune anything.
- **D**: An evaluator only computes a metric; it has no folds.

Objective S3.09 · [Revisit](10-validation-and-tuning.md#cross-validation-in-spark-ml)

</details>

### Question 5

An objective function passed to Hyperopt's fmin trains a model and returns {"loss": roc_auc, "status": STATUS_OK}. What will fmin find?

- **A.** The hyperparameters with the highest ROC AUC
- **B.** The hyperparameters with the lowest ROC AUC
- **C.** An error, because loss must be negative
- **D.** The hyperparameters with ROC AUC closest to 0.5

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — fmin minimises the loss, so returning a score to maximise finds the worst model; return -roc_auc instead.

Why the other options are wrong:

- **A**: That would require returning the negative AUC.
- **C**: Hyperopt accepts any numeric loss; the problem is the direction, not the sign.
- **D**: fmin minimises the value; it does not target 0.5.

Objective S3.05 · [Revisit](10-validation-and-tuning.md#hyperopt-fmin-on-the-exam)

</details>

### Question 6

After running best = fmin(objective, space, algo=tpe.suggest, max_evals=50) with space = {"penalty": hp.choice("penalty", ["l1", "l2"]), "C": hp.loguniform("C", -5, 2)}, a data scientist sees best == {"C": 0.82, "penalty": 1}. How should they obtain the actual penalty value?

- **A.** Use the value 1 directly as the penalty parameter
- **B.** Call hyperopt.space_eval(space, best)
- **C.** Rerun fmin with algo=rand.suggest
- **D.** Exponentiate the penalty value

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — For hp.choice, fmin returns the index of the chosen option; space_eval maps it back to "l2".

Why the other options are wrong:

- **A**: 1 is an index, not a valid penalty name.
- **C**: The search algorithm does not change the return format.
- **D**: Exponentiation applies to nothing here; hp.choice values are categorical.

Objective S3.05 · [Revisit](10-validation-and-tuning.md#hyperopt-fmin-on-the-exam)

</details>

### Question 7

A team has a budget of 30 training runs to tune 6 hyperparameters of an XGBoost model, and each run is expensive. Which search strategy is most likely to find a good configuration within the budget?

- **A.** Grid search over 3 values per hyperparameter
- **B.** Bayesian search such as TPE
- **C.** Exhaustive grid search over 2 values per hyperparameter
- **D.** Manually testing the default configuration 30 times

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — Bayesian search uses earlier results to propose promising candidates, so it typically finds good settings in fewer expensive trials.

Why the other options are wrong:

- **A**: 3^6 = 729 combinations far exceed the budget.
- **C**: 2^6 = 64 combinations still exceed the budget and cover each range poorly.
- **D**: Repeating one configuration explores nothing.

Objective S3.06 · [Revisit](10-validation-and-tuning.md#search-strategies-grid-random-and-bayesian)

</details>

### Question 8

In Hyperopt, which argument value switches fmin from Bayesian optimisation to random search?

- **A.** algo=hyperopt.rand.suggest
- **B.** trials=SparkTrials()
- **C.** max_evals=1
- **D.** space=hp.uniform("x", 0, 1)

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — The algo argument selects the search algorithm; rand.suggest is random search, tpe.suggest is TPE.

Why the other options are wrong:

- **B**: SparkTrials changes where trials run, not how they are chosen.
- **C**: max_evals only limits the number of trials.
- **D**: The space defines what can be sampled, not the search strategy.

Objective S3.06 · [Revisit](10-validation-and-tuning.md#hyperopt-fmin-on-the-exam)

</details>

### Question 9

A data scientist tunes a scikit-learn random forest on a 1 GB pandas DataFrame with Hyperopt on a classic cluster with 8 workers. Tuning runs one trial at a time on the driver. How can they speed it up most directly?

- **A.** Replace Trials() with SparkTrials(parallelism=8)
- **B.** Convert the model to Spark ML and keep Trials()
- **C.** Increase max_evals
- **D.** Use hp.choice for every hyperparameter

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — SparkTrials runs independent trials of a single-machine model as Spark tasks across the workers, so up to 8 trials run at the same time.

Why the other options are wrong:

- **B**: Rewriting the model is unnecessary; the bottleneck is sequential trials.
- **C**: More evaluations make tuning take longer.
- **D**: The type of distribution does not change parallelism.

Objective S3.07 · [Revisit](10-validation-and-tuning.md#parallelising-single-node-models)

</details>

### Question 10

Which TWO statements about parallel hyperparameter tuning on Databricks are correct?

- **A.** SparkTrials should be used with single-machine libraries such as scikit-learn
- **B.** When the objective trains a Spark ML model, use Trials rather than SparkTrials
- **C.** Higher SparkTrials parallelism always yields better hyperparameters
- **D.** SparkTrials parallelism can be set arbitrarily high, for example 1,000
- **E.** Parallel tuning requires converting the training data to a Spark DataFrame

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — SparkTrials distributes trials of single-node models; a Spark ML objective already distributes each fit, so Trials avoids oversubscribing the cluster.

Why the other options are wrong:

- **C**: Higher parallelism finishes sooner, but each Bayesian proposal learns from fewer completed trials.
- **D**: Parallelism is capped at 128 and by the cluster's concurrent task capacity.
- **E**: SparkTrials works with pandas data inside the objective function.

Objective S3.07 · [Revisit](10-validation-and-tuning.md#parallelising-single-node-models)

</details>
<!-- /questions -->
