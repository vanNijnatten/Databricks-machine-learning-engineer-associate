# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 10 tutorial: Validation and hyperparameter tuning
# MAGIC **Objectives**: S3.05–S3.10 · **Runtime**: 18 LTS ML · **Requires**: serverless or classic; `SparkTrials` needs classic Dedicated compute · **Free Edition**: partial
# MAGIC **Estimated time**: 75 min · **Clean-up**: none
# MAGIC
# MAGIC Cross-validate Spark ML and scikit-learn models, count the models a search trains, tune with grid, random
# MAGIC and Bayesian search, use Hyperopt `fmin` (exam) and Optuna (current recommendation), and parallelise
# MAGIC single-node tuning. On serverless (Free Edition) `SparkTrials` is skipped automatically.
# MAGIC Run `00_setup` first.

# COMMAND ----------

# MAGIC %pip install hyperopt==0.2.7

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

from db_ml_associate.common import CHURN_LABEL, DEFAULT_SEED, split_frame

churn = spark.table(uc_name("customer_churn"))
train, test = split_frame(churn.toPandas(), CHURN_LABEL)
mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').first()[0]}/ml-associate-ch10")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Cross-validation and counting models (S3.08, S3.09, S3.10)
# MAGIC
# MAGIC A `CrossValidator` with a 2 × 2 grid and 3 folds fits 4 × 3 = 12 models (plus one refit of the best
# MAGIC parameters on all training data, which the exam does not count). `TrainValidationSplit` fits 4.

# COMMAND ----------

from db_ml_associate.ch10_tuning.cross_validation import count_models, spark_cross_validator

print(
    "exam count for {regParam: 2, elasticNetParam: 2}, 3 folds:",
    count_models({"regParam": [0.0, 0.1], "elasticNetParam": [0.0, 1.0]}, folds=3),
)
spark_train, spark_test = churn.randomSplit([0.8, 0.2], seed=DEFAULT_SEED)
cv_model = spark_cross_validator(folds=3).fit(spark_train)  # mlflow.pyspark.ml.autolog logs this when enabled
for params, metric in zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics, strict=True):
    print({param.name: value for param, value in params.items()}, "mean AUC:", round(metric, 4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Grid, random and Bayesian search (S3.06)

# COMMAND ----------

from db_ml_associate.ch10_tuning.cross_validation import grid_search_cv, random_search_cv

grid = grid_search_cv(train)
print("GridSearchCV best:", grid.best_params_, "fits:", len(grid.cv_results_["params"]) * grid.n_splits_)
random_search = random_search_cv(train)
print("RandomizedSearchCV best:", random_search.best_params_)

# COMMAND ----------

from db_ml_associate.ch10_tuning.optuna_tuning import make_sampler, tune_with_optuna

for kind in ("grid", "random", "bayesian"):
    print(kind, tune_with_optuna(train, n_trials=10, sampler=make_sampler(kind)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Hyperopt `fmin` (S3.05)
# MAGIC
# MAGIC > **Platform change:** Hyperopt is not preinstalled after 16.4 LTS ML (installed at the top of this notebook)
# MAGIC > and is no longer maintained. The exam still asks about `fmin`; use Optuna for new work.

# COMMAND ----------

from db_ml_associate.ch10_tuning.hyperopt_tuning import tune_with_hyperopt

best = tune_with_hyperopt(train, max_evals=15)
print("best parameters:", best)

# COMMAND ----------

# MAGIC %md
# MAGIC > **⚠️ Wrong approach:** `fmin` *minimises*. Returning the F1 score as the loss selects the worst model.

# COMMAND ----------

from db_ml_associate.ch10_tuning import pitfalls

print("returning +F1:", pitfalls.wrong_fmin_maximise_metric(train))
print("returning -F1:", pitfalls.correct_return_negative_metric(train))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Parallelising single-node tuning (S3.07)
# MAGIC
# MAGIC `SparkTrials` runs each trial as a Spark task — it needs a `sparkContext`, i.e. classic compute. On serverless
# MAGIC this cell falls back to Optuna with parallel threads (`n_jobs`), which also works on Free Edition.

# COMMAND ----------

if has_spark_context():
    from db_ml_associate.ch10_tuning.hyperopt_tuning import tune_with_spark_trials

    with mlflow.start_run(run_name="hyperopt SparkTrials"):
        best, spark_trials = tune_with_spark_trials(train, max_evals=16, parallelism=4)
    print("SparkTrials best:", best, "trials:", len(spark_trials.trials))
else:
    print("No sparkContext (serverless / Spark Connect): SparkTrials is unavailable. Using Optuna with n_jobs=4.")
    print("Optuna best:", tune_with_optuna(train, n_trials=16, n_jobs=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Open the **Experiments** page: with `SparkTrials` each trial appears as a run. Compare the number of runs
# MAGIC with `max_evals`.
# MAGIC
# MAGIC **Self-check**: a grid of 3 × 2 × 3 values with 5-fold CV — how many models? (90.)
