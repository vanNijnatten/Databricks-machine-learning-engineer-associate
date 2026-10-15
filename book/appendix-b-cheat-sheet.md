# Appendix B — Cheat sheet

Use this in the last days before the exam. Every row links to the chapter section that explains it.

## Key API calls

| Task | Call | Where |
|---|---|---|
| Create a feature table | `FeatureEngineeringClient().create_table(name="cat.sch.t", primary_keys=["id"], df=df)` | [ch. 3](03-feature-engineering-uc.md#creating-a-feature-table) |
| Upsert features | `fe.write_table(name=..., df=df, mode="merge")` | [ch. 3](03-feature-engineering-uc.md#writing-data-to-a-feature-table) |
| Training set from feature lookups | `fe.create_training_set(df=labels, feature_lookups=[FeatureLookup(table_name=..., lookup_key="id")], label="y")` | [ch. 3](03-feature-engineering-uc.md#training-with-features-from-a-feature-table) |
| Log model with its lookups | `fe.log_model(model=m, artifact_path="model", flavor=mlflow.sklearn, training_set=ts)` | [ch. 3](03-feature-engineering-uc.md#training-with-features-from-a-feature-table) |
| Batch score with lookups (keys only) | `fe.score_batch(model_uri=..., df=keys)` | [ch. 3](03-feature-engineering-uc.md#scoring-with-features-from-a-feature-table) |
| Publish to online store | `fe.publish_table(online_store=..., source_table_name=..., online_table_name=...)` | [ch. 3](03-feature-engineering-uc.md#offline-vs-online-feature-tables) |
| Log params / metrics / files | `mlflow.log_params({...})`, `mlflow.log_metric("f1", v, step=i)`, `mlflow.log_artifact(path)`, `mlflow.log_figure(fig, "x.png")` | [ch. 4](04-mlflow-tracking.md#logging-manually) |
| Log a model (MLflow 3) | `mlflow.sklearn.log_model(m, name="model", signature=infer_signature(X, y_pred), input_example=X[:3])` | [ch. 4](04-mlflow-tracking.md#logging-manually) |
| Autologging | `mlflow.autolog()` or `mlflow.sklearn.autolog()` | [ch. 4](04-mlflow-tracking.md#autologging) |
| Best run | `MlflowClient().search_runs([exp_id], order_by=["metrics.f1 DESC"], max_results=1)[0]` (ASC for losses) | [ch. 4](04-mlflow-tracking.md#finding-the-best-run-with-the-client-api) |
| Register in Unity Catalog | `mlflow.register_model("runs:/<run_id>/model", "cat.sch.model")` | [ch. 5](05-model-lifecycle-uc.md#registering-a-model-in-unity-catalog) |
| Model tag set / remove | `client.set_registered_model_tag(name, k, v)` / `client.delete_registered_model_tag(name, k)` | [ch. 5](05-model-lifecycle-uc.md#tags) |
| Version tag set / remove | `client.set_model_version_tag(name, ver, k, v)` / `client.delete_model_version_tag(name, ver, k)` | [ch. 5](05-model-lifecycle-uc.md#tags) |
| Promote challenger | `client.set_registered_model_alias(name, "champion", ver)`; load `models:/cat.sch.model@champion` | [ch. 5](05-model-lifecycle-uc.md#aliases-champion-and-challenger) |
| Summary statistics | `df.summary()` (with quartiles), `df.describe()`, `dbutils.data.summarize(df, precise=False)` | [ch. 6](06-exploring-data.md#summary-statistics) |
| Impute | `Imputer(strategy="mean"/"median"/"mode", inputCols=..., outputCols=...).fit(train)` | [ch. 7](07-cleaning-and-transforming.md#imputing-missing-values) |
| One-hot encode | `StringIndexer` → `OneHotEncoder` (dropLast: k−1 slots) | [ch. 7](07-cleaning-and-transforming.md#one-hot-encoding) |
| IQR bounds | `df.approxQuantile(col, [0.25, 0.75], 0.0)` | [ch. 7](07-cleaning-and-transforming.md#iqr-tukeys-fences-filter) |
| Pipeline | `Pipeline(stages=[...]).fit(train)` → `PipelineModel.transform(test)` | [ch. 9](09-pipelines.md#building-a-spark-ml-training-pipeline) |
| Cross-validation (Spark) | `CrossValidator(estimator, estimatorParamMaps=ParamGridBuilder()...build(), evaluator, numFolds=k)` | [ch. 10](10-validation-and-tuning.md#cross-validation-in-spark-ml) |
| Hyperopt | `fmin(fn, space, algo=tpe.suggest, max_evals=n, trials=SparkTrials(parallelism=p))`; return `{"loss": -score, "status": STATUS_OK}` | [ch. 10](10-validation-and-tuning.md#hyperopt-fmin-on-the-exam) |
| Optuna | `optuna.create_study(direction="maximize").optimize(objective, n_trials=n, n_jobs=p)` | [ch. 10](10-validation-and-tuning.md#parallelising-single-node-models) |
| Batch inference (pandas) | `mlflow.pyfunc.load_model(uri).predict(pdf)` — load once | [ch. 12](12-batch-and-streaming-inference.md#batch-inference-with-pandas) |
| Batch / streaming inference (Spark) | `udf = mlflow.pyfunc.spark_udf(spark, uri)`; `df.withColumn("p", udf(F.struct(*cols)))` | [ch. 12](12-batch-and-streaming-inference.md#batch-inference-at-scale-with-a-spark-udf) |
| Serving endpoint | `served_entities=[ServedEntityInput(entity_name, entity_version, workload_size, scale_to_zero_enabled)]`, `traffic_config` routes sum to 100 | [ch. 13](13-real-time-serving.md#creating-an-endpoint) |
| Query an endpoint | POST `/serving-endpoints/<name>/invocations` with `{"dataframe_split": {"columns": [...], "data": [...]}}` | [ch. 13](13-real-time-serving.md#querying-an-endpoint) |

## Decision tables

| Situation | Choose | Where |
|---|---|---|
| Numeric, roughly symmetric, missing values | Mean imputation | [ch. 7](07-cleaning-and-transforming.md#imputing-missing-values) |
| Numeric, skewed or with outliers | Median imputation | [ch. 7](07-cleaning-and-transforming.md#imputing-missing-values) |
| Categorical missing values | Mode imputation | [ch. 7](07-cleaning-and-transforming.md#imputing-missing-values) |
| Outliers, roughly normal data | Remove beyond mean ± 3 standard deviations | [ch. 7](07-cleaning-and-transforming.md#standard-deviation-z-score-filter) |
| Outliers, skewed data | IQR fences | [ch. 7](07-cleaning-and-transforming.md#iqr-tukeys-fences-filter) |
| Low-cardinality category, linear model | One-hot encode | [ch. 7](07-cleaning-and-transforming.md#one-hot-encoding) |
| High-cardinality category or tree model | Index or target-encode, not one-hot | [ch. 7](07-cleaning-and-transforming.md#when-is-one-hot-encoding-inappropriate) |
| Positive, right-skewed values | Log transform (`log1p` with zeros) | [ch. 7](07-cleaning-and-transforming.md#log-transformation) |
| Rare positive class | Class weights, resample training split only, tune threshold, use F1/recall | [ch. 8](08-ml-foundations.md#class-imbalance) |
| Train ≫ validation score | High variance: simplify, regularise, more data | [ch. 8](08-ml-foundations.md#bias-variance-and-model-complexity) |
| Both scores poor | High bias: more complex model, better features | [ch. 8](08-ml-foundations.md#bias-variance-and-model-complexity) |
| Small data / reliable estimate | k-fold cross-validation | [ch. 10](10-validation-and-tuning.md#cross-validation-vs-a-trainvalidation-split) |
| Huge data / expensive training | Train-validation split | [ch. 10](10-validation-and-tuning.md#cross-validation-vs-a-trainvalidation-split) |
| Few, cheap combinations | Grid search | [ch. 10](10-validation-and-tuning.md#search-strategies-grid-random-and-bayesian) |
| Large space, small budget | Random or Bayesian (TPE) search | [ch. 10](10-validation-and-tuning.md#search-strategies-grid-random-and-bayesian) |
| Missing a positive is costly | Recall | [ch. 11](11-evaluation.md#choosing-the-right-metric) |
| False alarms are costly | Precision | [ch. 11](11-evaluation.md#choosing-the-right-metric) |
| Imbalanced, both errors matter | F1 | [ch. 11](11-evaluation.md#choosing-the-right-metric) |
| Probabilities used directly | Log loss | [ch. 11](11-evaluation.md#choosing-the-right-metric) |
| Large errors especially bad / outliers should not dominate | RMSE / MAE | [ch. 11](11-evaluation.md#choosing-the-right-metric) |
| Scheduled scoring of many rows | Batch inference | [ch. 12](12-batch-and-streaming-inference.md#choosing-a-serving-approach) |
| Continuous events, seconds-minutes latency | Streaming: Delta Live Tables pipeline with a Spark UDF | [ch. 12](12-batch-and-streaming-inference.md#streaming-inference-with-lakeflow-pipelines) |
| Low-latency predictions with fresh features | Online feature store + Model Serving endpoint | [ch. 3](03-feature-engineering-uc.md#offline-vs-online-feature-tables), [ch. 13](13-real-time-serving.md#creating-an-endpoint) |
| Custom pre/post-processing behind an endpoint | `mlflow.pyfunc.PythonModel` | [ch. 13](13-real-time-serving.md#custom-models) |
| A/B test or canary | Two served models, traffic split | [ch. 13](13-real-time-serving.md#splitting-traffic-between-models) |
| Default promotion strategy | Deploy code | [ch. 1](01-databricks-ml-platform.md#promoting-code-vs-promoting-models) |
| Very expensive one-off training | Deploy models | [ch. 1](01-databricks-ml-platform.md#promoting-code-vs-promoting-models) |
| Fast baseline with editable code | AutoML | [ch. 2](02-automl.md#advantages-for-the-model-development-process) |
| Full control, custom features or algorithms | Manual training | [ch. 2](02-automl.md#choosing-an-approach) |

## Formulas

| Quantity | Formula | Where |
|---|---|---|
| Precision | TP / (TP + FP) | [ch. 11](11-evaluation.md#classification-metrics) |
| Recall | TP / (TP + FN) | [ch. 11](11-evaluation.md#classification-metrics) |
| F1 | 2 · precision · recall / (precision + recall) | [ch. 11](11-evaluation.md#classification-metrics) |
| Accuracy | (TP + TN) / all | [ch. 11](11-evaluation.md#classification-metrics) |
| RMSE | √(mean((y − ŷ)²)) | [ch. 11](11-evaluation.md#regression-metrics) |
| MAE | mean(\|y − ŷ\|) | [ch. 11](11-evaluation.md#regression-metrics) |
| R² | 1 − SS_residual / SS_total | [ch. 11](11-evaluation.md#regression-metrics) |
| Back-transform | ŷ = exp(log-prediction), or `expm1` after `log1p` | [ch. 11](11-evaluation.md#log-transformed-targets-exponentiate-first) |
| z-score | (x − mean) / std; outlier if \|z\| > 3 | [ch. 7](07-cleaning-and-transforming.md#standard-deviation-z-score-filter) |
| IQR fences | [Q1 − 1.5 · IQR, Q3 + 1.5 · IQR], IQR = Q3 − Q1 | [ch. 7](07-cleaning-and-transforming.md#iqr-tukeys-fences-filter) |
| Models trained | grid combinations × folds (exam convention; +1 refit is not counted) | [ch. 10](10-validation-and-tuning.md#counting-the-models-a-search-trains) |
| Balanced class weight | n_samples / (n_classes · n_in_class) | [ch. 8](08-ml-foundations.md#cost-sensitive-learning-class-weights) |

## Exam vs current platform

| On the exam | Today | Where |
|---|---|---|
| Hyperopt `fmin` / `SparkTrials` | Deprecated, not preinstalled; Optuna or Ray Tune | [ch. 10](10-validation-and-tuning.md#hyperopt-fmin-on-the-exam) |
| Delta Live Tables | Lakeflow pipelines (`from pyspark import pipelines as dp`) | [ch. 12](12-batch-and-streaming-inference.md#streaming-inference-with-lakeflow-pipelines) |
| Online tables | Online Feature Stores (Lakebase), `publish_table` | [ch. 3](03-feature-engineering-uc.md#offline-vs-online-feature-tables) |
| Workspace registry stages | Unity Catalog aliases and tags | [ch. 5](05-model-lifecycle-uc.md#aliases-champion-and-challenger) |
| `FeatureStoreClient` | `FeatureEngineeringClient` | [ch. 3](03-feature-engineering-uc.md#feature-tables-in-unity-catalog-vs-the-workspace-feature-store) |
| AutoML built into the ML runtime | Separate package from 18.0 ML; Dedicated compute | [ch. 2](02-automl.md#running-automl) |
| `log_model(artifact_path=...)` | MLflow 3 `log_model(name=...)`, LoggedModel | [ch. 4](04-mlflow-tracking.md#experiments-runs-and-models) |
| MLlib automatic MLflow tracking | `mlflow.pyspark.ml.autolog()` | [ch. 10](10-validation-and-tuning.md#cross-validation-in-spark-ml) |

Full table with the expected exam answers: [Appendix A](appendix-a-exam-vs-platform.md).
