---
chapter: 3
title: Feature engineering in Unity Catalog
objectives: [S1.05, S1.06, S1.07, S1.08, S1.09, S1.10]
prerequisites: [1, 4, 9]
reading_minutes: 90
references:
  [feature-store-overview, feature-tables-uc, train-with-feature-store, online-feature-store, workspace-feature-store, unity-catalog, free-edition-limitations]
---

# 3. Feature engineering in Unity Catalog

> **Exam mapping**: Section 1 — Databricks Machine Learning · objectives S1.05–S1.10 ·
> **Runtime**: Databricks Runtime 18 LTS ML (`databricks-feature-engineering` 0.13.0.1) ·
> **Read after**: chapters 1, 4 and 9.

A **feature** is an input to a model — "support calls per year", "is on a month-to-month contract".
Several models need the same features, and the values used at training time must equal the values
used at prediction time. **Feature engineering in Unity Catalog** stores features in governed
**feature tables**, joins them into training sets, remembers which features each model used, and
looks them up again when the model scores — offline in batch or online in real time.

## Learning objectives

- Explain the benefits of feature tables in Unity Catalog over the (deprecated) workspace-level
  feature store.
- Create a feature table with `FeatureEngineeringClient` or SQL and write data to it.
- Train a model with features from a feature table and score it with automatic feature lookup.
- Describe the differences between offline and online feature tables.

## Feature tables in Unity Catalog vs the workspace feature store

<!-- objectives: S1.05 -->

In Unity Catalog, **any Delta table with a primary key is a feature table**
([feature tables in Unity Catalog](https://docs.databricks.com/aws/en/machine-learning/feature-store/uc/feature-tables-uc)).
Because it lives in the account-level Unity Catalog metastore rather than in one workspace, it gets
([Databricks Feature Store](https://docs.databricks.com/aws/en/machine-learning/feature-store/)):

| Benefit | Unity Catalog feature tables | Workspace Feature Store (legacy) |
|---|---|---|
| **Sharing and discovery across workspaces** | Yes — every workspace attached to the metastore can find and use them in Catalog Explorer | Only inside the workspace that created them |
| **Governance** | Unity Catalog privileges (`GRANT SELECT …`), the same as all data | Separate, workspace-local permissions |
| **Lineage** | Tracked from source tables to features, models and endpoints | Limited |
| **Point-in-time lookups** | Yes, with a time-series key | — |
| **Status** | Recommended | **Deprecated**; only in workspaces created before 19 August 2024 ([Workspace Feature Store](https://docs.databricks.com/aws/en/machine-learning/feature-store/workspace-feature-store/)) |
| **Client** | `FeatureEngineeringClient` (`databricks.feature_engineering`) | `FeatureStoreClient` (`databricks.feature_store`) |

> **Exam tip:** "account level in Unity Catalog vs workspace level" → cross-workspace **sharing and
> discovery**, **centralised governance/access control** and **lineage**.

> **Beyond the exam:** Databricks now also offers **Feature Views** — features declared as Unity
> Catalog objects whose pipelines Databricks manages. Feature tables, covered here, remain the
> approach the exam asks about.

## Creating a feature table

<!-- objectives: S1.06 -->

First compute the features — one row per entity, keyed by `customer_id`, **without the label**:

<!-- snippet: src/db_ml_associate/ch03_features/features.py#compute_features -->
```python
return customers.select(
    CHURN_KEY,  # primary key of the feature table
    "tenure_months",
    "monthly_charges",
    "data_usage_gb",
    "support_calls",
    (F.col("support_calls") / F.greatest(F.col("tenure_months") / 12, F.lit(1.0))).alias("support_calls_per_year"),
    (F.col("contract_type") == "month-to-month").cast("int").alias("is_month_to_month"),
    "payment_method",
)
```

Then create the table with the client (`fe = FeatureEngineeringClient()`, preinstalled in the ML
runtime):

<!-- snippet: src/db_ml_associate/ch03_features/workspace.py#create_table -->
```python
return fe.create_table(
    name=name,  # catalog.schema.table in Unity Catalog
    primary_keys=[CHURN_KEY],  # required: the key every lookup uses
    df=features,  # schema and initial data; or schema=... for an empty table
    description="Customer behaviour features for churn models",
)
```

`create_table` registers a Delta table in Unity Catalog with a primary key. Pass `df` to infer the
schema and write the data at once, or `schema` to create an empty table. Time-series features take
a `timeseries_column` (for point-in-time lookups).

The SQL alternative — any Delta table with a `PRIMARY KEY` constraint is a feature table:

<!-- snippet: src/db_ml_associate/ch03_features/sql/create_feature_table.sql#create_table_sql -->
```sql
CREATE TABLE ml_associate.tutorial.customer_features (
  customer_id BIGINT NOT NULL,          -- primary key columns must be NOT NULL
  tenure_months BIGINT,
  monthly_charges DOUBLE,
  support_calls_per_year DOUBLE,
  CONSTRAINT customer_features_pk PRIMARY KEY (customer_id)
);
```

or add the key to an existing table (only the owner can):

<!-- snippet: src/db_ml_associate/ch03_features/sql/add_primary_key.sql#add_primary_key_sql -->
```sql
ALTER TABLE ml_associate.tutorial.customer_features ALTER COLUMN customer_id SET NOT NULL;
ALTER TABLE ml_associate.tutorial.customer_features
  ADD CONSTRAINT customer_features_pk PRIMARY KEY (customer_id);
```

> **Exam tip:** the official sample question's answer is "use the `create_table` method of the
> `FeatureEngineeringClient`". Clauses like `AS FEATURE STORE` or `SET AS FEATURE STORE` do not
> exist, and `register_table` on `FeatureStoreClient` is the legacy path.

## Writing data to a feature table

<!-- objectives: S1.07 -->

<!-- snippet: src/db_ml_associate/ch03_features/workspace.py#write_table -->
```python
fe.write_table(name=name, df=features, mode="merge")  # upsert by primary key
```

`write_table` **merges** by primary key — rows with new keys are inserted, rows with existing keys are
updated, and existing keys missing from `df` are left unchanged. Pass a streaming DataFrame to keep
a table up to date continuously. Because a feature table is a Delta table, plain `MERGE INTO` SQL
works too. `fe.read_table(name=...)` returns the current contents.

## Training with features from a feature table

<!-- objectives: S1.08 -->

The **training set** starts from a DataFrame of keys and labels; `FeatureLookup`s say which features
to join on which key
([Train models with feature tables](https://docs.databricks.com/aws/en/machine-learning/feature-store/train-models-with-feature-store)):

<!-- snippet: src/db_ml_associate/ch03_features/workspace.py#training_set -->
```python
lookup = FeatureLookup(
    table_name=table_name,
    lookup_key=CHURN_KEY,  # column in `labels` that matches the table's primary key
    feature_names=None,  # None = every feature column of the table
)
training_set = fe.create_training_set(
    df=labels,  # customer_id + churned
    feature_lookups=[lookup],
    label=CHURN_LABEL,
    exclude_columns=[CHURN_KEY],  # keys are not features
)
training_df = training_set.load_df()  # Spark DataFrame: features + label → train any model on it
```

- `lookup_key` must match the feature table's primary key (same type and order).
- `feature_names` selects features (`None` = all); `rename_outputs` avoids name clashes;
  `timestamp_lookup_key` enables point-in-time joins for time-series tables; `default_values` fills
  missing features.
- Columns of the input DataFrame that are not keys (e.g. a request-time `browser` column) are kept
  as features unless excluded.

Train any model on `training_set.load_df()` (convert to pandas for scikit-learn). Then log it **with
the training set**, so the model remembers its feature lookups:

<!-- snippet: src/db_ml_associate/ch03_features/workspace.py#log_model -->
```python
fe.log_model(
    model=model,
    artifact_path="model",  # the feature engineering client still uses artifact_path
    flavor=mlflow.sklearn,
    training_set=training_set,  # records which features to look up at inference
    registered_model_name=registered_model_name,  # e.g. "ml_associate.tutorial.churn_model"
    infer_input_example=True,
)
```

> **Platform change:** the feature engineering client's `log_model` still uses `artifact_path`, while
> plain MLflow 3 uses `name` (chapter 4).

## Scoring with features from a feature table

<!-- objectives: S1.09 -->

A model logged with `fe.log_model` needs only the **keys** at scoring time — the client looks the
features up:

<!-- snippet: src/db_ml_associate/ch03_features/workspace.py#score_batch -->
```python
predictions = fe.score_batch(model_uri=model_uri, df=keys)  # adds a `prediction` column
```

`score_batch` returns the input DataFrame plus the looked-up features and a `prediction` column.
Columns in `df` with the same name as a feature *override* the stored value. Because the same lookup
logic runs at training and scoring time, **training/serving skew** is avoided. The same model can be
served on Model Serving with automatic lookups from an online store (chapter 13).

## Offline vs online feature tables

<!-- objectives: S1.10 -->

| | Offline feature table | Online feature table |
|---|---|---|
| Storage | Delta table in Unity Catalog | Low-latency store: Databricks **Online Feature Store** (Lakebase) |
| Holds | Full history, all rows | Latest values for fast key lookups |
| Latency | Seconds to minutes (Spark) | Milliseconds |
| Used for | Training, batch scoring, analysis | Real-time inference (Model Serving), feature serving endpoints |
| Written by | `write_table` / Delta writes | **Published** from the offline table |

The offline table stays the source of truth; you **publish** it to an online store
([Online Feature Stores](https://docs.databricks.com/aws/en/machine-learning/feature-store/online-feature-store)):

<!-- snippet: src/db_ml_associate/ch03_features/workspace.py#online_store -->
```python
return fe.create_online_store(name=name, capacity=capacity)  # CU_1, CU_2, CU_4 or CU_8
```

<!-- snippet: src/db_ml_associate/ch03_features/workspace.py#publish_table -->
```python
# Requires a primary key and Change Data Feed on the source table:
# ALTER TABLE <source> SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
return fe.publish_table(
    online_store=online_store,
    source_table_name=source_table_name,
    online_table_name=online_table_name,
    streaming=False,  # False → TRIGGERED sync; True → CONTINUOUS sync
)
```

> **Platform change:** the documentation shows `publish_table(..., publish_mode="TRIGGERED" |
> "CONTINUOUS" | "SNAPSHOT")`. The client preinstalled in Runtime 18 LTS ML (0.13.0.1) has no
> `publish_mode`: `streaming=False` gives a triggered sync and `streaming=True` a continuous one, and
> publishing fails unless Change Data Feed is enabled on the source table. Older "online tables" are
> replaced by Online Feature Stores.

> **Requires:** an online store is a paid, always-on resource (not available on Free Edition, which
> does not support online tables). Delete the online table and the store at the end of the tutorial
> (`notebooks/99_cleanup.py`).

> **Exam tip:** "training and batch scoring" → offline feature table. "Low-latency lookups for a
> real-time endpoint" → online feature table/store, published from the offline table.

## Choosing an approach

| Need | Use |
|---|---|
| Create a feature table from Python | `fe.create_table(name, primary_keys, df \| schema)` |
| Make an existing Delta table a feature table | SQL `ALTER TABLE … ADD CONSTRAINT … PRIMARY KEY` |
| Upsert new feature values | `fe.write_table(..., mode="merge")` |
| Build training data | `FeatureLookup` + `fe.create_training_set` + `load_df()` |
| Score in batch without re-implementing feature logic | `fe.log_model` + `fe.score_batch` |
| Serve features with millisecond latency | Online Feature Store + `publish_table` |
| Features that change over time, no leakage | Time-series feature table + `timestamp_lookup_key` |

## Pitfalls

**1. Null or duplicate primary keys.** Primary key columns must be `NOT NULL` and unique. Check
before writing:

<!-- snippet: src/db_ml_associate/ch03_features/pitfalls.py#primary_key_problems -->
```python
problems = []
nulls = frame[list(keys)].isna().any(axis=1).sum()
if nulls:
    problems.append(f"{nulls} rows have a null primary key")
duplicates = frame.duplicated(subset=list(keys)).sum()
if duplicates:
    problems.append(f"{duplicates} duplicate primary keys")
```

**2. Using the legacy client.**

> **⚠️ Wrong approach:** `from databricks.feature_store import FeatureStoreClient` — deprecated
> workspace feature store. Use `FeatureEngineeringClient` with Unity Catalog.

**3. Putting the label in the feature table.** Labels belong in the DataFrame you pass to
`create_training_set`; a label column in a feature table leaks into training.

**4. Changing `load_df()` output before training.** Transformations applied to the training
DataFrame outside the logged model are not repeated by `score_batch`. Put them in the model
pipeline (chapter 9) or in the feature computation.

**5. Logging with plain `mlflow.<flavor>.log_model`.** The model then knows nothing about feature
lookups and `score_batch` cannot fetch features. Use `fe.log_model(..., training_set=...)`.

**6. Publishing without Change Data Feed.** Enable it on the source table first.

## Summary

- Unity Catalog feature tables are Delta tables with a primary key, shared, governed and lineage-
  tracked across workspaces; the workspace feature store is deprecated.
- `create_table` creates, `write_table(mode="merge")` upserts, `FeatureLookup` +
  `create_training_set` build training data, `fe.log_model` packages the lookups and `score_batch`
  scores with automatic lookups.
- Offline tables serve training and batch scoring; online stores, published from them, serve
  real-time lookups.

## Review questions

<!-- questions: ch03 -->

### Question 1

A company with five Databricks workspaces wants data scientists in all of them to reuse the same customer features. Which TWO benefits do feature tables in Unity Catalog offer over workspace-level feature tables?

- **A.** Features can be discovered and shared across all workspaces attached to the metastore
- **B.** Access is governed with Unity Catalog privileges, with lineage from source data to models
- **C.** Feature tables no longer need a primary key
- **D.** Features are automatically computed without any pipeline code
- **E.** Models trained on them no longer need to be logged to MLflow

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — Unity Catalog feature tables are governed centrally and visible across workspaces, with lineage tracking; the workspace feature store is deprecated and workspace-local.

Why the other options are wrong:

- **C**: A primary key is exactly what makes a Delta table a feature table.
- **D**: You still own the code or pipeline that computes the features.
- **E**: Logging with fe.log_model is how models remember their feature lookups.

Objective S1.05 · [Revisit](03-feature-engineering-uc.md#feature-tables-in-unity-catalog-vs-the-workspace-feature-store)

</details>

### Question 2

A data scientist has a Spark DataFrame features_df with one row per store_id and wants it governed as a feature table in Unity Catalog. Which call creates the table?

- **A.** fe.create_table(name="main.retail.store_features", primary_keys=["store_id"], df=features_df)
- **B.** features_df.write.format("featurestore").save("store_features")
- **C.** spark.sql("CREATE FEATURE TABLE store_features AS SELECT * FROM features_df")
- **D.** mlflow.log_table(features_df, "store_features")

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — FeatureEngineeringClient.create_table registers a Delta table with a primary key in Unity Catalog; passing df infers the schema and writes the data.

Why the other options are wrong:

- **B**: There is no "featurestore" data source format.
- **C**: CREATE FEATURE TABLE is not valid SQL; a PRIMARY KEY constraint is the SQL route.
- **D**: log_table logs an MLflow artifact; it does not create a feature table.

Objective S1.06 · [Revisit](03-feature-engineering-uc.md#creating-a-feature-table)

</details>

### Question 3

An existing Unity Catalog Delta table main.retail.store_metrics has a non-null, unique column store_id. How can it be turned into a feature table using SQL only?

- **A.** ALTER TABLE main.retail.store_metrics ADD CONSTRAINT store_pk PRIMARY KEY (store_id)
- **B.** ALTER TABLE main.retail.store_metrics SET AS FEATURE STORE
- **C.** CREATE FEATURE STORE main.retail.store_metrics
- **D.** OPTIMIZE main.retail.store_metrics ZORDER BY (store_id)

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Any Unity Catalog Delta table with a primary key constraint is a feature table.

Why the other options are wrong:

- **B**: SET AS FEATURE STORE is not a valid clause.
- **C**: There is no CREATE FEATURE STORE statement.
- **D**: OPTIMIZE ZORDER changes data layout; it adds no primary key.

Objective S1.06 · [Revisit](03-feature-engineering-uc.md#creating-a-feature-table)

</details>

### Question 4

A nightly job recomputes features for customers whose data changed and calls fe.write_table(name=table, df=changed_df, mode="merge"). What happens to customers that are in the table but not in changed_df?

- **A.** Their rows are left unchanged
- **B.** Their rows are deleted
- **C.** Their feature values are set to null
- **D.** The write fails because the keys do not match

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — merge upserts by primary key: new keys are inserted, existing keys in df are updated, and keys absent from df are not touched.

Why the other options are wrong:

- **B**: merge never deletes rows.
- **C**: Values are only changed for keys present in df.
- **D**: Partial updates are exactly what merge supports.

Objective S1.07 · [Revisit](03-feature-engineering-uc.md#writing-data-to-a-feature-table)

</details>

### Question 5

A labels DataFrame contains customer_id and churned. Which code builds a training DataFrame that joins the features of main.ml.customer_features to these labels?

- **A.** fe.create_training_set(df=labels, feature_lookups=[FeatureLookup(table_name="main.ml.customer_features", lookup_key="customer_id")], label="churned").load_df()
- **B.** fe.read_table(name="main.ml.customer_features").join(labels).toPandas()
- **C.** fe.score_batch(model_uri="main.ml.customer_features", df=labels)
- **D.** FeatureLookup(labels, "churned").load_df()

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — create_training_set joins the looked-up features to the label DataFrame and records the lookups so that fe.log_model can package them with the model.

Why the other options are wrong:

- **B**: A manual join works for training but loses the recorded lookups needed for automatic scoring, and an unconditioned join() is a cross join.
- **C**: score_batch scores an existing model; it does not build training data.
- **D**: FeatureLookup describes a lookup; it has no load_df method.

Objective S1.08 · [Revisit](03-feature-engineering-uc.md#training-with-features-from-a-feature-table)

</details>

### Question 6

A model trained on a training set from create_training_set is logged with mlflow.sklearn.log_model. Later, fe.score_batch fails because the feature columns are missing from the input DataFrame, which contains only customer_id. What should have been done?

- **A.** Log the model with fe.log_model(..., training_set=training_set, flavor=mlflow.sklearn)
- **B.** Register the model in the workspace registry
- **C.** Call score_batch with result_type="string"
- **D.** Add customer_id to exclude_columns at scoring time

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Only fe.log_model stores the feature lookups with the model, which score_batch needs to fetch features from the feature table.

Why the other options are wrong:

- **B**: The registry choice does not add feature lookup metadata.
- **C**: result_type changes the output type, not feature retrieval.
- **D**: exclude_columns applies when building training sets, and keys are needed for lookups.

Objective S1.08 · [Revisit](03-feature-engineering-uc.md#pitfalls)

</details>

### Question 7

A churn model was logged with fe.log_model and registered as main.ml.churn. A batch job has a DataFrame with only customer_id for today's active customers. How should it score them?

- **A.** fe.score_batch(model_uri="models:/main.ml.churn@champion", df=active_customers)
- **B.** Join all features manually and call mlflow.pyfunc.load_model(...).predict(...)
- **C.** Publish the model to an online store
- **D.** Call fe.create_training_set with the active customers

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — score_batch looks up the model's features by key, adds them to the DataFrame and returns predictions, using the same logic as at training time.

Why the other options are wrong:

- **B**: Manual joins duplicate feature logic and risk training/serving skew.
- **C**: Online stores hold features, not models, and target real-time serving.
- **D**: Training sets need labels and are for training, not scoring.

Objective S1.09 · [Revisit](03-feature-engineering-uc.md#scoring-with-features-from-a-feature-table)

</details>

### Question 8

A fraud model served on a real-time endpoint must fetch a card's latest spending features within a few milliseconds per request. Where should these features be read from?

- **A.** An online feature table published from the offline feature table to a Databricks Online Feature Store
- **B.** The offline Delta feature table, queried with Spark for every request
- **C.** An MLflow artifact containing a CSV of features
- **D.** The training set DataFrame cached on the driver

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Online stores serve low-latency key lookups for real-time inference; the offline table remains the source of truth and is published to the online store.

Why the other options are wrong:

- **B**: Spark queries on Delta tables take seconds, too slow per request.
- **C**: Artifacts are static files, not a serving store.
- **D**: A cached training DataFrame is stale and not reachable from an endpoint.

Objective S1.10 · [Revisit](03-feature-engineering-uc.md#offline-vs-online-feature-tables)

</details>

### Question 9

Which TWO statements describe offline feature tables correctly?

- **A.** They are Delta tables in Unity Catalog used for training and batch scoring
- **B.** They are the source from which online feature tables are published
- **C.** They provide millisecond lookups for real-time endpoints
- **D.** They are deleted automatically after a model is trained
- **E.** They cannot store more than one row per key

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — Offline tables hold the full feature data for training and batch inference and feed the online store through publishing.

Why the other options are wrong:

- **C**: Millisecond lookups are the role of online stores.
- **D**: Feature tables persist and are reused by many models.
- **E**: Time-series feature tables store many rows per key, one per timestamp.

Objective S1.10 · [Revisit](03-feature-engineering-uc.md#offline-vs-online-feature-tables)

</details>

### Question 10

A notebook imports FeatureStoreClient from databricks.feature_store to create a new feature table in a Unity Catalog workspace created in 2026. What should the data scientist use instead?

- **A.** FeatureEngineeringClient from databricks.feature_engineering
- **B.** mlflow.FeatureClient
- **C.** The same FeatureStoreClient with register_table
- **D.** databricks.sdk.WorkspaceClient().tables.create

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — FeatureStoreClient belongs to the deprecated workspace feature store; Unity Catalog feature tables use FeatureEngineeringClient.

Why the other options are wrong:

- **B**: MLflow has no FeatureClient.
- **C**: register_table is the legacy workspace feature store API.
- **D**: The SDK can create tables but does not provide feature lookups and training sets.

Objective S1.06 · [Revisit](03-feature-engineering-uc.md#pitfalls)

</details>
<!-- /questions -->
