# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 3 tutorial: Feature engineering in Unity Catalog
# MAGIC **Objectives**: S1.05–S1.10 · **Runtime**: 18 LTS ML · **Requires**: Unity Catalog; online store (optional, paid) · **Free Edition**: partial
# MAGIC **Estimated time**: 60 min · **Clean-up**: delete the online store if you created one (last cell or `99_cleanup`)
# MAGIC
# MAGIC Create a feature table, upsert into it, train a model with `FeatureLookup`, score it with `score_batch`, and —
# MAGIC optionally — publish the table to an online store. Run `00_setup` first.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

dbutils.widgets.dropdown("publish_online", "no", ["no", "yes"], "Create an online store (paid)")
PUBLISH_ONLINE = dbutils.widgets.get("publish_online") == "yes"

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()
FEATURE_TABLE = uc_name("customer_features")
MODEL_NAME = uc_name("churn_feature_model")  # takes customer_id as input; features are looked up
ONLINE_STORE = "ml-associate-online-store"
churn = spark.table(uc_name("customer_churn"))
spark.sql(f"DROP TABLE IF EXISTS {FEATURE_TABLE}")  # makes the notebook re-runnable

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Create a feature table (S1.05, S1.06)
# MAGIC
# MAGIC Any Delta table in Unity Catalog with a primary key is a feature table. Because it lives in Unity Catalog it is
# MAGIC governed, discoverable and lineage-tracked across every workspace attached to the metastore.

# COMMAND ----------

from db_ml_associate.ch03_features.features import compute_customer_features
from db_ml_associate.ch03_features.workspace import create_customer_feature_table

features = compute_customer_features(churn)
create_customer_feature_table(fe, FEATURE_TABLE, features)
display(spark.sql(f"DESCRIBE TABLE EXTENDED {FEATURE_TABLE}"))

# COMMAND ----------

# MAGIC %md
# MAGIC Open **Catalog → ml_associate → tutorial → customer_features**: the table shows a primary key constraint, and
# MAGIC the **Features** page in the sidebar lists it. The SQL alternative (`CREATE TABLE … CONSTRAINT … PRIMARY KEY`)
# MAGIC is printed below for reference; it is not run.

# COMMAND ----------

from db_ml_associate.ch03_features.workspace import ADD_PRIMARY_KEY_SQL, CREATE_FEATURE_TABLE_SQL

print(CREATE_FEATURE_TABLE_SQL)
print(ADD_PRIMARY_KEY_SQL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Write to the feature table (S1.07)
# MAGIC
# MAGIC `mode="merge"` upserts by primary key: changed rows are updated, new keys inserted, other rows untouched.

# COMMAND ----------

from pyspark.sql import functions as F

from db_ml_associate.ch03_features.workspace import upsert_features

changed = features.filter(F.col("customer_id") <= 100).withColumn("support_calls", F.col("support_calls") + 1)
upsert_features(fe, FEATURE_TABLE, changed)
display(fe.read_table(name=FEATURE_TABLE).filter("customer_id <= 3"))
print("rows:", fe.read_table(name=FEATURE_TABLE).count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Train with features from the feature table (S1.08)
# MAGIC
# MAGIC The labels table holds only the key and the label. `create_training_set` joins the features; `fe.log_model`
# MAGIC stores the lookups with the model so that scoring needs only the keys.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from db_ml_associate.ch03_features.workspace import build_training_set, log_feature_model

labels = churn.select("customer_id", "churned")
training_set, training_df = build_training_set(fe, labels, FEATURE_TABLE)
train_pdf = training_df.toPandas()
numeric = [c for c in train_pdf.columns if c not in ("churned", "payment_method")]
model = Pipeline(
    [
        (
            "prepare",
            ColumnTransformer(
                [
                    (
                        "numeric",
                        Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                        numeric,
                    ),
                    ("categorical", OneHotEncoder(handle_unknown="ignore"), ["payment_method"]),
                ]
            ),
        ),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ]
).fit(train_pdf.drop(columns="churned"), train_pdf["churned"])

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/ml-associate-ch03")
log_feature_model(fe, model, training_set, MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Score with features from the feature table (S1.09)

# COMMAND ----------

from mlflow import MlflowClient

from db_ml_associate.ch03_features.workspace import score

latest = max(int(v.version) for v in MlflowClient().search_model_versions(f"name = '{MODEL_NAME}'"))
predictions = score(fe, f"models:/{MODEL_NAME}/{latest}", labels.select("customer_id"))  # keys only
display(predictions.select("customer_id", "prediction").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Online feature store (S1.10) — optional, paid
# MAGIC
# MAGIC The offline table serves training and batch scoring. A real-time endpoint needs millisecond lookups, so the
# MAGIC table is published to a Lakebase-backed **online store**.
# MAGIC
# MAGIC > **Requires:** an online feature store is billed while it exists (capacity `CU_1` is the smallest). Not
# MAGIC > available on Free Edition. Set the `publish_online` widget to `yes` to run this step and **delete the store**
# MAGIC > with the last cell when you are done.

# COMMAND ----------

if PUBLISH_ONLINE:
    from db_ml_associate.ch03_features.workspace import create_online_store, publish_online

    spark.sql(f"ALTER TABLE {FEATURE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    store = create_online_store(fe, ONLINE_STORE)
    publish_online(fe, store, FEATURE_TABLE, f"{FEATURE_TABLE}_online")
    print("published to", f"{FEATURE_TABLE}_online")
else:
    print("Skipped (publish_online = no). Offline vs online: training/batch vs low-latency serving lookups.")

# COMMAND ----------

if PUBLISH_ONLINE:
    fe.delete_online_store(name=ONLINE_STORE)  # stop the cost; 99_cleanup also does this
    print("online store deleted")

# COMMAND ----------

# MAGIC %md
# MAGIC **Self-check**: why does `score_batch` need only `customer_id`, and what would change if the model had been
# MAGIC logged with `mlflow.sklearn.log_model` instead of `fe.log_model`?
