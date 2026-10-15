"""Lakeflow pipeline source: streaming churn inference (exam objective S4.04).

This file is *pipeline source code*: add it to a Lakeflow pipeline (formerly a Delta Live Tables
pipeline) instead of running it in a notebook. The pipeline runs the decorated functions, manages
checkpoints and autoscales its compute with the event volume. Configure the model and source table
with the pipeline settings ``churn.model_uri`` and ``churn.source_table``.
"""

# --8<-- [start:pipeline]
import mlflow
from pyspark import pipelines as dp  # formerly: import dlt
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

spark = SparkSession.active()
MODEL_URI = spark.conf.get("churn.model_uri", "models:/ml_associate.tutorial.churn_model@champion")
SOURCE_TABLE = spark.conf.get("churn.source_table", "ml_associate.tutorial.customer_events_raw")
FEATURES = ["tenure_months", "monthly_charges", "data_usage_gb", "support_calls", "contract_type", "payment_method"]

# Load the model once as a Spark UDF; the champion alias picks up promotions on the next update.
churn_model = mlflow.pyfunc.spark_udf(spark, model_uri=MODEL_URI, result_type="double")


@dp.table(comment="Customer events arriving incrementally")
def customer_events() -> DataFrame:
    return spark.readStream.table(SOURCE_TABLE)  # streaming read → streaming table


@dp.table(comment="Churn prediction for every new customer event")
def churn_predictions() -> DataFrame:
    events = spark.readStream.table("customer_events")  # another dataset of this pipeline
    return events.withColumn("churn_prediction", churn_model(F.struct(*FEATURES)))


# --8<-- [end:pipeline]
