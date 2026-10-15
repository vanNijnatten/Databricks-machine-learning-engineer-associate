# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 8 tutorial: ML foundations
# MAGIC **Objectives**: S3.01, S3.02, S3.15 · **Runtime**: 18 LTS ML · **Requires**: serverless or classic compute · **Free Edition**: full
# MAGIC **Estimated time**: 30 min · **Clean-up**: none
# MAGIC
# MAGIC Compare imbalance mitigations on the churn data and watch bias turn into variance as a tree grows deeper.
# MAGIC Run `00_setup` first.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

from db_ml_associate.common import CHURN_LABEL, HOUSE_TARGET, split_frame

churn_pdf = spark.table(uc_name("customer_churn")).toPandas()
house_pdf = spark.table(uc_name("house_prices")).toPandas()
train, test = split_frame(churn_pdf, CHURN_LABEL)  # split FIRST, stratified on the label
print("churn rate train/test:", round(train[CHURN_LABEL].mean(), 3), round(test[CHURN_LABEL].mean(), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Choosing an algorithm (S3.01)
# MAGIC
# MAGIC Churn is binary classification on tabular data with a need to explain decisions → start with logistic
# MAGIC regression; try gradient-boosted trees when accuracy matters more than explainability. House prices are
# MAGIC regression; customer segments without labels would be clustering.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Mitigating class imbalance (S3.02)

# COMMAND ----------

from db_ml_associate.ch08_foundations.imbalance import (
    balanced_class_weights,
    make_churn_classifier,
    minority_recall,
    random_oversample,
    random_undersample,
)

features, labels = train.drop(columns=CHURN_LABEL), train[CHURN_LABEL]
test_features, test_labels = test.drop(columns=CHURN_LABEL), test[CHURN_LABEL]
print("balanced weights:", balanced_class_weights(labels))

results = {}
results["unweighted"] = minority_recall(make_churn_classifier().fit(features, labels), test_features, test_labels)
results["class_weight=balanced"] = minority_recall(
    make_churn_classifier(class_weight="balanced").fit(features, labels), test_features, test_labels
)
for name, resampled in (
    ("oversampled", random_oversample(train, CHURN_LABEL)),
    ("undersampled", random_undersample(train, CHURN_LABEL)),
):
    model = make_churn_classifier().fit(resampled.drop(columns=CHURN_LABEL), resampled[CHURN_LABEL])
    results[name] = minority_recall(model, test_features, test_labels)
display(spark.createDataFrame([(k, float(v)) for k, v in results.items()], "method string, minority_recall double"))

# COMMAND ----------

# MAGIC %md
# MAGIC > **⚠️ Wrong approach:** resampling before the split copies minority rows into both train and validation —
# MAGIC > the validation score is inflated.

# COMMAND ----------

from db_ml_associate.ch08_foundations import pitfalls

print("F1, resampled before split (inflated):", round(pitfalls.wrong_resample_before_split(churn_pdf), 3))
print("F1, resampled after split (honest):  ", round(pitfalls.correct_resample_after_split(churn_pdf), 3))
print("Always 'no churn':", pitfalls.wrong_accuracy_on_imbalanced(churn_pdf))
print("Class-weighted model:", pitfalls.correct_use_recall_f1(churn_pdf))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Bias, variance and model complexity (S3.15)

# COMMAND ----------

import matplotlib.pyplot as plt

from db_ml_associate.ch08_foundations.bias_variance import complexity_sweep

house_train, house_validation = split_frame(house_pdf, HOUSE_TARGET, stratify=False)
sweep = complexity_sweep(house_train, house_validation)
display(sweep)
figure, axis = plt.subplots(figsize=(6, 4))
axis.plot(sweep["depth"], sweep["train_rmse"], marker="o", label="training")
axis.plot(sweep["depth"], sweep["validation_rmse"], marker="o", label="validation")
axis.set(xlabel="max_depth (complexity)", ylabel="RMSE of log(price)", title="Bias–variance trade-off")
axis.legend()
display(figure)

# COMMAND ----------

# MAGIC %md
# MAGIC Depth 1: both errors high → **high bias** (underfitting). Large depth: training error near zero, validation
# MAGIC error rising → **high variance** (overfitting). The best depth minimises validation error.
# MAGIC
# MAGIC **Self-check**: which depth would you pick, and what would more training data do to the gap?
