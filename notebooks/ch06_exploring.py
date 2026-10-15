# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 6 tutorial: Exploring data
# MAGIC **Objectives**: S2.01, S2.03, S2.04 · **Runtime**: 18 LTS ML · **Requires**: serverless or classic compute · **Free Edition**: full
# MAGIC **Estimated time**: 35 min · **Clean-up**: none
# MAGIC
# MAGIC Compute summary statistics three ways, visualise categorical and continuous features, and compare pairs
# MAGIC of features with the right method. Run `00_setup` first.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

churn = spark.table(uc_name("customer_churn"))
houses = spark.table(uc_name("house_prices"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Summary statistics (S2.01)
# MAGIC
# MAGIC `.summary()` adds percentiles to what `.describe()` returns; you can also name the statistics you want.

# COMMAND ----------

from db_ml_associate.ch06_exploring.summary import summary_examples

everything, chosen, basic = summary_examples(churn.select("tenure_months", "monthly_charges", "support_calls"))
display(everything)
display(chosen)
display(basic)

# COMMAND ----------

# MAGIC %md
# MAGIC `dbutils.data.summarize` renders an interactive profile: histograms, missing values and frequent values per
# MAGIC column. By default its statistics are approximate (fast on large data); `precise=True` computes exact values.

# COMMAND ----------

dbutils.data.summarize(churn)

# COMMAND ----------

# MAGIC %md
# MAGIC The same profile is available without code: run `display(churn)`, click **+** next to the results table and
# MAGIC choose **Data Profile**.

# COMMAND ----------

display(churn)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Visualising features (S2.03)
# MAGIC
# MAGIC Categorical → bar chart of counts. Continuous → histogram (shape, skew) and box plot (outliers).
# MAGIC The matplotlib helpers below work anywhere; in the notebook you can also click **+ → Visualization** on a
# MAGIC `display()` result and build the same charts in the visualization editor.

# COMMAND ----------

from db_ml_associate.ch06_exploring import plots

churn_pdf = churn.toPandas()
house_pdf = houses.toPandas()
display(plots.bar_counts(churn_pdf, "contract_type"))
display(plots.histogram(churn_pdf, "data_usage_gb"))
display(plots.histogram(house_pdf, "price"))
display(plots.box_by_category(churn_pdf, "support_calls", "contract_type"))

# COMMAND ----------

# MAGIC %md
# MAGIC `data_usage_gb` and `price` are right-skewed (long tail to the right) — remember them for the log transform
# MAGIC in chapter 7. The box plot shows the injected `support_calls` outliers as points beyond the whiskers.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Comparing two features (S2.04)
# MAGIC
# MAGIC | Pair | Method |
# MAGIC |---|---|
# MAGIC | Two continuous | Scatter plot, Pearson (linear) or Spearman (monotonic) correlation |
# MAGIC | Two categorical | Contingency table (crosstab), chi-square test |
# MAGIC | Continuous vs categorical | Box plot per category, group statistics |

# COMMAND ----------

from db_ml_associate.ch06_exploring.compare import (
    chi_square_test,
    contingency_table,
    continuous_correlation,
    numeric_by_category,
)

print("Pearson living area ~ price:", round(continuous_correlation(houses, "living_area_sqm", "price"), 3))
print(
    "Spearman living area ~ price:",
    round(continuous_correlation(houses, "living_area_sqm", "price", method="spearman"), 3),
)
display(plots.scatter(house_pdf, "living_area_sqm", "price"))

# COMMAND ----------

display(contingency_table(churn, "contract_type", "churned"))
print("Chi-square p-value contract_type ~ churned:", chi_square_test(churn, "contract_type", "churned"))

# COMMAND ----------

display(numeric_by_category(churn, "tenure_months", "churned"))

# COMMAND ----------

# MAGIC %md
# MAGIC A tiny p-value says contract type and churn are *associated*; it does not say the contract causes churn.
# MAGIC
# MAGIC **Self-check**: which method would you use to compare `neighborhood` and `condition`? (Both categorical:
# MAGIC crosstab and chi-square.)
