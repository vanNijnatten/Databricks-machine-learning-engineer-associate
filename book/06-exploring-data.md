---
chapter: 6
title: Exploring data
objectives: [S2.01, S2.03, S2.04]
prerequisites: []
reading_minutes: 70
references: [dbutils, eda-tutorial, notebook-visualizations, spark-mllib]
---

# 6. Exploring data

> **Exam mapping**: Section 2 — Data Processing · objectives S2.01, S2.03, S2.04 ·
> **Runtime**: Databricks Runtime 18 LTS ML · **No prerequisites** — a good second chapter after
> the introduction.

Before you clean or model data you need to *see* it: how values are distributed, where values are
missing, which features move together. The exam asks for three skills: computing summary
statistics on a Spark DataFrame, choosing visualisations for categorical and continuous features,
and comparing two features with the right method. The examples use `customer_churn` and
`house_prices` ([`data/README.md`](../data/README.md)).

## Learning objectives

- Compute summary statistics with `summary()`, `describe()` and `dbutils.data.summarize`, and know
  what each returns.
- Generate a data profile and visualisations from a notebook result without code.
- Pick a chart for a categorical or a continuous feature.
- Compare two continuous, two categorical, or one of each with the appropriate method.
- Recognise NaN values that masquerade as data in Spark.

## Feature types

<!-- objectives: S2.03, S2.04 -->

| Type | Values | Examples in the tutorial data |
|---|---|---|
| **Continuous** (numeric) | Any number in a range | `monthly_charges`, `data_usage_gb`, `price` |
| **Discrete numeric** | Counts | `support_calls`, `bedrooms` |
| **Categorical (nominal)** | Labels without order | `contract_type`, `payment_method`, `neighborhood` |
| **Ordinal** | Labels with an order | `condition` (poor < fair < good < excellent) |

The type decides which statistics, charts and comparison methods make sense.

## Summary statistics

<!-- objectives: S2.01 -->

### `summary()` and `describe()` on a Spark DataFrame

<!-- snippet: src/db_ml_associate/ch06_exploring/summary.py#spark_summary -->
```python
everything = frame.summary()  # count, mean, stddev, min, 25%, 50%, 75%, max
chosen = frame.summary("count", "min", "25%", "75%", "max")  # only the statistics you name
basic = frame.describe()  # count, mean, stddev, min, max: no percentiles
```

| Method | Statistics | Notes |
|---|---|---|
| `df.summary()` | count, mean, stddev, min, **25 %, 50 %, 75 %**, max | Pass names to choose statistics, e.g. `summary("count", "50%")`; any percentile such as `"90%"` works; percentiles are approximate on large data |
| `df.describe()` | count, mean, stddev, min, max | No percentiles |
| pandas `df.describe(include="all")` | numeric statistics + count/unique/top/freq for strings | Data must fit in driver memory |

Both Spark methods return a **DataFrame of strings** (one row per statistic), so call `.show()` or
`display()` to look at it, and cast before doing arithmetic. String columns get count, min and max
(lexicographic); their other statistics are null.

On the churn data:

| summary | tenure_months | support_calls |
|---|---|---|
| count | 5000 | 5000 |
| mean | 36.41 | 1.57 |
| stddev | 20.84 | 1.67 |
| min | 1 | 0 |
| 25% | 18 | 1 |
| 50% | 36 | 1 |
| 75% | 55 | 2 |
| max | 72 | 25 |

`support_calls` already tells a story: a median of 1, a 75th percentile of 2 — and a maximum of
25. Those are the injected outliers chapter 7 removes.

### `dbutils.data.summarize` and the data profile

In a Databricks notebook, `dbutils.data.summarize(df)` renders an interactive summary of a Spark or
pandas DataFrame: statistics and a histogram per column
([Databricks Utilities reference](https://docs.databricks.com/aws/en/dev-tools/databricks-utils)).

<!-- snippet: src/db_ml_associate/ch06_exploring/summary.py#dbutils_summarize -->
```python
dbutils.data.summarize(frame, precise=precise)  # precise=True: exact statistics, slower
```

- It analyses the **whole** DataFrame, so it can be expensive on large data.
- By default (`precise=False`) some statistics are **approximate**: distinct counts of
  high-cardinality columns can be off by about 5 %, and histograms and percentiles by up to 0.01 % of
  the row count. `precise=True` computes exact statistics, except for numeric histograms and
  percentiles.
- It returns nothing — it only displays.

The same profile is available without code: run `display(df)`, then click **+ → Data Profile** next
to the **Table** tab of the result
([EDA tutorial](https://docs.databricks.com/aws/en/notebooks/eda-tutorial)).

> **Exam tip:** "quickly compute count, mean, standard deviation, min, max **and quartiles** of a
> Spark DataFrame" → `df.summary()`. "Interactive statistics and histograms for every column in a
> notebook" → `dbutils.data.summarize(df)` or the Data Profile tab.

## Visualising features

<!-- objectives: S2.03 -->

| Feature | Question | Chart |
|---|---|---|
| One **continuous** | Shape, skew, outliers? | **Histogram**, box plot, density plot |
| One **categorical** | How frequent is each category? | **Bar chart** of counts (avoid pie charts with many categories) |
| Continuous by categorical | Do groups differ? | **Box plot** per category, grouped bar of means |
| Two continuous | Do they move together? | **Scatter plot** (add a trend line) |
| Two categorical | How do they combine? | **Heatmap** of a cross-tab, stacked bar chart |

**Without code:** `display(df)` and **+ → Visualization** in the results pane opens the
visualisation editor: choose a type (bar, line, area, pie, histogram, heatmap, scatter, box, …), the
columns, and aggregation
([Visualizations in Databricks notebooks](https://docs.databricks.com/aws/en/notebooks/visualizations/)).

**With code** (matplotlib on pandas; for big data aggregate in Spark first, or plot a sample):

<!-- snippet: src/db_ml_associate/ch06_exploring/plots.py#histogram -->
```python
figure, axes = plt.subplots(figsize=(6, 4))
axes.hist(frame[column].dropna(), bins=bins)
axes.set(title=f"Distribution of {column}", xlabel=column, ylabel="rows")
```

<!-- snippet: src/db_ml_associate/ch06_exploring/plots.py#bar_counts -->
```python
counts = frame[column].value_counts()
figure, axes = plt.subplots(figsize=(6, 4))
axes.bar(counts.index.astype(str), counts.to_numpy())
axes.set(title=f"Rows per {column}", xlabel=column, ylabel="rows")
```

<!-- snippet: src/db_ml_associate/ch06_exploring/plots.py#box_by_category -->
```python
groups = {name: group[numeric].dropna() for name, group in frame.groupby(category)}
figure, axes = plt.subplots(figsize=(6, 4))
axes.boxplot(list(groups.values()), tick_labels=list(groups))
axes.set(title=f"{numeric} by {category}", xlabel=category, ylabel=numeric)
```

A histogram of `data_usage_gb` shows a long right tail (a candidate for a log transform, chapter
7); a bar chart of `contract_type` shows that month-to-month dominates; a box plot of
`tenure_months` by `churned` shows churners have much shorter tenure.

## Comparing two features

<!-- objectives: S2.04 -->

Pick the method from the two feature types:

| Feature types | Numeric method | Visual |
|---|---|---|
| Continuous × continuous | **Correlation**: Pearson (linear relation), Spearman (any monotonic relation, robust to outliers) | Scatter plot |
| Categorical × categorical | **Contingency table** (cross-tab) + **chi-square test** of independence | Heatmap / stacked bars |
| Categorical × continuous | **Statistics per group** (mean, median, spread); t-test / ANOVA for significance | Box plot per group |

### Two continuous features: correlation

<!-- snippet: src/db_ml_associate/ch06_exploring/compare.py#correlation -->
```python
if method == "pearson":
    return float(frame.stat.corr(first, second))  # DataFrame API supports Pearson only
# Spearman (rank correlation) needs the vector-based MLlib API
vectors = VectorAssembler(inputCols=[first, second], outputCol="pair").transform(frame.select(first, second))
matrix = Correlation.corr(vectors, "pair", method="spearman").head()[0]
return float(matrix[0, 1])
```

`df.stat.corr()` computes Pearson only; Spearman needs `pyspark.ml.stat.Correlation` on a vector
column. Living area against house price: Pearson **0.530**, Spearman **0.528** — a moderate positive
relation. Correlation runs from −1 to 1, measures association (not causation), and Pearson misses
non-linear patterns entirely: always look at the scatter plot too.

### Two categorical features: cross-tab and chi-square

<!-- snippet: src/db_ml_associate/ch06_exploring/compare.py#crosstab -->
```python
return frame.stat.crosstab(first, second)  # first column is named "<first>_<second>"
```

| contract_type | not churned (0) | churned (1) | churn rate |
|---|---|---|---|
| month-to-month | 2400 | 392 | 14.0 % |
| one-year | 1197 | 60 | 4.8 % |
| two-year | 903 | 48 | 5.0 % |

The **chi-square test** asks whether the observed counts could arise if the two features were
independent:

<!-- snippet: src/db_ml_associate/ch06_exploring/compare.py#chi_square -->
```python
indexed = StringIndexer(inputCol=category, outputCol="category_index").fit(frame).transform(frame)
vectors = VectorAssembler(inputCols=["category_index"], outputCol="features").transform(indexed)
result = ChiSquareTest.test(vectors.withColumn(label, F.col(label).cast("double")), "features", label).head()
p_value = float(result.pValues[0])  # small p-value: the two are not independent
```

For `contract_type` vs `churned` the p-value is effectively 0 — strongly related. For
`payment_method` vs `churned` it is 0.82 — no evidence of a relation in this data.

### A categorical and a continuous feature: statistics per group

<!-- snippet: src/db_ml_associate/ch06_exploring/compare.py#numeric_by_category -->
```python
return frame.groupBy(category).agg(
    F.count("*").alias("rows"),
    F.mean(numeric).alias(f"mean_{numeric}"),
    F.median(numeric).alias(f"median_{numeric}"),
    F.stddev(numeric).alias(f"stddev_{numeric}"),
)
```

| churned | rows | mean tenure | median tenure | stddev |
|---|---|---|---|---|
| 0 | 4500 | 37.8 | 38 | 20.7 |
| 1 | 500 | 23.7 | 21 | 17.7 |

> **Exam tip:** two categorical features → chi-square / cross-tab; two continuous → correlation /
> scatter; "the relation is monotonic but not linear" or "outliers are present" → Spearman rather than
> Pearson.

## Choosing an approach

| Need | Use |
|---|---|
| Fast numeric overview in code, including quartiles | `df.summary()` |
| Overview without percentiles | `df.describe()` |
| Interactive per-column statistics and histograms | `dbutils.data.summarize(df)` or Data Profile tab |
| Quick charts while exploring | `display(df)` + visualisation editor |
| Reproducible charts in a report or job | matplotlib/seaborn/plotly code (preinstalled in the ML runtime) |
| Exact statistics on a modest table | `dbutils.data.summarize(df, precise=True)` |

## Pitfalls

**1. NaN is not null.** When a pandas DataFrame with missing floats is converted with
`spark.createDataFrame`, the gaps arrive as **NaN**, a valid double, not SQL `null`. `summary()` then
counts them, and NaN poisons the mean, standard deviation and maximum:

> **⚠️ Wrong approach:** summarise NaN-filled columns directly.

| `data_usage_gb` | count | mean | max |
|---|---|---|---|
| NaN left in place | 5000 | NaN | NaN |
| NaN converted to null | 4800 | 17.30 | 174.51 |

Convert NaN to null first:

<!-- snippet: src/db_ml_associate/ch06_exploring/pitfalls.py#nan_to_null -->
```python
for column in columns:
    frame = frame.withColumn(column, F.when(F.isnan(column), None).otherwise(F.col(column)))
```

**2. `summary()` returns strings.** Cast before comparing or calculating: `"9" > "10"` as strings.

**3. Profiling everything on huge tables.** `dbutils.data.summarize` reads the whole DataFrame;
`toPandas()` for a chart can exhaust the driver. Sample or aggregate first.

**4. Pearson for non-linear or outlier-heavy data.** A strong curved relation can have Pearson ≈ 0;
one outlier can create or destroy a Pearson correlation. Check the scatter plot and Spearman.

**5. Correlation ≠ causation.** A correlation between support calls and churn does not mean calls cause
churn.

## Summary

- `summary()` gives count, mean, stddev, min, quartiles and max; `describe()` drops the quartiles;
  `dbutils.data.summarize` and the Data Profile tab give interactive per-column statistics and
  histograms (approximate unless `precise=True`).
- Histogram/box plot for continuous, bar chart for categorical, scatter for two continuous, box plot
  per group for mixed pairs.
- Compare continuous pairs with correlation, categorical pairs with a cross-tab and chi-square test,
  mixed pairs with per-group statistics.

## Review questions

<!-- questions: ch06 -->

### Question 1

A data scientist needs the count, mean, standard deviation, minimum, quartiles and maximum of every numeric column of a large Spark DataFrame, using a single DataFrame method. Which call returns all of these?

- **A.** df.describe()
- **B.** df.summary()
- **C.** df.count()
- **D.** df.dtypes

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — summary() returns count, mean, stddev, min, 25%, 50%, 75% and max by default.

Why the other options are wrong:

- **A**: describe() omits the quartiles (25%, 50%, 75%).
- **C**: count() returns only the number of rows.
- **D**: dtypes lists column names and types, not statistics.

Objective S2.01 · [Revisit](06-exploring-data.md#summary-and-describe-on-a-spark-dataframe)

</details>

### Question 2

In a Databricks notebook, an analyst runs dbutils.data.summarize(df) on a table with a high-cardinality customer_city column and notices that the reported number of distinct cities differs slightly from a COUNT(DISTINCT) query. What explains this, and how can they get exact numbers?

- **A.** summarize samples 1% of rows; call df.cache() first
- **B.** Some statistics are approximate by default; pass precise=True
- **C.** summarize ignores string columns; cast the column to integer
- **D.** summarize only reads the first 1,000 rows; use display(df) instead

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — By default dbutils.data.summarize computes some statistics approximately (distinct counts of high-cardinality columns can be off by about 5%); precise=True computes them exactly.

Why the other options are wrong:

- **A**: summarize analyses the whole DataFrame; caching does not change accuracy.
- **C**: String columns are summarised; casting city names to integers is impossible.
- **D**: summarize is not limited to the first rows.

Objective S2.01 · [Revisit](06-exploring-data.md#dbutilsdatasummarize-and-the-data-profile)

</details>

### Question 3

A pandas DataFrame with missing values in a float column is converted with spark.createDataFrame(pdf). df.summary() then reports the column's count as equal to the number of rows and its mean as NaN. What is the most likely cause?

- **A.** The missing values were converted to NaN, which Spark treats as a value rather than null
- **B.** summary() cannot compute means on float columns
- **C.** The DataFrame has too many partitions
- **D.** The column contains negative numbers

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Missing pandas floats become NaN in Spark. NaN is a valid double, so it is counted and propagates through mean and stddev; convert NaN to null first.

Why the other options are wrong:

- **B**: summary() computes means for numeric columns.
- **C**: Partitioning does not change statistics.
- **D**: Negative numbers do not produce a NaN mean.

Objective S2.01 · [Revisit](06-exploring-data.md#pitfalls)

</details>

### Question 4

A data scientist wants to see whether the continuous feature transaction_amount is skewed and whether it has extreme values. Which visualisation is most appropriate?

- **A.** A histogram of transaction_amount
- **B.** A pie chart of transaction_amount
- **C.** A bar chart of the count of each customer_id
- **D.** A line chart of transaction_amount sorted by customer_id

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — A histogram shows the distribution of one continuous feature, including skew and outliers.

Why the other options are wrong:

- **B**: Pie charts suit a few categories, not a continuous variable.
- **C**: This shows customer frequency, not the amount distribution.
- **D**: Ordering by an identifier imposes a meaningless sequence.

Objective S2.03 · [Revisit](06-exploring-data.md#visualising-features)

</details>

### Question 5

In a Databricks notebook, which steps produce a chart of a query result without writing any plotting code?

- **A.** Run display(df), then use + > Visualization in the results pane
- **B.** Run df.show(), then right-click the text output
- **C.** Run df.summary() and export it to CSV
- **D.** Run dbutils.fs.ls() on the table location

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — display() renders a results table whose + menu adds visualizations (and data profiles).

Why the other options are wrong:

- **B**: show() prints plain text without a visualization menu.
- **C**: summary() returns statistics, not a chart.
- **D**: dbutils.fs.ls lists files.

Objective S2.03 · [Revisit](06-exploring-data.md#visualising-features)

</details>

### Question 6

A data scientist wants to know whether subscription_plan (Basic, Plus, Premium) and region (North, South, East, West) are associated. Which method is appropriate?

- **A.** Pearson correlation between the two columns
- **B.** A contingency table and a chi-square test of independence
- **C.** A scatter plot of the two columns
- **D.** RMSE between the two columns

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — Both features are categorical, so cross-tabulate them and test independence with chi-square.

Why the other options are wrong:

- **A**: Pearson correlation requires numeric variables.
- **C**: Scatter plots compare two continuous variables.
- **D**: RMSE is a regression error metric, not an association measure.

Objective S2.04 · [Revisit](06-exploring-data.md#two-categorical-features-cross-tab-and-chi-square)

</details>

### Question 7

The relation between advertising spend and sign-ups is clearly monotonic (more spend, more sign-ups) but strongly curved, and there are a few extreme spend values. Which statistic best summarises the strength of the relation?

- **A.** Pearson correlation
- **B.** Spearman rank correlation
- **C.** Chi-square statistic
- **D.** Mean of each column

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — Spearman correlation measures monotonic relations on ranks, so it handles curvature and is robust to outliers.

Why the other options are wrong:

- **A**: Pearson only measures linear relations and is sensitive to outliers.
- **C**: Chi-square compares categorical variables.
- **D**: Means describe each column separately, not their relation.

Objective S2.04 · [Revisit](06-exploring-data.md#two-continuous-features-correlation)

</details>

### Question 8

Which TWO are appropriate ways to compare the continuous feature tenure_months across the categories of the label churned (yes/no)?

- **A.** A box plot of tenure_months per churned value
- **B.** The mean and median of tenure_months per churned value using groupBy().agg()
- **C.** Pearson correlation between churned and customer_id
- **D.** A pie chart of tenure_months
- **E.** A chi-square test between tenure_months and customer_id

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — For a continuous feature against a categorical one, compare per-group statistics and distributions, for example with grouped aggregates and box plots.

Why the other options are wrong:

- **C**: customer_id is an identifier, not tenure, and correlation with it is meaningless.
- **D**: Pie charts are not suitable for a continuous feature.
- **E**: Neither variable pair is relevant, and chi-square needs categorical features.

Objective S2.04 · [Revisit](06-exploring-data.md#a-categorical-and-a-continuous-feature-statistics-per-group)

</details>

### Question 9

Which Spark DataFrame method computes the Pearson correlation between two numeric columns directly, without assembling a vector?

- **A.** df.stat.corr("col_a", "col_b")
- **B.** df.stat.crosstab("col_a", "col_b")
- **C.** df.summary("corr")
- **D.** df.corr_spearman("col_a", "col_b")

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — DataFrame.stat.corr returns the Pearson correlation of two numeric columns.

Why the other options are wrong:

- **B**: crosstab builds a contingency table of two (categorical) columns.
- **C**: summary has no correlation statistic.
- **D**: There is no such method; Spearman needs pyspark.ml.stat.Correlation.

Objective S2.04 · [Revisit](06-exploring-data.md#two-continuous-features-correlation)

</details>
<!-- /questions -->
