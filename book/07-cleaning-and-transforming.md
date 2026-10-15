---
chapter: 7
title: Cleaning and transforming data
objectives: [S2.02, S2.05, S2.06, S2.07, S2.08, S2.09]
prerequisites: [6]
reading_minutes: 90
references: [spark-mllib]
---

# 7. Cleaning and transforming data

> **Exam mapping**: Section 2 — Data Processing · objectives S2.02, S2.05–S2.09 ·
> **Runtime**: Databricks Runtime 18 LTS ML · **Read after**: chapter 6.

Real data has extreme values, gaps and text categories; most algorithms want clean numbers. This
chapter covers the four preparation steps the exam tests: removing outliers, imputing missing
values, one-hot encoding categories and log-transforming skewed values — and when *not* to apply
each one.

## Learning objectives

- Remove outliers from a Spark DataFrame with a standard-deviation (z-score) or IQR filter.
- Compare mean, median and mode imputation and choose one by looking at the distribution.
- Impute missing values in Spark and pandas.
- One-hot encode categorical features, and explain for which models and datasets it is (not)
  appropriate.
- Recognise when a log transformation helps and how to undo it.

## Removing outliers

<!-- objectives: S2.02 -->

An **outlier** is a value far from the rest. It may be an error (a typo, a sensor glitch) or a rare
but real event. Remove it when it is an error or when it would distort the model; keep it when the
extremes are what you want to predict (fraud!).

### Standard-deviation (z-score) filter

Keep values within *t* standard deviations of the mean (*t* = 3 is common):

<!-- snippet: src/db_ml_associate/ch07_cleaning/outliers.py#stddev_filter -->
```python
stats = frame.agg(F.mean(column).alias("mean"), F.stddev(column).alias("std")).first()
lower = stats["mean"] - threshold * stats["std"]
upper = stats["mean"] + threshold * stats["std"]
kept = frame.filter(F.col(column).isNull() | F.col(column).between(lower, upper))
```

Simple, but the mean and standard deviation are themselves inflated by the outliers, and the rule
assumes a roughly symmetric, bell-shaped distribution.

### IQR (Tukey's fences) filter

The **interquartile range** IQR = Q3 − Q1 covers the middle 50 % of the data. Values below
Q1 − 1.5·IQR or above Q3 + 1.5·IQR are outliers (the box-plot whiskers):

<!-- snippet: src/db_ml_associate/ch07_cleaning/outliers.py#iqr_bounds -->
```python
q1, q3 = frame.approxQuantile(column, [0.25, 0.75], relative_error)  # 0.0 = exact (costly on big data)
iqr = q3 - q1
return q1 - factor * iqr, q3 + factor * iqr
```

<!-- snippet: src/db_ml_associate/ch07_cleaning/outliers.py#iqr_filter -->
```python
lower, upper = iqr_bounds(frame, column, factor)
return frame.filter(F.col(column).isNull() | F.col(column).between(lower, upper))
```

`approxQuantile(column, probabilities, relativeError)` computes quantiles in one pass;
`relativeError=0.0` is exact but expensive on big tables, `0.01` is usually fine.

| | Standard deviation | IQR |
|---|---|---|
| Assumes | Roughly normal, symmetric data | Nothing about shape |
| Robust to the outliers themselves | No (they inflate mean and std) | Yes (quartiles barely move) |
| Skewed data | Removes many legitimate high values | Better, though still asymmetric |
| Spark functions | `F.mean`, `F.stddev` | `approxQuantile` |

On the `support_calls` column (median 1, a handful of injected values between 15 and 25) both
filters remove the injected values; the tests use a small fixture with two planted outliers to prove
each filter removes exactly those.

Note the `isNull()` clause in both filters: a **missing value is not an outlier** — without it,
`filter` silently drops every row with a null, because comparisons with null are never true.

## Imputing missing values

<!-- objectives: S2.05, S2.06 -->

Most algorithms cannot handle missing values. Dropping rows loses information; **imputation**
fills the gap with a representative value.

| Strategy | Fill value | Use for | Weakness |
|---|---|---|---|
| **Mean** | Arithmetic average | Continuous, roughly symmetric data without outliers | Pulled by skew and outliers |
| **Median** | Middle value | Continuous **skewed** data or data with outliers | Ignores the distribution's shape |
| **Mode** | Most frequent value | **Categorical** and discrete data | Can over-represent one category |

Look at the distribution first (chapter 6): `data_usage_gb` is right-skewed (skewness 2.9). Its mean
is **17.3** GB but its median is **12.7** GB — 66 % of customers use less than the mean, so the mean
is not a "typical" value. `monthly_charges` is symmetric: mean 65.67, median 65.65, so either works.

> **Exam tip:** the official sample question's answer is to **examine the distribution and choose
> accordingly** — no strategy is automatically right for every continuous column, and
> `SimpleImputer` does not choose one for you (its default is the mean).

### In Spark

Spark ML's `Imputer` is an estimator: it learns the fill value per column in `fit()`:

<!-- snippet: src/db_ml_associate/ch07_cleaning/impute.py#spark_imputer -->
```python
imputer = Imputer(
    strategy=strategy,  # "mean" (default), "median" or "mode"
    inputCols=list(columns),
    outputCols=list(columns),  # overwrite; use new names to keep the originals
)  # an estimator: it learns the fill values in fit()
model = imputer.fit(frame)  # fit on TRAINING data only
imputed = model.transform(frame)  # treats both null and NaN as missing
```

`Imputer` works on **numeric** columns only and supports `mean` (default), `median` and `mode`. It
treats both null and NaN as missing (chapter 6 explains why both occur). For a string column, fill
with the most frequent value yourself:

<!-- snippet: src/db_ml_associate/ch07_cleaning/impute.py#categorical_mode -->
```python
mode = frame.filter(F.col(column).isNotNull()).groupBy(column).count().orderBy(F.desc("count")).first()[column]
return frame.fillna({column: mode})
```

### In pandas and scikit-learn

<!-- snippet: src/db_ml_associate/ch07_cleaning/impute.py#pandas_impute -->
```python
fill_values = {
    "mean": frame[column].mean(),
    "median": frame[column].median(),
    "mode": frame[column].mode().iloc[0],  # mode() can return several values; take the first
}
return frame.fillna({column: fill_values[strategy]})
```

scikit-learn's `SimpleImputer(strategy="mean" | "median" | "most_frequent" | "constant")` is the
pipeline-friendly version (chapter 9). Adding a **missing indicator** (`<column>_was_missing`, or
`SimpleImputer(add_indicator=True)`) keeps the information that the value was absent, which is
sometimes predictive.

## One-hot encoding

<!-- objectives: S2.07, S2.08 -->

Algorithms need numbers. **One-hot encoding** replaces a categorical column with one 0/1 column per
category. Encoding categories as integers 0, 1, 2 instead would invent an order and distances
("two-year" = 2 × "one-year") that linear models would take literally.

In Spark it is two steps — `StringIndexer` (string → index) and `OneHotEncoder` (index → sparse
vector):

<!-- snippet: src/db_ml_associate/ch07_cleaning/encoding.py#one_hot_spark -->
```python
indexed = [f"{column}_index" for column in columns]
encoded = [f"{column}_vector" for column in columns]
pipeline = Pipeline(
    stages=[
        StringIndexer(
            inputCols=list(columns), outputCols=indexed, handleInvalid="keep" if keep_unseen else "error"
        ),
        OneHotEncoder(inputCols=indexed, outputCols=encoded),  # dropLast=True drops one slot
    ]
)
return pipeline.fit(frame).transform(frame)
```

`OneHotEncoder` drops the last category by default (`dropLast=True`): with *k* categories, *k − 1*
slots suffice because "all zeros" means the last category, and it avoids perfectly collinear features
in linear models. With `handleInvalid="keep"`, `StringIndexer` reserves one more index for unseen
categories, so the vector has *k* slots. In pandas:

<!-- snippet: src/db_ml_associate/ch07_cleaning/encoding.py#one_hot_pandas -->
```python
return pd.get_dummies(frame, columns=list(columns), dtype=int)  # drop_first=True drops one category
```

### When is one-hot encoding (in)appropriate?

| Situation | One-hot? | Why / alternative |
|---|---|---|
| Linear or logistic regression, SVM, k-means, neural networks with a few categories | **Yes** | They need numeric inputs without a false order |
| Low-cardinality nominal feature (`contract_type`: 3 values) | **Yes** | A handful of extra columns |
| **High-cardinality** feature (`city`: 791 values in our data) | **No** | Hundreds of sparse columns, slow training, overfitting; use target/frequency encoding, hashing, embeddings, or group rare categories |
| **Tree-based models** (decision trees, random forests, gradient-boosted trees) | **Often unnecessary** | Trees can split on an index; Spark ML trees and LightGBM handle categorical indices; one-hot spreads a feature over many weak columns |
| **Ordinal** feature (`condition`: poor < fair < good < excellent) | Usually not | Map to ordered integers to keep the order |
| Category that appears only at prediction time | Needs care | `handleInvalid="keep"` / `handle_unknown="ignore"` |

> **Exam tip:** "one-hot encoding is not appropriate" → high-cardinality columns, tree-based models,
> ordinal features. "Appropriate" → nominal features with few categories for linear models.

## Log transformation

<!-- objectives: S2.09 -->

A **log transformation** compresses large values and stretches small ones. It is appropriate when:

- the feature or target is **right-skewed** with a long tail (incomes, prices, data usage, counts);
- effects are **multiplicative** (a 10 % price increase rather than +€10,000);
- the variance grows with the value (heteroscedasticity);
- the values span several orders of magnitude.

It requires values ≥ 0 (use `log1p(x) = log(1 + x)` when zeros occur) and is pointless for symmetric
data or for tree-based models' features (trees are invariant to monotonic transforms of a feature).

<!-- snippet: src/db_ml_associate/ch07_cleaning/transforms.py#log_transform -->
```python
return frame.withColumn(f"{column}_log", F.log1p(column))  # log1p is defined at 0; log(0) is not
```

| Column | Skewness before | Skewness after `log1p` |
|---|---|---|
| `data_usage_gb` | 2.93 | 0.20 |
| `price` | 1.20 | −0.08 |

When you log-transform the **target**, predictions come out on the log scale: exponentiate them
(`expm1` / `exp`) before computing metrics or showing them (chapter 11).

## Choosing an approach

| Question | Answer |
|---|---|
| Outliers in roughly normal data | Standard-deviation filter |
| Outliers in skewed data, or you want robustness | IQR filter |
| Missing values, symmetric continuous column | Mean (or median) |
| Missing values, skewed column or outliers | Median |
| Missing values, categorical column | Mode (most frequent) |
| Missingness itself is informative | Impute **and** add a missing indicator |
| Few categories, linear model | One-hot encoding |
| Many categories or tree model | Index / target encoding / hashing |
| Right-skewed positive values, multiplicative effects | Log (`log1p` with zeros) |

## Pitfalls

**1. Mean-imputing skewed data.** The mean of `data_usage_gb` (17.3) is above two-thirds of all real
values; the median (12.7) is not.

**2. Imputing (or computing outlier bounds) before the split.** The fill value then depends on test
rows.

> **⚠️ Wrong approach:** compute the median on all rows, then split.

<!-- snippet: src/db_ml_associate/ch07_cleaning/pitfalls.py#wrong_impute_before_split -->
```python
fill_value = churn[column].median()  # includes rows that will become test data
imputed = churn.fillna({column: fill_value})
train, test = split_frame(imputed, CHURN_LABEL, seed=seed)
```

<!-- snippet: src/db_ml_associate/ch07_cleaning/pitfalls.py#correct_impute_after_split -->
```python
train, test = split_frame(churn, CHURN_LABEL, seed=seed)
fill_value = train[column].median()  # training rows only
train, test = train.fillna({column: fill_value}), test.fillna({column: fill_value})
```

In practice put `Imputer` / `SimpleImputer` in the pipeline (chapter 9) so this happens automatically.

**3. One-hot encoding `city` for a tree model.** 791 new columns instead of one index column.

**4. `log(0)`.** `support_calls` contains zeros; `np.log` returns `-inf` for them. Use `log1p`.

**5. Filtering without handling nulls.** `filter(col.between(lo, hi))` drops null rows too.

**6. Forgetting the reverse transform.** Log-target predictions shown to users must be exponentiated.

## Summary

- Outliers: z-score filter for symmetric data, IQR filter for anything else; keep nulls.
- Imputation: mean for symmetric, median for skewed, mode for categorical — decide after looking at
  the distribution; learn fill values on training data only.
- One-hot encoding suits low-cardinality nominal features for linear models; avoid it for
  high-cardinality features and usually for tree models.
- Log transforms suit right-skewed positive values; use `log1p` with zeros and invert with `expm1`.

## Review questions

<!-- questions: ch07 -->

### Question 1

A column of order values is strongly right-skewed and contains a few extreme errors. The data scientist wants an outlier rule that is not itself distorted by those extreme values. Which rule should they use on the Spark DataFrame?

- **A.** Remove values more than 3 standard deviations from the mean
- **B.** Remove values outside Q1 - 1.5 x IQR and Q3 + 1.5 x IQR, computed with approxQuantile
- **C.** Remove the top and bottom 50% of values
- **D.** Remove all values above the mean

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — Quartiles barely move when extreme values are added, so IQR fences are robust and make no assumption about a symmetric distribution.

Why the other options are wrong:

- **A**: The mean and standard deviation are inflated by the outliers and assume roughly normal data.
- **C**: That removes all data except the median.
- **D**: Half of all legitimate values lie above the mean in symmetric data.

Objective S2.02 · [Revisit](07-cleaning-and-transforming.md#iqr-tukeys-fences-filter)

</details>

### Question 2

An engineer removes outliers with df.filter(F.col("amount").between(lower, upper)). Afterwards the DataFrame has 8% fewer rows than expected, although only 0.5% of amounts were outside the bounds. What is the most likely cause?

- **A.** between() excludes the boundary values
- **B.** Rows where amount is null were dropped, because comparisons with null are never true
- **C.** The filter ran on a cached DataFrame
- **D.** approxQuantile always returns the minimum and maximum

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — A filter keeps only rows where the condition is true; for null amounts it is null, so those rows disappear. Keep them explicitly with F.col("amount").isNull() | ... if they will be imputed.

Why the other options are wrong:

- **A**: between() is inclusive of both bounds.
- **C**: Caching does not change filter results.
- **D**: approxQuantile returns the requested quantiles.

Objective S2.02 · [Revisit](07-cleaning-and-transforming.md#removing-outliers)

</details>

### Question 3

The annual_income column of a customer table is heavily right-skewed: the median is 48,000 and the mean is 81,000 because of a few very high earners. 6% of values are missing. Which imputation is most appropriate?

- **A.** Mean, because it uses all values
- **B.** Median, because it is not pulled up by the long tail
- **C.** Mode, because income is continuous
- **D.** Zero, because missing income means no income

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — For skewed data the median represents a typical value; the mean is dragged towards the extreme incomes.

Why the other options are wrong:

- **A**: Using all values is exactly why the mean is distorted by the tail.
- **C**: The mode is for categorical or discrete data; continuous incomes rarely repeat.
- **D**: Missing does not mean zero, and zero would distort the distribution.

Objective S2.05 · [Revisit](07-cleaning-and-transforming.md#imputing-missing-values)

</details>

### Question 4

Which TWO statements about mean, median and mode imputation are correct?

- **A.** Mode imputation is the usual choice for categorical features
- **B.** Median imputation is more robust to outliers than mean imputation
- **C.** Mean imputation never changes the variance of a feature
- **D.** scikit-learn's SimpleImputer automatically picks the best strategy per column
- **E.** Median imputation can only be used on categorical data

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — Categories have no mean or median, so the most frequent value is used; the median ignores extreme values while the mean does not.

Why the other options are wrong:

- **C**: Filling many rows with one value shrinks the variance.
- **D**: SimpleImputer uses the strategy you specify (default "mean").
- **E**: The median requires ordered numeric values.

Objective S2.05 · [Revisit](07-cleaning-and-transforming.md#imputing-missing-values)

</details>

### Question 5

A data scientist wants to fill missing values in the numeric columns age and balance of a Spark DataFrame with each column's median, using a reusable step that can be part of a Spark ML pipeline. What should they use?

- **A.** Imputer(strategy="median", inputCols=["age", "balance"], outputCols=["age", "balance"])
- **B.** df.fillna("median")
- **C.** StringIndexer(inputCols=["age", "balance"])
- **D.** df.dropna(subset=["age", "balance"])

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Spark ML's Imputer is an estimator that learns per-column medians during fit() and fills them in transform(), so it fits into a Pipeline.

Why the other options are wrong:

- **B**: fillna fills a literal value; "median" would be inserted as text, not computed.
- **C**: StringIndexer encodes categories; it does not impute.
- **D**: dropna removes rows instead of imputing.

Objective S2.06 · [Revisit](07-cleaning-and-transforming.md#in-spark)

</details>

### Question 6

A Spark ML pipeline must turn the string column payment_method into a format a LogisticRegression can use. Which sequence of stages is correct?

- **A.** OneHotEncoder, then StringIndexer, then VectorAssembler
- **B.** StringIndexer, then OneHotEncoder, then VectorAssembler
- **C.** VectorAssembler only
- **D.** StandardScaler, then StringIndexer

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — StringIndexer maps strings to indices, OneHotEncoder turns indices into vectors, and VectorAssembler combines all features into the features column.

Why the other options are wrong:

- **A**: OneHotEncoder needs numeric indices, so it must come after StringIndexer.
- **C**: VectorAssembler cannot take string columns.
- **D**: StandardScaler expects a numeric vector, and no encoding into a vector happens.

Objective S2.07 · [Revisit](07-cleaning-and-transforming.md#one-hot-encoding)

</details>

### Question 7

A model predicts delivery delays and includes the feature postal_code with 12,000 distinct values. The team plans to one-hot encode it for a gradient-boosted tree model. What is the best advice?

- **A.** One-hot encoding is required for all tree models
- **B.** Avoid one-hot encoding here; use an index, target or frequency encoding, or group rare codes
- **C.** One-hot encode and then remove the tree model's depth limit
- **D.** Convert postal_code to a float and use it as a continuous feature without further thought

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — One-hot encoding a high-cardinality feature creates thousands of sparse columns; trees can split on compact encodings instead.

Why the other options are wrong:

- **A**: Trees do not need one-hot encoding and often work better without it.
- **C**: Unlimited depth increases overfitting and does not fix the width problem.
- **D**: Treating codes as numbers implies an order and distance that postal codes do not have.

Objective S2.08 · [Revisit](07-cleaning-and-transforming.md#when-is-one-hot-encoding-inappropriate)

</details>

### Question 8

For which model and feature is one-hot encoding most clearly appropriate?

- **A.** A logistic regression with the nominal feature contract_type (3 categories)
- **B.** A random forest with the feature customer_id (2 million values)
- **C.** A linear regression with the ordinal feature education_level, which must keep its order
- **D.** Any model with a continuous feature such as temperature

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Linear models need numeric inputs without an artificial order; a few nominal categories become a few binary columns.

Why the other options are wrong:

- **B**: An identifier with millions of values should not be one-hot encoded and is usually not a feature at all.
- **C**: One-hot encoding discards the order; an ordered integer mapping keeps it.
- **D**: Continuous features are not encoded.

Objective S2.08 · [Revisit](07-cleaning-and-transforming.md#when-is-one-hot-encoding-inappropriate)

</details>

### Question 9

In which TWO scenarios is a log transformation most appropriate?

- **A.** A target variable of household income with a long right tail
- **B.** Website session counts ranging from 1 to 2 million, where effects are multiplicative
- **C.** A symmetric, normally distributed temperature feature
- **D.** A categorical feature with three values
- **E.** A feature containing many negative values

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — Log transforms compress long right tails and turn multiplicative relations into additive ones for positive values spanning orders of magnitude.

Why the other options are wrong:

- **C**: Symmetric data does not need to be compressed.
- **D**: Logarithms do not apply to categories.
- **E**: The logarithm is undefined for negative values.

Objective S2.09 · [Revisit](07-cleaning-and-transforming.md#log-transformation)

</details>

### Question 10

A feature number_of_returns has many zeros and a long right tail. Applying np.log produces -inf values. What should the data scientist do?

- **A.** Use np.log1p, which computes log(1 + x) and is defined at 0
- **B.** Drop all rows with zero returns
- **C.** Replace -inf with the column's maximum
- **D.** Use np.exp instead

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — log1p handles zeros and still compresses the tail; invert it with expm1.

Why the other options are wrong:

- **B**: Zeros are valid and frequent; dropping them biases the data.
- **C**: Replacing -inf with the maximum turns the smallest values into the largest.
- **D**: exp stretches the tail further instead of compressing it.

Objective S2.09 · [Revisit](07-cleaning-and-transforming.md#log-transformation)

</details>
<!-- /questions -->
