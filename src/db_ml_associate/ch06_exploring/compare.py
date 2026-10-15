"""Comparing two features with the appropriate method (exam objective S2.04).

* Two continuous features → correlation (Pearson for linear, Spearman for monotonic relations) and
  a scatter plot.
* Two categorical features → a contingency table (cross-tabulation) and a chi-square test.
* One categorical and one continuous feature → statistics of the numeric feature per category
  and a box plot.
"""

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import ChiSquareTest, Correlation
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def continuous_correlation(frame: DataFrame, first: str, second: str, method: str = "pearson") -> float:
    """Correlation coefficient between two numeric columns (``pearson`` or ``spearman``)."""
    # --8<-- [start:correlation]
    if method == "pearson":
        return float(frame.stat.corr(first, second))  # DataFrame API supports Pearson only
    # Spearman (rank correlation) needs the vector-based MLlib API
    vectors = VectorAssembler(inputCols=[first, second], outputCol="pair").transform(frame.select(first, second))
    matrix = Correlation.corr(vectors, "pair", method="spearman").head()[0]
    return float(matrix[0, 1])
    # --8<-- [end:correlation]


def contingency_table(frame: DataFrame, first: str, second: str) -> DataFrame:
    """Counts of every combination of two categorical columns."""
    # --8<-- [start:crosstab]
    return frame.stat.crosstab(first, second)  # first column is named "<first>_<second>"
    # --8<-- [end:crosstab]


def chi_square_test(frame: DataFrame, category: str, label: str) -> float:
    """p-value of Pearson's chi-square independence test between a categorical feature and a label."""
    # --8<-- [start:chi_square]
    indexed = StringIndexer(inputCol=category, outputCol="category_index").fit(frame).transform(frame)
    vectors = VectorAssembler(inputCols=["category_index"], outputCol="features").transform(indexed)
    result = ChiSquareTest.test(vectors.withColumn(label, F.col(label).cast("double")), "features", label).head()
    p_value = float(result.pValues[0])  # small p-value: the two are not independent
    # --8<-- [end:chi_square]
    return p_value


def numeric_by_category(frame: DataFrame, numeric: str, category: str) -> DataFrame:
    """Mean, median and spread of a numeric column per category (the numbers behind a box plot)."""
    # --8<-- [start:numeric_by_category]
    return frame.groupBy(category).agg(
        F.count("*").alias("rows"),
        F.mean(numeric).alias(f"mean_{numeric}"),
        F.median(numeric).alias(f"median_{numeric}"),
        F.stddev(numeric).alias(f"stddev_{numeric}"),
    )
    # --8<-- [end:numeric_by_category]
