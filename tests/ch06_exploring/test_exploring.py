"""Summary statistics, visualisations and feature comparisons (S2.01, S2.03, S2.04)."""

from unittest.mock import MagicMock

import matplotlib.figure
import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession
from scipy.stats import chi2_contingency

from db_ml_associate.ch06_exploring.compare import (
    chi_square_test,
    contingency_table,
    continuous_correlation,
    numeric_by_category,
)
from db_ml_associate.ch06_exploring.plots import bar_counts, box_by_category, histogram, scatter
from db_ml_associate.ch06_exploring.summary import (
    databricks_data_summary,
    pandas_summary,
    spark_summary,
    summary_examples,
)
from db_ml_associate.datasets import load_customer_churn


@pytest.fixture(scope="module")
def churn() -> pd.DataFrame:
    return load_customer_churn()


@pytest.fixture(scope="module")
def churn_spark(spark: SparkSession, churn: pd.DataFrame) -> DataFrame:
    return spark.createDataFrame(churn)


@pytest.mark.spark
def test_summary_returns_the_standard_statistics(churn_spark: DataFrame) -> None:
    summary = spark_summary(churn_spark, ["tenure_months", "support_calls"]).toPandas()
    assert list(summary["summary"]) == ["count", "mean", "stddev", "min", "25%", "50%", "75%", "max"]
    assert float(summary.set_index("summary").loc["count", "tenure_months"]) == 5000


@pytest.mark.spark
def test_summary_can_select_statistics(churn_spark: DataFrame) -> None:
    summary = spark_summary(churn_spark, ["tenure_months"], statistics=("count", "min", "max")).toPandas()
    assert list(summary["summary"]) == ["count", "min", "max"]


def test_pandas_summary_includes_categoricals(churn: pd.DataFrame) -> None:
    summary = pandas_summary(churn)
    assert "contract_type" in summary.columns
    assert "unique" in summary.index


def test_databricks_data_summary_calls_dbutils() -> None:
    dbutils, frame = MagicMock(), MagicMock()
    databricks_data_summary(dbutils, frame, precise=True)
    dbutils.data.summarize.assert_called_once_with(frame, precise=True)


@pytest.mark.spark
def test_pearson_and_spearman_match_pandas(churn: pd.DataFrame, churn_spark: DataFrame) -> None:
    pair = ["tenure_months", "support_calls"]
    for method in ("pearson", "spearman"):
        expected = churn[pair].corr(method=method).iloc[0, 1]
        assert continuous_correlation(churn_spark, *pair, method=method) == pytest.approx(expected, abs=1e-6)


@pytest.mark.spark
def test_contingency_table(churn: pd.DataFrame, churn_spark: DataFrame) -> None:
    table = contingency_table(churn_spark, "contract_type", "churned").toPandas().set_index("contract_type_churned")
    expected = pd.crosstab(churn["contract_type"], churn["churned"])
    assert int(table.loc["month-to-month", "1"]) == expected.loc["month-to-month", 1]


@pytest.mark.spark
def test_chi_square_matches_scipy(churn: pd.DataFrame, churn_spark: DataFrame) -> None:
    p_value = chi_square_test(churn_spark, "contract_type", "churned")
    _, expected, _, _ = chi2_contingency(pd.crosstab(churn["contract_type"], churn["churned"]))
    assert p_value == pytest.approx(expected, abs=1e-9)
    assert p_value < 0.001  # contract type and churn are clearly related


@pytest.mark.spark
def test_numeric_by_category(churn_spark: DataFrame) -> None:
    stats = numeric_by_category(churn_spark, "tenure_months", "contract_type").toPandas()
    assert set(stats["contract_type"]) == {"month-to-month", "one-year", "two-year"}
    assert {"mean_tenure_months", "median_tenure_months"} <= set(stats.columns)


@pytest.mark.parametrize(
    "plot,args",
    [
        (histogram, ("data_usage_gb",)),
        (bar_counts, ("contract_type",)),
        (box_by_category, ("tenure_months", "contract_type")),
        (scatter, ("tenure_months", "monthly_charges")),
    ],
)
def test_plots_return_figures(churn: pd.DataFrame, plot: object, args: tuple[str, ...]) -> None:
    figure = plot(churn, *args)  # type: ignore[operator]
    assert isinstance(figure, matplotlib.figure.Figure)
    assert figure.axes[0].get_title()


@pytest.mark.spark
def test_summary_examples(churn_spark: DataFrame) -> None:
    everything, chosen, basic = summary_examples(churn_spark.select("tenure_months"))
    assert everything.count() == 8
    assert chosen.count() == 5
    assert [row["summary"] for row in basic.collect()] == ["count", "mean", "stddev", "min", "max"]
