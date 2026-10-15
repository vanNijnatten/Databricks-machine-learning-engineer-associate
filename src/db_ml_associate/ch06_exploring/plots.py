"""Matplotlib visualisations for categorical and continuous features (exam objective S2.03).

In a Databricks notebook you can get the same charts without code: ``display(df)`` and then
**+ → Visualization** in the results pane. These functions are the code equivalent and work on a
pandas DataFrame (convert a Spark DataFrame with ``.toPandas()`` on a sample, or aggregate first).
"""

import matplotlib

matplotlib.use("Agg")  # render without a display (tests, jobs)

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def histogram(frame: pd.DataFrame, column: str, bins: int = 40) -> Figure:
    """Distribution of one continuous feature: shape, skew, outliers."""
    # --8<-- [start:histogram]
    figure, axes = plt.subplots(figsize=(6, 4))
    axes.hist(frame[column].dropna(), bins=bins)
    axes.set(title=f"Distribution of {column}", xlabel=column, ylabel="rows")
    # --8<-- [end:histogram]
    return figure


def bar_counts(frame: pd.DataFrame, column: str) -> Figure:
    """Frequency of each category of one categorical feature."""
    # --8<-- [start:bar_counts]
    counts = frame[column].value_counts()
    figure, axes = plt.subplots(figsize=(6, 4))
    axes.bar(counts.index.astype(str), counts.to_numpy())
    axes.set(title=f"Rows per {column}", xlabel=column, ylabel="rows")
    # --8<-- [end:bar_counts]
    return figure


def box_by_category(frame: pd.DataFrame, numeric: str, category: str) -> Figure:
    """A continuous feature split by a categorical one: compare medians and spread."""
    # --8<-- [start:box_by_category]
    groups = {name: group[numeric].dropna() for name, group in frame.groupby(category)}
    figure, axes = plt.subplots(figsize=(6, 4))
    axes.boxplot(list(groups.values()), tick_labels=list(groups))
    axes.set(title=f"{numeric} by {category}", xlabel=category, ylabel=numeric)
    # --8<-- [end:box_by_category]
    return figure


def scatter(frame: pd.DataFrame, x: str, y: str) -> Figure:
    """Two continuous features against each other: direction, strength and shape of a relation."""
    figure, axes = plt.subplots(figsize=(6, 4))
    axes.scatter(frame[x], frame[y], s=4, alpha=0.4)
    axes.set(title=f"{y} vs {x}", xlabel=x, ylabel=y)
    return figure
