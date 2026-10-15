"""Deterministic synthetic datasets used throughout the tutorials.

Two datasets cover every phenomenon the exam asks about:

``customer_churn`` (classification)
    About 10 % churners (class imbalance), a low-cardinality and a high-cardinality categorical
    column, ~5 % missing ``monthly_charges``, a right-skewed ``data_usage_gb`` with missing values
    and outliers in ``support_calls``.

``house_prices`` (regression)
    A right-skewed ``price`` target that becomes roughly symmetric after a log transform, a
    categorical ``neighborhood``, missing values and outliers in ``lot_size_sqm``.

The generated data is committed under ``data/`` (see ``data/README.md``). Tutorials load those
files with :func:`load_customer_churn` and :func:`load_house_prices`; a test proves that the
generators reproduce them exactly.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SEED = 42
CHURN_ROWS = 5000
HOUSE_ROWS = 3000
CHURN_RATE = 0.10

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CHURN_FILE = DATA_DIR / "customer_churn.parquet"
HOUSE_FILE = DATA_DIR / "house_prices.parquet"

CONTRACT_TYPES = ("month-to-month", "one-year", "two-year")
PAYMENT_METHODS = ("card", "bank_transfer", "electronic_check", "mailed_check")
NEIGHBORHOODS = (
    "Brookside",
    "Cedar Park",
    "Downtown",
    "Elm Grove",
    "Hillcrest",
    "Lakeside",
    "Maple Heights",
    "Northgate",
    "Oakwood",
    "Riverside",
    "Southfield",
    "Westend",
)
CONDITIONS = ("poor", "fair", "good", "excellent")
NUMBER_OF_CITIES = 800


def _set_missing(values: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Return a float copy of ``values`` with exactly ``fraction`` of the entries set to NaN."""
    result = values.astype(float)
    missing = rng.choice(len(result), size=round(fraction * len(result)), replace=False)
    result[missing] = np.nan
    return result


def make_customer_churn(n_rows: int = CHURN_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Generate the customer churn dataset (one row per customer, label ``churned``)."""
    rng = np.random.default_rng(seed)
    tenure_months = rng.integers(1, 73, size=n_rows)
    contract_type = rng.choice(CONTRACT_TYPES, size=n_rows, p=[0.55, 0.25, 0.20])
    payment_method = rng.choice(PAYMENT_METHODS, size=n_rows, p=[0.35, 0.25, 0.25, 0.15])
    city_weights = 1 / np.arange(1, NUMBER_OF_CITIES + 1) ** 0.5
    city_index = rng.choice(NUMBER_OF_CITIES, size=n_rows, p=city_weights / city_weights.sum())
    city = np.array([f"city_{index:04d}" for index in city_index])
    monthly_charges = np.round(rng.normal(65, 20, size=n_rows).clip(18, 120), 2)
    data_usage_gb = np.round(rng.lognormal(mean=2.5, sigma=0.8, size=n_rows), 2)
    support_calls = rng.poisson(1.5, size=n_rows)
    outliers = rng.choice(n_rows, size=max(1, round(0.004 * n_rows)), replace=False)
    support_calls[outliers] = rng.integers(15, 26, size=len(outliers))

    logit = (
        -2.2
        + 1.2 * (contract_type == "month-to-month")
        - 0.04 * tenure_months
        + 0.35 * np.minimum(support_calls, 8)
        + 0.012 * (monthly_charges - 65)
        + 0.3 * (payment_method == "electronic_check")
    )
    score = logit + rng.logistic(size=n_rows)
    churned = np.zeros(n_rows, dtype=np.int64)
    churned[np.argsort(-score)[: round(CHURN_RATE * n_rows)]] = 1

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n_rows + 1, dtype=np.int64),
            "tenure_months": tenure_months.astype(np.int64),
            "contract_type": contract_type.astype(object),
            "payment_method": payment_method.astype(object),
            "city": city.astype(object),
            "monthly_charges": _set_missing(monthly_charges, 0.05, rng),
            "data_usage_gb": _set_missing(data_usage_gb, 0.04, rng),
            "support_calls": support_calls.astype(np.int64),
            "churned": churned,
        }
    )


def make_house_prices(n_rows: int = HOUSE_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Generate the house prices dataset (one row per sold house, target ``price``)."""
    rng = np.random.default_rng(seed)
    neighborhood_effect = dict(zip(NEIGHBORHOODS, rng.normal(0, 0.25, size=len(NEIGHBORHOODS)), strict=True))
    condition_effect = {"poor": -0.20, "fair": -0.05, "good": 0.05, "excellent": 0.18}

    neighborhood = rng.choice(NEIGHBORHOODS, size=n_rows)
    living_area_sqm = np.round(rng.normal(120, 35, size=n_rows).clip(35, 400), 1)
    bedrooms = np.clip(np.round(living_area_sqm / 35 + rng.normal(0, 0.7, size=n_rows)), 1, 7).astype(np.int64)
    year_built = rng.integers(1900, 2025, size=n_rows)
    has_garage = (rng.random(size=n_rows) < 0.6).astype(np.int64)
    condition = rng.choice(CONDITIONS, size=n_rows, p=[0.1, 0.3, 0.45, 0.15])
    lot_size_sqm = np.round(rng.lognormal(mean=6.0, sigma=0.5, size=n_rows), 1)
    huge_lots = rng.choice(n_rows, size=max(1, round(0.005 * n_rows)), replace=False)
    lot_size_sqm[huge_lots] *= 10

    log_price = (
        11.0
        + 0.0065 * living_area_sqm
        + np.array([neighborhood_effect[name] for name in neighborhood])
        + 0.08 * has_garage
        + np.array([condition_effect[name] for name in condition])
        + 0.002 * (year_built - 1960)
        + rng.normal(0, 0.25, size=n_rows)
    )
    price = np.round(np.exp(log_price), -2)

    return pd.DataFrame(
        {
            "house_id": np.arange(1, n_rows + 1, dtype=np.int64),
            "neighborhood": neighborhood.astype(object),
            "living_area_sqm": living_area_sqm,
            "bedrooms": bedrooms,
            "year_built": year_built.astype(np.int64),
            "has_garage": has_garage,
            "condition": condition.astype(object),
            "lot_size_sqm": _set_missing(lot_size_sqm, 0.05, rng),
            "price": price,
        }
    )


def load_customer_churn() -> pd.DataFrame:
    """Load the committed ``data/customer_churn.parquet``."""
    return pd.read_parquet(CHURN_FILE)


def load_house_prices() -> pd.DataFrame:
    """Load the committed ``data/house_prices.parquet``."""
    return pd.read_parquet(HOUSE_FILE)
