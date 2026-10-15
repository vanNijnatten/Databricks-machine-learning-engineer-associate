"""Column lists and helpers shared by the tutorials."""

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_SEED = 42

CHURN_KEY = "customer_id"
CHURN_LABEL = "churned"
CHURN_NUMERIC = ("tenure_months", "monthly_charges", "data_usage_gb", "support_calls")
CHURN_CATEGORICAL = ("contract_type", "payment_method")
CHURN_FEATURES = (*CHURN_NUMERIC, *CHURN_CATEGORICAL)  # model inputs; keys and `city` are not features

HOUSE_KEY = "house_id"
HOUSE_TARGET = "price"
HOUSE_NUMERIC = (
    "living_area_sqm",
    "bedrooms",
    "year_built",
    "has_garage",
    "lot_size_sqm",
)
HOUSE_CATEGORICAL = ("neighborhood", "condition")


def split_frame(
    frame: pd.DataFrame,
    label: str,
    test_size: float = 0.2,
    seed: int = DEFAULT_SEED,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test parts, stratified on ``label`` for classification.

    Split *before* any step that learns from the data (imputation, scaling, resampling) so that
    the test rows never influence training.
    """
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame[label] if stratify else None,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
