"""Chapter 3 pitfalls: invalid primary keys and the legacy FeatureStoreClient."""

import pandas as pd

from db_ml_associate.ch03_features.pitfalls import (
    LEGACY_CLIENT_IMPORT,
    primary_key_problems,
    wrong_feature_store_client,
)


def test_valid_key_has_no_problems() -> None:
    assert primary_key_problems(pd.DataFrame({"customer_id": [1, 2, 3]}), ["customer_id"]) == []


def test_null_keys_are_rejected() -> None:
    problems = primary_key_problems(pd.DataFrame({"customer_id": [1, None, 3]}), ["customer_id"])
    assert any("null" in problem for problem in problems)


def test_duplicate_keys_are_rejected() -> None:
    problems = primary_key_problems(pd.DataFrame({"customer_id": [1, 1, 3]}), ["customer_id"])
    assert any("duplicate" in problem for problem in problems)


def test_composite_keys() -> None:
    frame = pd.DataFrame({"customer_id": [1, 1], "day": ["2026-01-01", "2026-01-02"]})
    assert primary_key_problems(frame, ["customer_id", "day"]) == []


def test_legacy_client_is_documented_as_deprecated() -> None:
    assert "databricks.feature_store" in LEGACY_CLIENT_IMPORT
    assert "deprecated" in (wrong_feature_store_client.__doc__ or "").lower()
