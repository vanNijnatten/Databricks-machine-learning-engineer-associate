"""Chapter 3 pitfalls (constitution Principle V).

The ``wrong_*`` function is deliberately incorrect — never copy it.
"""

from collections.abc import Sequence

import pandas as pd

LEGACY_CLIENT_IMPORT = "from databricks.feature_store import FeatureStoreClient"


def wrong_feature_store_client() -> str:
    """⚠️ WRONG: the legacy Workspace Feature Store client.

    ``FeatureStoreClient`` belongs to the deprecated Workspace Feature Store (only available in
    workspaces created before 19 August 2024). Use ``FeatureEngineeringClient`` from
    ``databricks.feature_engineering`` with Unity Catalog instead.
    """
    return LEGACY_CLIENT_IMPORT


def primary_key_problems(frame: pd.DataFrame, keys: Sequence[str]) -> list[str]:
    """Check that ``keys`` can serve as a feature-table primary key: no nulls, no duplicates.

    Run this before ``create_table``/``write_table``: primary key columns must be NOT NULL and
    unique, otherwise lookups are ambiguous or the write fails.
    """
    # --8<-- [start:primary_key_problems]
    problems = []
    nulls = frame[list(keys)].isna().any(axis=1).sum()
    if nulls:
        problems.append(f"{nulls} rows have a null primary key")
    duplicates = frame.duplicated(subset=list(keys)).sum()
    if duplicates:
        problems.append(f"{duplicates} duplicate primary keys")
    # --8<-- [end:primary_key_problems]
    return problems
