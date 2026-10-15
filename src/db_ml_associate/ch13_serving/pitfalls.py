"""Chapter 13 pitfalls (constitution Principle V).

The ``wrong_*`` functions are deliberately incorrect — never copy them.
"""

from typing import Any

from databricks.sdk.service.serving import EndpointCoreConfigInput

from db_ml_associate.ch13_serving.workspace import ServedVersion, endpoint_config

INPUT_KEYS = ("dataframe_split", "dataframe_records", "instances", "inputs")
VERSIONS = [
    ServedVersion("champion", "ml_associate.tutorial.churn_model", "1"),
    ServedVersion("challenger", "ml_associate.tutorial.churn_model", "2"),
]


def wrong_traffic_split_not_100() -> EndpointCoreConfigInput:
    """⚠️ WRONG: 80 % + 30 % — the endpoint rejects routes that do not add up to 100 %."""
    return endpoint_config(VERSIONS, {"champion": 80, "challenger": 30})


def correct_traffic_split() -> EndpointCoreConfigInput:
    """90 % to the champion, 10 % to the challenger (a canary)."""
    return endpoint_config(VERSIONS, {"champion": 90, "challenger": 10})


def wrong_payload_without_input_key() -> dict[str, Any]:
    """⚠️ WRONG: a bare list of rows. Endpoints expect one of the documented input keys."""
    return {"data": [[12, 70.5, 3.1, 2, "one-year", "card"]]}


def correct_payload() -> dict[str, Any]:
    """``dataframe_split``: the recommended format for tabular custom models."""
    columns = ["tenure_months", "monthly_charges", "data_usage_gb", "support_calls", "contract_type", "payment_method"]
    return {"dataframe_split": {"columns": columns, "data": [[12, 70.5, 3.1, 2, "one-year", "card"]]}}


def payload_problems(payload: dict[str, Any]) -> list[str]:
    """Check that a request body uses exactly one of the documented input keys."""
    keys = [key for key in INPUT_KEYS if key in payload]
    if len(keys) != 1:
        return [f"request body must contain exactly one of {', '.join(INPUT_KEYS)}"]
    return []
