"""Model Serving endpoints: create, query, split traffic, delete (S4.02, S4.05, S4.06).

Requires a workspace with Model Serving. The ``WorkspaceClient`` (``databricks-sdk``) and the MLflow
deployments client are passed in, so the code can be tested without a workspace. Endpoints cost money
while they run: use ``scale_to_zero_enabled`` for learning and delete endpoints afterwards.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from databricks.sdk.service.serving import EndpointCoreConfigInput, Route, ServedEntityInput, TrafficConfig

FULL_TRAFFIC = 100


@dataclass(frozen=True)
class ServedVersion:
    """A registered model version served under ``name`` on an endpoint."""

    name: str
    model_name: str
    version: str


def _served_entities(versions: Sequence[ServedVersion]) -> list[ServedEntityInput]:
    return [
        ServedEntityInput(
            name=served.name,  # e.g. "champion"; routes refer to this name
            entity_name=served.model_name,  # catalog.schema.model in Unity Catalog
            entity_version=served.version,
            workload_size="Small",  # Small / Medium / Large concurrency
            scale_to_zero_enabled=True,  # cheap for learning; avoid for latency-critical production
        )
        for served in versions
    ]


def _traffic(versions: Sequence[ServedVersion], traffic: Mapping[str, int]) -> TrafficConfig:
    names = {served.name for served in versions}
    if set(traffic) != names:
        raise ValueError(f"traffic must name every served model exactly once: {sorted(names)}")
    if sum(traffic.values()) != FULL_TRAFFIC:
        raise ValueError(f"traffic percentages must add up to 100, got {sum(traffic.values())}")
    return TrafficConfig(
        routes=[Route(served_model_name=name, traffic_percentage=share) for name, share in traffic.items()]
    )


def endpoint_config(versions: Sequence[ServedVersion], traffic: Mapping[str, int]) -> EndpointCoreConfigInput:
    """Serve one or more model versions and split requests between them by percentage."""
    # --8<-- [start:endpoint_config]
    return EndpointCoreConfigInput(
        served_entities=_served_entities(versions),  # e.g. champion v1 and challenger v2
        traffic_config=_traffic(versions, traffic),  # e.g. {"champion": 80, "challenger": 20}
    )
    # --8<-- [end:endpoint_config]


def create_endpoint(workspace: Any, name: str, config: EndpointCoreConfigInput) -> Any:
    """Create the endpoint and wait until it is ready (can take several minutes)."""
    # --8<-- [start:create_endpoint]
    # from databricks.sdk import WorkspaceClient; workspace = WorkspaceClient()
    return workspace.serving_endpoints.create_and_wait(name=name, config=config)
    # --8<-- [end:create_endpoint]


def update_traffic(workspace: Any, name: str, versions: Sequence[ServedVersion], traffic: Mapping[str, int]) -> Any:
    """Change the served versions or the split; Model Serving updates without downtime."""
    # --8<-- [start:update_traffic]
    return workspace.serving_endpoints.update_config_and_wait(
        name=name,
        served_entities=_served_entities(versions),
        traffic_config=_traffic(versions, traffic),
    )
    # --8<-- [end:update_traffic]


def dataframe_split_payload(frame: pd.DataFrame) -> dict[str, Any]:
    """The recommended request body: column names once, then the rows."""
    # --8<-- [start:dataframe_split]
    return {"dataframe_split": {"columns": list(frame.columns), "data": frame.to_numpy().tolist()}}
    # --8<-- [end:dataframe_split]


def query_endpoint(deploy_client: Any, name: str, frame: pd.DataFrame) -> list[Any]:
    """Send rows to the endpoint and return its predictions."""
    # --8<-- [start:query_endpoint]
    # import mlflow.deployments; deploy_client = mlflow.deployments.get_deploy_client("databricks")
    response = deploy_client.predict(endpoint=name, inputs=dataframe_split_payload(frame))
    return list(response["predictions"])
    # --8<-- [end:query_endpoint]


def delete_endpoint(workspace: Any, name: str) -> None:
    """Delete the endpoint so it stops costing money."""
    workspace.serving_endpoints.delete(name=name)
