"""Models in Unity Catalog: register, tag and promote with aliases (S1.14, S1.15, S1.17, S1.18).

On Databricks with MLflow 3 the registry URI defaults to Unity Catalog (``databricks-uc``); on older
setups call ``mlflow.set_registry_uri("databricks-uc")`` first. Registered model names have three
levels: ``catalog.schema.model``. Every new version needs a model signature.
"""

from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

UC_NAME_PARTS = 3


def validate_uc_name(name: str) -> str:
    """Raise ``ValueError`` unless ``name`` has the form ``catalog.schema.model``."""
    if len(name.split(".")) != UC_NAME_PARTS or not all(name.split(".")):
        raise ValueError(f"Unity Catalog model names must be catalog.schema.model, got {name!r}")
    return name


def register_model(model_uri: str, name: str) -> ModelVersion:
    """Register a logged model (``models:/<model_id>`` or ``runs:/<run_id>/model``) as a new version."""
    validate_uc_name(name)
    # --8<-- [start:register_model]
    # mlflow.set_registry_uri("databricks-uc")  # default in MLflow 3 on Databricks
    version = mlflow.register_model(model_uri=model_uri, name=name)  # name = catalog.schema.model
    # alternative: mlflow.sklearn.log_model(model, name="model", registered_model_name=name, ...)
    # --8<-- [end:register_model]
    return version


def set_model_tag(client: MlflowClient, name: str, key: str, value: str) -> None:
    """Tag the registered model (all versions)."""
    # --8<-- [start:model_tags]
    client.set_registered_model_tag(name, key, value)  # e.g. ("team", "retention")
    # --8<-- [end:model_tags]


def remove_model_tag(client: MlflowClient, name: str, key: str) -> None:
    """Remove a tag from the registered model."""
    # --8<-- [start:remove_model_tag]
    client.delete_registered_model_tag(name, key)
    # --8<-- [end:remove_model_tag]


def set_version_tag(client: MlflowClient, name: str, version: str, key: str, value: str) -> None:
    """Tag one model version."""
    # --8<-- [start:version_tags]
    client.set_model_version_tag(name, version, key, value)  # e.g. ("validation_status", "approved")
    # --8<-- [end:version_tags]


def remove_version_tag(client: MlflowClient, name: str, version: str, key: str) -> None:
    """Remove a tag from one model version."""
    # --8<-- [start:remove_version_tag]
    client.delete_model_version_tag(name, version, key)
    # --8<-- [end:remove_version_tag]


def set_alias(client: MlflowClient, name: str, alias: str, version: str) -> None:
    """Point an alias (e.g. ``champion``) at a version; an alias names exactly one version."""
    # --8<-- [start:set_alias]
    client.set_registered_model_alias(name, alias, version)  # moves the alias if it already exists
    # --8<-- [end:set_alias]


def load_by_alias(name: str, alias: str) -> Any:
    """Load whichever version the alias currently points to."""
    # --8<-- [start:load_by_alias]
    model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")  # e.g. models:/prod.ml.churn@champion
    # --8<-- [end:load_by_alias]
    return model


def _metric(client: MlflowClient, version: ModelVersion, metric: str) -> float:
    run_id = version.run_id
    if run_id is None:
        raise ValueError(f"model version {version.version} has no source run")
    return float(client.get_run(run_id).data.metrics[metric])


def promote_challenger(client: MlflowClient, name: str, metric: str, higher_is_better: bool = True) -> bool:
    """Make the challenger the champion if it scores better on ``metric``; return True if promoted."""
    # --8<-- [start:promote_challenger]
    champion = client.get_model_version_by_alias(name, "champion")
    challenger = client.get_model_version_by_alias(name, "challenger")
    champion_score = _metric(client, champion, metric)
    challenger_score = _metric(client, challenger, metric)
    better = challenger_score > champion_score if higher_is_better else challenger_score < champion_score
    if better:
        client.set_registered_model_alias(name, "champion", str(challenger.version))  # instant switch
        client.delete_registered_model_alias(name, "challenger")
    # --8<-- [end:promote_challenger]
    return better


def copy_to_environment(client: Any, source_name: str, version: str, target_name: str) -> Any:
    """Deploy-models pattern: copy a version to another catalog, e.g. staging → prod."""
    # --8<-- [start:copy_model_version]
    return client.copy_model_version(src_model_uri=f"models:/{source_name}/{version}", dst_name=target_name)
    # --8<-- [end:copy_model_version]
