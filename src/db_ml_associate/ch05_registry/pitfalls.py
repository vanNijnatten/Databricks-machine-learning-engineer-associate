"""Chapter 5 pitfalls (constitution Principle V).

The ``wrong_*`` functions are deliberately incorrect — never copy them.
"""

import mlflow
from mlflow import MlflowClient


def _registry_version_of(uri: str, name: str) -> str:
    """Load the model behind ``uri`` and return the registry version it came from."""
    model = mlflow.pyfunc.load_model(uri)
    run_id = model.metadata.run_id
    versions = MlflowClient().search_model_versions(f"name = '{name}'")
    return next(str(version.version) for version in versions if version.run_id == run_id)


def wrong_load_by_version_number(uri: str) -> str:
    """⚠️ WRONG: hard-code ``models:/<name>/<version>`` in a production job.

    After a promotion the job keeps loading the old version until someone edits and redeploys it.
    Returns the registry version that was actually loaded.
    """
    name = uri.removeprefix("models:/").rsplit("/", 1)[0]
    return _registry_version_of(uri, name)


def correct_load_by_alias(name: str, alias: str) -> str:
    """Load ``models:/<name>@<alias>``: a promotion (moving the alias) needs no code change."""
    return _registry_version_of(f"models:/{name}@{alias}", name)


def wrong_transition_stage(client: MlflowClient, name: str, version: str) -> None:
    """⚠️ WRONG: workspace-registry stages. Stages are not supported for models in Unity Catalog.

    ``transition_model_version_stage`` belongs to the legacy workspace model registry; use the
    environment in the three-level name (catalog) and aliases such as ``champion`` instead.
    """
    client.transition_model_version_stage(name, version, stage="Production")
