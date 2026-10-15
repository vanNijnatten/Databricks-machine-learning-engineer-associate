"""Shared pytest configuration: Spark fixture and automatic skipping of optional test groups.

Markers (declared in pyproject.toml):

- ``spark``: needs a local Spark session; skipped when Java 17+ is not installed.
- ``databricks``: needs a Databricks workspace; skipped unless ``DATABRICKS_HOST`` or
  ``DATABRICKS_RUNTIME_VERSION`` is set.
- ``network``: needs internet access; skipped unless ``--run-network`` or ``RUN_NETWORK_TESTS=1``.
"""

import os
import re
import shutil
import subprocess  # nosec B404 - only used to query the local Java version
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

MINIMUM_JAVA_VERSION = 17


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-network`` command-line option."""
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked 'network' (they access the internet)",
    )


def _java_major_version() -> int | None:
    """Return the major version of the local ``java`` executable, or None if unavailable."""
    java = shutil.which("java")
    if java is None:
        return None
    result = subprocess.run([java, "-version"], capture_output=True, text=True, check=False)  # nosec B603
    match = re.search(r'version "(\d+)', result.stderr + result.stdout)
    return int(match.group(1)) if match else None


def _workspace_configured() -> bool:
    return bool(os.environ.get("DATABRICKS_HOST") or os.environ.get("DATABRICKS_RUNTIME_VERSION"))


def _network_enabled(config: pytest.Config) -> bool:
    return bool(config.getoption("--run-network")) or os.environ.get("RUN_NETWORK_TESTS") == "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``databricks`` and ``network`` tests when their environment is not available."""
    skip_workspace = pytest.mark.skip(reason="no Databricks workspace configured")
    skip_network = pytest.mark.skip(reason="network tests disabled (use --run-network)")
    workspace = _workspace_configured()
    network = _network_enabled(config)
    for item in items:
        if "databricks" in item.keywords and not workspace:
            item.add_marker(skip_workspace)
        if "network" in item.keywords and not network:
            item.add_marker(skip_network)


@pytest.fixture(scope="session")
def spark() -> Iterator["SparkSession"]:
    """A small, deterministic local Spark session shared by all ``spark`` tests."""
    java_version = _java_major_version()
    if java_version is None or java_version < MINIMUM_JAVA_VERSION:
        pytest.skip("Java 17+ not available")
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("db-ml-associate-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture
def local_mlflow(tmp_path: Path) -> Iterator[str]:
    """Point MLflow tracking and the model registry at a throw-away SQLite database.

    Yields the name of a fresh experiment whose artifacts live under ``tmp_path``.
    """
    import mlflow

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    previous_tracking, previous_registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    name = "tests"
    mlflow.create_experiment(name, artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment(name)
    yield name
    mlflow.end_run()
    mlflow.set_tracking_uri(previous_tracking)
    mlflow.set_registry_uri(previous_registry)
