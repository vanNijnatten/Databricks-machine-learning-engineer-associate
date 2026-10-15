"""runtime_info() reports the target runtime and where the code is running."""

import pytest

from db_ml_associate.ch01_platform.runtime_info import runtime_info


def test_target_comes_from_runtime_toml() -> None:
    info = runtime_info()
    assert info.target_runtime == "18 LTS ML"
    assert info.target_python == "3.12.3"
    assert info.target_libraries["scikit-learn"] == "1.6.1"


def test_local_run_is_not_databricks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    info = runtime_info()
    assert info.running_on_databricks is False
    assert info.runtime_version is None


def test_databricks_runtime_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "18.0")
    info = runtime_info()
    assert info.running_on_databricks is True
    assert info.runtime_version == "18.0"


def test_installed_versions_match_pins_locally() -> None:
    info = runtime_info()
    assert info.installed["pandas"] == info.target_libraries["pandas"]
    assert info.mismatches() == {}
