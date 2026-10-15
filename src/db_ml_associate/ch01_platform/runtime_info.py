"""Which runtime does the tutorial target, and which one is it running on? (exam objective S1.02)

Run ``runtime_info()`` at the top of a tutorial notebook: on Databricks the environment variable
``DATABRICKS_RUNTIME_VERSION`` is set, and the installed library versions should match the pins in
``runtime.toml``. A mismatch usually means the compute uses a different runtime version.
"""

import os
import tomllib
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

RUNTIME_FILE = Path(__file__).resolve().parents[3] / "runtime.toml"


@dataclass(frozen=True)
class RuntimeInfo:
    """Target runtime from ``runtime.toml`` and facts about the current environment."""

    target_runtime: str
    target_python: str
    target_libraries: dict[str, str]
    running_on_databricks: bool
    runtime_version: str | None
    installed: dict[str, str | None] = field(default_factory=dict)

    def mismatches(self) -> dict[str, tuple[str, str | None]]:
        """Libraries whose installed version differs from the target (None = not installed)."""
        return {
            name: (target, self.installed.get(name))
            for name, target in self.target_libraries.items()
            if self.installed.get(name) not in (target, None)
        }


def _installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def runtime_info(runtime_file: Path = RUNTIME_FILE) -> RuntimeInfo:
    """Read ``runtime.toml`` and inspect the current environment."""
    # --8<-- [start:runtime_info]
    target = tomllib.loads(runtime_file.read_text(encoding="utf-8"))
    runtime_version = os.environ.get("DATABRICKS_RUNTIME_VERSION")  # e.g. "18.0" on Databricks
    libraries = {str(name): str(version) for name, version in target["libraries"].items()}
    # --8<-- [end:runtime_info]
    return RuntimeInfo(
        target_runtime=str(target["databricks_runtime"]),
        target_python=str(target["python"]),
        target_libraries=libraries,
        running_on_databricks=runtime_version is not None,
        runtime_version=runtime_version,
        installed={name: _installed_version(name) for name in libraries},
    )
