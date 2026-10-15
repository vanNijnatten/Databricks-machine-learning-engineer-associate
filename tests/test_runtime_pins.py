"""pyproject.toml pins and the local Python version match runtime.toml (Principle II)."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([^;\s]+)")


def pyproject_pins() -> dict[str, str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements: list[str] = list(pyproject["project"]["dependencies"])
    for group in pyproject.get("dependency-groups", {}).values():
        requirements.extend(item for item in group if isinstance(item, str))
    pins: dict[str, str] = {}
    for requirement in requirements:
        match = PIN.match(requirement)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def runtime() -> dict[str, object]:
    return tomllib.loads((ROOT / "runtime.toml").read_text())


def test_runtime_libraries_are_pinned_identically() -> None:
    pins = pyproject_pins()
    libraries = runtime()["libraries"]
    assert isinstance(libraries, dict)
    mismatches = {
        name: (version, pins[name.lower()])
        for name, version in libraries.items()
        if name.lower() in pins and pins[name.lower()] != version
    }
    assert mismatches == {}


def test_every_runtime_library_is_pinned_somewhere() -> None:
    pins = pyproject_pins()
    libraries = runtime()["libraries"]
    assert isinstance(libraries, dict)
    assert sorted(name for name in libraries if name.lower() not in pins) == []


def test_local_python_matches_runtime_minor_version() -> None:
    local = (ROOT / ".python-version").read_text().strip()
    target = str(runtime()["python"])
    assert local.split(".")[:2] == target.split(".")[:2]
