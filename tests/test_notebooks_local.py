"""Smoke-run the tutorial notebooks that need no workspace features, with local Spark and MLflow.

Notebooks that need AutoML, feature tables, Model Serving or a Lakeflow pipeline (chapters 2, 3, 13
and the last cell of chapter 12) can only run in a workspace; ``notebooks/00_run_workspace_tests.py``
and the tutorial verification (``book/exam/tutorial-verification.md``) cover them.
"""

import shutil
import subprocess  # nosec B404 - runs the notebook runner with the current interpreter
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("helpers") / "notebook_runner.py"
LOCAL_NOTEBOOKS = [
    ["notebooks/ch01_platform.py"],
    ["notebooks/ch04_tracking.py"],
    ["notebooks/ch05_registry.py", "notebooks/ch12_inference.py"],  # chapter 12 scores chapter 5's champion
    ["notebooks/ch06_exploring.py"],
    ["notebooks/ch07_cleaning.py"],
    ["notebooks/ch08_foundations.py"],
    ["notebooks/ch09_pipelines.py"],
    ["notebooks/ch10_tuning.py"],
    ["notebooks/ch11_evaluation.py"],
]
WORKSPACE_ONLY_CELLS = ['uc_name("churn_predictions")']  # output of the Lakeflow pipeline


@pytest.mark.slow
@pytest.mark.spark
@pytest.mark.parametrize("notebooks", LOCAL_NOTEBOOKS, ids=lambda group: "+".join(Path(n).stem for n in group))
def test_notebook_runs_locally(notebooks: list[str], tmp_path: Path) -> None:
    if shutil.which("java") is None:
        pytest.skip("Java 17+ not available")
    skips = [argument for text in WORKSPACE_ONLY_CELLS for argument in ("--skip-cell-containing", text)]
    result = subprocess.run(  # nosec B603
        [sys.executable, str(RUNNER), str(ROOT), str(tmp_path), *skips, *notebooks],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    for notebook in notebooks:
        assert f"PASSED {notebook}" in result.stdout
