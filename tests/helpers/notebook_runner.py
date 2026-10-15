"""Run Databricks source notebooks locally: local Spark, a SQLite MLflow store and a stubbed ``dbutils``.

Used by ``tests/test_notebooks_local.py`` in a subprocess, so every run gets its own Spark session,
warehouse and MLflow store in a temporary directory. ``spark_catalog.default`` stands in for the
tutorial's Unity Catalog schema, and a local folder for ``/Volumes``. ``%pip`` cells are skipped
(the test environment already has the packages) and ``%run`` is followed.

Usage::

    python notebook_runner.py <repo_root> <work_dir> [--skip-cell-containing TEXT ...] <notebook> ...

Exit code: the number of notebooks that failed.
"""

import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

LOCAL_SCHEMA = {"catalog": "spark_catalog", "schema": "default"}
CELL_SEPARATOR = "# COMMAND ----------"


class NotebookExit(Exception):
    """Raised by the ``dbutils.notebook.exit`` stub."""


class _Widgets:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def text(self, name: str, default: str, label: str | None = None) -> None:
        self.values.setdefault(name, LOCAL_SCHEMA.get(name, default))

    def dropdown(self, name: str, default: str, choices: list[str], label: str | None = None) -> None:
        self.values.setdefault(name, default)

    def get(self, name: str) -> str:
        return self.values[name]


class _Data:
    def summarize(self, frame: Any, precise: bool = False) -> None:
        print("[dbutils.data.summarize]", list(frame.columns))


class _Library:
    def restartPython(self) -> None:
        """Nothing to restart locally."""


class _Fs:
    def __init__(self, volumes: Path) -> None:
        self.volumes = volumes

    def rm(self, path: str, recurse: bool = False) -> None:
        shutil.rmtree(path, ignore_errors=True)


class _Notebook:
    def exit(self, value: str) -> None:
        raise NotebookExit(value)


class DBUtils:
    """The subset of ``dbutils`` the tutorials use."""

    def __init__(self, volumes: Path) -> None:
        self.widgets = _Widgets()
        self.data = _Data()
        self.library = _Library()
        self.fs = _Fs(volumes)
        self.notebook = _Notebook()


def display(value: Any) -> None:
    """Print a short preview instead of rendering a table or figure."""
    shown = value.limit(5).toPandas() if hasattr(value, "limit") and hasattr(value, "toPandas") else value
    print("[display]", type(value).__name__, str(shown)[:300].replace("\n", " | "))


def run_notebook(path: Path, namespace: dict[str, Any], volumes: Path, skip: list[str]) -> None:
    """Execute every code cell of ``path`` in ``namespace``; follow ``%run`` and ignore other magics."""
    for number, cell in enumerate(path.read_text(encoding="utf-8").split(CELL_SEPARATOR)):
        lines = [line for line in cell.strip().splitlines() if line.strip() and not line.startswith("# Databricks")]
        if not lines:
            continue
        if all(line.startswith("# MAGIC") for line in lines):
            magic = lines[0].removeprefix("# MAGIC").strip()
            if magic.startswith("%run"):
                run_notebook((path.parent / f"{magic.split()[1]}.py").resolve(), namespace, volumes, skip)
            continue
        if any(text in cell for text in skip):
            print(f"  skipped cell {number} (workspace only)")
            continue
        code = cell.replace('f"/Volumes/{CATALOG}/{SCHEMA}/files', f'f"{volumes}')
        exec(compile(code, f"{path.name}[cell {number}]", "exec"), namespace)  # nosec B102


def _prepare(root: Path, work: Path) -> Any:
    """Start Spark and MLflow in ``work`` and load the tutorial tables like ``00_setup`` does."""
    import mlflow
    from pyspark.sql import SparkSession

    sys.path.insert(0, str(root / "src"))
    from db_ml_associate.ch06_exploring.pitfalls import nan_to_null
    from db_ml_associate.datasets import load_customer_churn, load_house_prices

    spark = (
        SparkSession.builder.master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.warehouse.dir", str(work / "warehouse"))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    for name, frame in (("customer_churn", load_customer_churn()), ("house_prices", load_house_prices())):
        floats = [column for column, dtype in frame.dtypes.items() if dtype.kind == "f"]
        nan_to_null(spark.createDataFrame(frame), floats).write.saveAsTable(f"spark_catalog.default.{name}")
    uri = f"sqlite:///{work / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_registry_uri = lambda _uri: None  # 00_common would switch to databricks-uc
    return spark


def main(argv: list[str]) -> int:
    """Run the notebooks named in ``argv``; return the number of failures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--skip-cell-containing", action="append", default=[])
    parser.add_argument("notebooks", nargs="+", type=Path)
    args = parser.parse_args(argv)
    root, work = args.root.resolve(), args.work.resolve()
    # 00_common finds src/ as ../src from the working directory; run from work/notebooks so that MLflow's
    # default ./mlruns artifact folder lands in the temporary directory, not in the repository.
    (work / "notebooks").mkdir(parents=True, exist_ok=True)
    if not (work / "src").exists():
        (work / "src").symlink_to(root / "src", target_is_directory=True)
    os.chdir(work / "notebooks")
    spark = _prepare(root, work)
    failures = 0
    for notebook in args.notebooks:
        path = (root / notebook).resolve()
        volumes = work / "volumes"
        namespace: dict[str, Any] = {
            "spark": spark,
            "dbutils": DBUtils(volumes),
            "display": display,
            "__name__": "__main__",
        }
        try:
            run_notebook(path, namespace, volumes, args.skip_cell_containing)
            print(f"PASSED {notebook}")
        except NotebookExit as stop:
            print(f"EXITED {notebook}: {stop}")
        except Exception:  # noqa: BLE001 - report every notebook failure, then continue
            failures += 1
            traceback.print_exc()
            print(f"FAILED {notebook}")
    spark.stop()
    return failures


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
