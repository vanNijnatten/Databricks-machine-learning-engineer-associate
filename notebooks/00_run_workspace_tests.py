# Databricks notebook source
# MAGIC %md
# MAGIC # Run the workspace tests
# MAGIC
# MAGIC Runs the tests marked `databricks` (real Unity Catalog feature tables, AutoML, …) inside the workspace, where
# MAGIC `DATABRICKS_RUNTIME_VERSION` is set so they are not skipped. The notebook fails if any test fails.
# MAGIC Use classic compute with Databricks Runtime 18 LTS ML (Dedicated access mode) and run `00_setup` first.

# COMMAND ----------

# MAGIC %pip install pytest==9.0.1 pytest-mock==3.15.1 hyperopt==0.2.7 "typer>=0.20.0" "rich>=14.2.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

import os

import pytest

sys.dont_write_bytecode = True  # the Git folder is not a place for __pycache__ files
os.chdir(REPO_ROOT)  # pytest reads its configuration from pyproject.toml
exit_code = pytest.main(["-m", "databricks", "-p", "no:cacheprovider", "-v", "tests"])
print("pytest exit code:", exit_code)

# COMMAND ----------

if exit_code not in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
    raise RuntimeError(f"workspace tests failed (pytest exit code {exit_code})")
