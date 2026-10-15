# Tutorial verification record

Results of running the quickstart (`specs/001-exam-study-guide/quickstart.md`) and the tutorial
notebooks. Update this page after every runtime upgrade (see the runtime upgrade checklist in
`docs/CONTRIBUTING.md`).

## 2026-10-15 — local verification (devcontainer)

Environment: Python 3.12, pinned libraries from `runtime.toml` (pyspark 4.1.0, mlflow 3.8.1,
scikit-learn 1.6.1), Java 17, local Spark `local[2]`, SQLite MLflow tracking and registry.

| Quickstart section | Result |
|---|---|
| 1. Local quality gates | `ruff check` and `ruff format`: clean. `mypy src`: no issues. `bandit -r src`: no issues. `pytest --cov=src`: 408 passed, 3 skipped (`databricks`/`network` markers), 96 % line coverage; a teardown error of the shared Spark session after the last test was fixed in `tests/conftest.py` and the rerun passed. Inside `pre-commit run --all-files` the mypy hook was killed for memory (exit 137) because pre-commit starts several whole-`src` mypy runs in parallel; every run that completed reported no issues. |
| 2. Content validation | `book check --strict`: 0 errors, 0 warnings. `book sync-snippets --check`: 0 problems. `book coverage`: 48/48 objectives with a chapter section and a question; every practical objective has a tutorial step. Question bank 217 (121 review, 96 practice). |
| 3. Docs verification | `book check-links`: 43/43 URLs answer 200 at their recorded path, after updating two moved pages (`notebooks/visualizations` → `visualizations`; "How AutoML works" folded into the AutoML page). Docs-coverage sweep recorded in [docs-coverage.md](docs-coverage.md). |
| 4. Practice exam | `exam take PE1 --no-timer` (answers piped in): 48 questions, per-section scores for sections 1–4 with 18/9/15/6 questions, 80 % readiness verdicts and a chapter link per missed question. `exam render PE2 --answers`: 48 explanations. |
| 5. Workspace tutorials | **Not run yet** — needs a Databricks workspace on Runtime 18 LTS ML (task T130). See the local notebook runs below. |

### Tutorial notebooks run locally

`tests/test_notebooks_local.py` executes the notebooks cell by cell with local Spark, a SQLite MLflow
store and a stub for `dbutils`, with `spark_catalog.default` standing in for `ml_associate.tutorial`.

| Notebook | Local result | Not exercised locally |
|---|---|---|
| ch01_platform | passed | Catalog Explorer tour (UI) |
| ch02_automl | not run (AutoML needs a workspace) | everything |
| ch03_features | not run (feature engineering client needs a workspace) | everything |
| ch04_tracking | passed | MLflow UI tour (UI) |
| ch05_registry | passed | Unity Catalog registry (SQLite registry used; stages do not fail there) |
| ch06_exploring | passed | `dbutils.data.summarize` rendering, visualization editor (UI) |
| ch07_cleaning | passed | — |
| ch08_foundations | passed | — |
| ch09_pipelines | passed | — |
| ch10_tuning | passed, including `SparkTrials` (16 trials) | serverless fallback path |
| ch11_evaluation | passed | — |
| ch12_inference | passed up to the Lakeflow pipeline | Lakeflow pipeline and its output table, `ai_query` |
| ch13_serving | not run (Model Serving needs a workspace) | everything |

Defects found and fixed by these runs: tables created from pandas stored missing values as NaN
instead of null, so Spark's `mean` returned NaN and null counts were 0 — `00_setup` now converts NaN to
null; the chapter 4 model signature included `customer_id` and `city`, which would have broken
scoring with the six feature columns in chapters 12 and 13 — the signature now lists only the model's
features; the chapter 12 pipeline's own `customer_events` table would have collided with its source
table — the source is now `customer_events_raw`.

## Workspace verification (open)

To do on Runtime 18 LTS ML (classic, Dedicated) and on Free Edition (serverless), per task T130:
run every notebook, record outcome, actual run time and Free Edition status (`full`, `partial`,
`read_through`), confirm the `SparkTrials` fallback on serverless, and mark documentation-vs-behaviour
differences in `book/references.yaml` as `status: discrepancy`.

| Notebook | 18 LTS ML (classic) | Free Edition (serverless) | Minutes |
|---|---|---|---|
| 00_setup | | | |
| ch01–ch13 | | | |
| 00_run_workspace_tests | | | |
| 99_cleanup | | | |
