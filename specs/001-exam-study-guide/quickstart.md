# Quickstart: Validating the Study Guide End to End

**Feature**: `001-exam-study-guide`. This guide proves the feature works; it does not contain implementation code. Commands refer to [contracts/cli.md](contracts/cli.md); formats to [contracts/content-formats.md](contracts/content-formats.md).

## Prerequisites

- Devcontainer (Python 3.12, Java 17+, uv 0.12.5) or equivalent local setup.
- For workspace validation: a Databricks workspace with Unity Catalog, permission to create a catalog/schema (default `ml_associate.tutorial`), and Databricks Runtime **18 LTS ML** (see `runtime.toml`). Model Serving / online store scenarios need a non-Free-Edition workspace.

## 1. Local quality gates

```bash
uv sync --all-groups
uv run pre-commit run --all-files          # ruff check, ruff format, mypy
uv run pytest                              # local + spark tests; databricks/network tests skipped
```

**Expected**: all hooks clean; pytest passes; summary shows `databricks` and `network` tests as *skipped* with reason "no Databricks workspace configured" / "network tests disabled" (SC-005).

## 2. Content validation

```bash
uv run db-ml-associate book check --strict
uv run db-ml-associate book sync-snippets --check
uv run db-ml-associate book coverage
```

**Expected**: `0 errors`; coverage table lists all 48 objectives, each with ≥ 1 chapter section and ≥ 1 question, and every practical objective with a tutorial step (SC-001); totals line reports ≥ 200 questions (SC-008); all 13 chapters (01–13, exam-guide order) pass structure checks and every docs URL is registered (SC-006); platform-change set complete (SC-007); both study plans within budget (SC-002).

## 3. Docs verification (network)

```bash
uv run pytest -m network --run-network
uv run db-ml-associate book check-links
```

**Expected**: every reference returns 200 at its recorded path; none older than 90 days. Then open `book/exam/docs-coverage.md` and confirm every listed docs sub-page has a decision.

## 4. Practice exam

```bash
uv run db-ml-associate exam take PE1 --no-timer
uv run db-ml-associate exam render PE2 --answers > /tmp/pe2-key.md
```

**Expected**: 48 questions; final report shows per-section scores for sections 1–4 with counts 18/9/15/6, readiness verdict at 80 %, and a chapter link per missed question (FR-017, FR-018).

## 5. Workspace tutorials (on 18 LTS ML)

1. Add the repo as a Git folder in the workspace.
2. Run `notebooks/00_setup.py` → creates catalog/schema and loads the committed `data/*.parquet` files into `customer_churn` and `house_prices` tables.
3. Run each `notebooks/chNN_*.py` top to bottom (Run all).
4. Run `notebooks/00_run_workspace_tests.py` → executes `pytest -m databricks` in the workspace.
5. Run `notebooks/99_cleanup.py` → deletes serving endpoints, online store/tables, registered models and the tutorial schema.

**Expected**: every notebook completes without errors and outputs match the book (SC-004); key checkpoints:

| Notebook | Checkpoint |
|---|---|
| ch03 | Feature table visible in Catalog Explorer with PK; `score_batch` returns predictions; online store published (paid workspaces) |
| ch04 | `MlflowClient.search_runs(order_by=[...])` returns the run with the best metric |
| ch05 | Model `ml_associate.tutorial.churn_model` has versions with tags; `@champion` alias moves to the challenger after comparison |
| ch07 | Imputed column has no nulls; outlier filter removes the injected outliers only |
| ch10 | Grid × CV run logs `combinations × folds` child runs; Hyperopt and Optuna both find comparable best params; on serverless the notebook falls back from `SparkTrials` |
| ch12 | Lakeflow pipeline produces a predictions streaming table |
| ch13 | Endpoint serves two versions with an 80/20 split; a `dataframe_split` query returns `predictions` |

## 6. Learner walkthrough (timing)

Follow `book/plans/fast-14.yaml` or `standard-21.yaml`; record actual minutes per day for one pilot learner. **Expected**: within ±20 % of plan estimates; practice exam scores ≥ 80 % per section by the final day (SC-002, SC-003).
