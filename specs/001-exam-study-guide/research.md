# Phase 0 Research: Exam-Aligned Study Guide and Tutorial

**Feature**: `001-exam-study-guide` | **Date**: 2026-10-14 | **Docs snapshot**: docs.databricks.com/aws/en as fetched on 2026-10-14

Each entry follows *Decision / Rationale / Alternatives considered*. Facts about Databricks cite the page they came from; those citations seed `book/references.yaml` (see [contracts/references.md](contracts/references.md)).

---

## R1. Target runtime and library versions

**Decision**: Target **Databricks Runtime 18 LTS for Machine Learning** (released 2026-06-10). Record in `runtime.toml` (single source of truth, Principle II):

| Component | Version | Source |
|---|---|---|
| Python | 3.12.3 | [DBR 18 LTS](https://docs.databricks.com/aws/en/release-notes/runtime/18) |
| Apache Spark | 4.1.0 | [DBR 18 LTS](https://docs.databricks.com/aws/en/release-notes/runtime/18) |
| mlflow-skinny | 3.8.1 | [DBR 18 LTS ML](https://docs.databricks.com/aws/en/release-notes/runtime/18ml) |
| scikit-learn | 1.6.1 | DBR 18 LTS ML |
| pandas | 2.2.3 | DBR 18 LTS ML |
| numpy | 2.1.3 | DBR 18 LTS ML |
| xgboost | 3.1.1 | DBR 18 LTS ML |
| lightgbm | 4.6.0 | DBR 18 LTS ML |
| optuna | 3.6.1 | DBR 18 LTS ML |
| databricks-feature-engineering | 0.13.0.1 | DBR 18 LTS ML |
| databricks-sdk | 0.67.0 | DBR 18 LTS ML |
| ray | 2.37.0 | DBR 18 LTS ML |
| hyperopt | **not included** (install 0.2.7 via `%pip`) | [Hyperparameter tuning](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/) |
| AutoML | **not built in from 18.0 ML** (`databricks-automl-runtime` package; classic Dedicated compute) | [AutoML](https://docs.databricks.com/aws/en/machine-learning/automl/), [18.0 ML notes](https://docs.databricks.com/aws/en/release-notes/runtime/18.0ml) |

**Rationale**: Constitution Principle II mandates the latest LTS. 19 is newer but not LTS.

**Alternatives considered**: 17.3 LTS ML (still has AutoML built in, MLflow 3.0.1) — rejected: not latest LTS; the AutoML gap is handled by an install step and a "platform change" note.

## R2. Local Python version (resolves constitution follow-up TODO)

**Decision**: Change the project to **Python 3.12** (`requires-python = ">=3.12,<3.13"`, `.python-version = 3.12`, devcontainer base `python:3.12`, ruff `target-version = "py312"`, mypy `python_version = "3.12"`).

**Rationale**: The repo currently declares `>=3.14`, but tutorial code must not use features unavailable on the runtime's Python 3.12.3. Matching the local interpreter makes ruff/mypy enforce this automatically and lets tests install the same pinned libraries as the runtime.

**Alternatives considered**: Keep 3.14 locally and rely on `ruff target-version=py312` — rejected: pinned runtime libraries (pandas 2.2.3, numpy 2.1.3, scikit-learn 1.6.1) do not ship 3.14 wheels; tests would run against different versions than learners use.

## R3. Dependency layout

**Decision**:
- `[project].dependencies`: what the learner-facing package/CLI needs (pyyaml, rich, typer, pandas==2.2.3, numpy==2.1.3, scikit-learn==1.6.1, mlflow==3.8.1). Drop `textual` unless the quiz TUI needs it (it does not — `rich` prompts suffice). Relax `pandas>=2.3.3` to `==2.2.3` to match the runtime.
- Dependency group `spark`: `pyspark==4.1.0` (local Spark tests).
- Dependency group `tuning`: `hyperopt==0.2.7`, `optuna==3.6.1`.
- Dependency group `databricks`: `databricks-sdk==0.67.0`, `databricks-feature-engineering==0.13.0.1` (imported only inside workspace-boundary modules; locally used for type-level checks and mocked in tests). Fallback: if `databricks-feature-engineering` cannot be locked together with the runtime pins, it is workspace-only; `workspace.py` imports it lazily inside functions so local imports never need it.
- `dev` includes all groups.

**Rationale**: Pins mirror R1 so behaviour seen locally matches the workspace (Principle II). Groups keep CI fast and make optional heavy installs explicit.

**Alternatives considered**: Unpinned `>=` ranges — rejected: drift from the runtime causes "works locally, fails on Databricks".

## R4. Local Spark testing

**Decision**: Use `pyspark==4.1.0` in `local[2]` mode via a session-scoped pytest fixture; tests marked `@pytest.mark.spark` and skipped automatically if Java 17+ is unavailable. Add `default-jre-headless` (OpenJDK 17/21) to the devcontainer and `actions/setup-java` (Temurin 17) to CI.

**Rationale**: Section 2 (summary stats, outliers, imputation, one-hot encoding) and Section 3 (Spark ML pipelines, CrossValidator) objectives are Spark-specific and can be fully tested locally.

**Alternatives considered**: Databricks Connect against a workspace — rejected for the default suite (needs credentials/network; violates deterministic local tests). Kept as an optional path for `databricks`-marked tests.

## R5. Workspace-dependent code boundary and tests

**Decision**: Workspace-only operations (Unity Catalog feature tables, UC model registry, AutoML, Model Serving, Lakeflow pipelines, online stores) live in modules named `workspace.py` within each chapter package, accepting injected clients (`FeatureEngineeringClient`, `MlflowClient`, `WorkspaceClient`). Local tests use fakes/`pytest-mock`; real-workspace tests are marked `@pytest.mark.databricks` and skipped unless `DATABRICKS_HOST` is set **or** they run inside a Databricks runtime (`DATABRICKS_RUNTIME_VERSION` present). A notebook `notebooks/00_run_workspace_tests.py` runs `pytest -m databricks` inside the workspace (pattern already anticipated by the `notebook_run unit_tests.py` ruff ignore).

**Rationale**: Constitution Principle III requires isolation behind clear boundaries and automatic skipping.

**Alternatives considered**: Testing only through notebooks — rejected: not runnable in CI.

## R6. Tutorial delivery format

**Decision**: Two layers per chapter:
1. **Package module(s)** in `src/db_ml_associate/chNN_<topic>/` — small, typed, documented, tested functions (the "how").
2. **Databricks notebook** in `notebooks/chNN_<topic>.py` (Databricks source format, `# Databricks notebook source` + `# COMMAND ----------`) that walks the learner step by step, calls package functions, and shows inline the short API calls the exam asks about (e.g., `fe.create_table`, `client.set_registered_model_alias`). Notebooks import the package from the Git folder (`sys.path` to `src`, or `%pip install -e`).

**Rationale**: Learners need to see exam APIs literally (constitution IV clarity), while the package keeps logic testable (III). Source-format notebooks diff cleanly and are linted by ruff.

**Alternatives considered**: `.ipynb` — rejected: noisy diffs, outputs committed. Package only — rejected: learners must read code across files to follow a tutorial.

## R7. Snippet synchronisation (book ↔ code)

**Decision**: In `src/` and `notebooks/`, mark regions with `# --8<-- [start:name]` / `# --8<-- [end:name]`. In chapters, embed with an HTML comment marker followed by a fenced block: `<!-- snippet: src/db_ml_associate/ch07_.../impute.py#impute_median -->`. A CLI command `db-ml-associate book sync-snippets` rewrites the fenced block from source; `book check` and a pytest test fail if any block is stale. Illustrative-only blocks carry `<!-- illustrative -->`.

**Rationale**: Principle III ("snippets sourced from or kept in sync with tested code"). Marker syntax is compatible with pymdown "snippets" should a site generator be added later.

**Alternatives considered**: Manual copy — rejected (drifts). Literate notebooks rendered to Markdown — rejected (heavy toolchain, out of scope).

## R8. Book structure and chapter breakdown

**Decision** (revised after `/speckit-analyze` D1/D2): an unnumbered introduction, 13 chapters **numbered in exam-guide order** (Section 1 → 4, objectives in guide order within each section), and appendices A–D. The introduction and appendices are not chapters.

| # | Chapter | Exam objectives (see `book/exam/objectives.yaml`) |
|---|---|---|
| — | Introduction (`book/introduction.md`, front matter): exam format, how to use this book, workspace setup, runtime, reading order | — |
| 01 | The Databricks ML platform: ML runtimes, Unity Catalog, MLOps strategy, deploy-code vs deploy-models | S1.01, S1.02, S1.16 |
| 02 | AutoML | S1.03, S1.04 |
| 03 | Feature engineering in Unity Catalog: create/write/train/score, online vs offline | S1.05–S1.10 |
| 04 | Experiment tracking with MLflow: manual logging, UI, best run via Client API | S1.11–S1.13 |
| 05 | Model lifecycle in Unity Catalog: register, tags, aliases (champion/challenger), UC vs workspace registry | S1.14, S1.15, S1.17, S1.18 |
| 06 | Exploring data: summary statistics, data profiles, visualisations, comparing features | S2.01, S2.03, S2.04 |
| 07 | Cleaning & transforming data: outliers, imputation, one-hot encoding, log transforms | S2.02, S2.05–S2.09 |
| 08 | ML foundations: algorithm choice, imbalance, bias–variance | S3.01, S3.02, S3.15 |
| 09 | Pipelines: estimators vs transformers (Spark ML & scikit-learn), training pipelines | S3.03, S3.04 |
| 10 | Validation & tuning: CV vs train/validation split, grid/random/Bayesian, Hyperopt `fmin`, Optuna, parallel single-node tuning, model counting | S3.05–S3.10 |
| 11 | Evaluation: classification & regression metrics, metric choice, back-transforming log targets | S3.11–S3.14 |
| 12 | Batch & streaming inference: serving approaches, pandas batch, Spark UDF, Lakeflow (ex-DLT) streaming | S4.01, S4.03, S4.04 |
| 13 | Real-time serving: custom model endpoints, querying, traffic splitting | S4.02, S4.05, S4.06 |
| A | Exam vs current platform (renames/deprecations) | FR-005 |
| B | Cheat sheet | FR-020 |
| C | Study plans (14-day, 21-day) | FR-016 |
| D | Practice exams 1 & 2 + answer keys | FR-017/018 |

**Rationale**: The constitution requires chapters ordered to follow the exam guide. Learning dependencies (Section 1 assumes the learner can already train and evaluate models) are handled by a `prerequisites` field per chapter and by the study plans, whose recommended reading order is: introduction → 01 → 06 → 07 → 08 → 09 → 11 → 10 → 04 → 02 → 03 → 05 → 12 → 13. Chapters are *written* in a similar dependency order (08 → 09 → 11 → 10 → 06 → 07 → 01 → 04 → 02 → 03 → 05 → 12 → 13) so shared code exists first. The book README holds the exam-objective coverage table (generated, see R10).

**Alternatives considered**: Numbering chapters in learning order — rejected: conflicts with the constitution ("ordered to follow the exam guide") and makes the section → chapter mapping harder to follow. Amending the constitution — rejected: study plans already solve reading order without a governance change.

## R9. Datasets

**Decision** (revised after `/speckit-analyze` D3): a deterministic **synthetic data generator** in `src/db_ml_associate/datasets.py` whose output is **committed** as `data/customer_churn.parquet` (5,000 rows) and `data/house_prices.parquet` (3,000 rows), each < 500 KB, under the repository's MIT licence, documented in `data/README.md`. Tutorials and tests load the committed files; a test regenerates them and asserts equality. The generator produces:
- `customer_churn` (classification): ~10% positive class (mirrors official sample Q3), categorical (low- and high-cardinality), numeric with injected outliers and missing values, `customer_id` primary key.
- `house_prices` (regression): right-skewed target (motivates log transform and back-transform), numeric/categorical features, outliers, missing values.
Locally the files are read with pandas; in the workspace `00_setup.py` reads them from the Git folder and saves them as UC Delta tables. Optionally referencing `samples`/`/databricks-datasets` in "try it yourself" boxes.

**Rationale**: No licence concerns, no network in tests, every Section 2/3 phenomenon can be dialled in deterministically (seeded). Works on all workspace types.

**Alternatives considered**: `/databricks-datasets` (wine quality, Airbnb) — not available locally; scikit-learn bundled datasets — lack missing values/high-cardinality categoricals/imbalance; generating at runtime without committing files — conflicts with the constitution's datasets rule.

## R10. Question bank, practice exams and coverage tracking

**Decision**: Structured YAML under `book/`:
- `book/exam/objectives.yaml` — the 48 objectives with stable IDs `S<section>.<nn>`, verbatim text, exam-guide version.
- `book/questions/chNN.yaml` — questions (schema in [contracts/question-bank.md](contracts/question-bank.md)).
- `book/exams/practice-1.yaml`, `practice-2.yaml` — ordered question IDs, 48 each, 90 minutes.
- Chapter front matter lists covered objective IDs and tutorial references.
A typer CLI (`db-ml-associate`) provides `book check` (coverage, schema, snippets, links), `book coverage` (renders the coverage table into `book/README.md`), `exam take <id>` (timed terminal quiz with per-section scoring via `rich`), and `exam render <id>` (Markdown version for paper practice).

**Per-section question counts** (proportional to objective count, 48 total): S1 = 18, S2 = 9, S3 = 15, S4 = 6 — identical to objective counts since there are exactly 48 objectives; each practice exam therefore contains one question per objective. The two exams contribute 96 questions; SC-008 (≥ 200 total) therefore requires ≥ 104 chapter-review questions, i.e. **≥ 8 per chapter for chapters 01–13** (FR-003's minimum of 5 is raised to 8). `book check` enforces both totals. Practice-exam questions are distinct from chapter-review questions.

**Rationale**: Machine-checkable coverage satisfies FR-001/SC-001 and constitution "gaps tracked as open work". Existing deps (`typer`, `rich`, `pyyaml`) already fit.

**Alternatives considered**: Questions embedded in Markdown only — rejected: no automated coverage or scoring. External quiz platform — rejected: out of repo, not versioned.

## R11. Continuous documentation verification (user instruction: "keep on checking the docs website")

**Decision**: Make docs verification a first-class, repeatable activity:
1. `book/references.yaml` registers every cited docs URL (chapters use plain inline Markdown links; `book check` errors on any docs.databricks.com URL not registered) with `objectives`, `chapters`, `last_verified` date and a one-line `claim` it supports.
2. `db-ml-associate book check-links` (pytest marker `network`, off by default, run weekly in CI via `schedule`) fetches every URL and fails on 404/redirect-to-different-page — the docs site renames paths often (e.g., `/ldp/`, `automl-hyperparam-tuning`).
3. Each chapter task in `tasks.md` includes a **"Verify against docs"** step: re-read the cited pages, update `last_verified`, record discrepancies (Principle II).
4. A **docs-coverage sweep** task re-crawls the ML section index (`/machine-learning/`, `/mlflow/`, `/ldp/`, `/machine-learning/model-serving/`, `/machine-learning/feature-store/`) and records each sub-page as *covered*, *beyond the exam*, or *out of scope* in `book/exam/docs-coverage.md`, so new or changed docs pages are noticed.
5. `book check` warns when any reference's `last_verified` is older than 90 days.

**Rationale**: Directly implements the user's instruction and constitution Principle II (cite docs; re-verify on change). The 404s encountered during this research (`/machine-learning/automl/hyperparameter-tuning/`, `/release-notes/runtime/18.0lts-ml`) show paths move.

**Alternatives considered**: One-off manual check before release — rejected: docs change faster than the 2–3 week study window.

## R12. Platform drift — how each item is taught

| Exam wording | Current platform (2026-10) | Book approach | Source |
|---|---|---|---|
| Hyperopt `fmin`, `SparkTrials` | Hyperopt not included after 16.4 LTS ML; Optuna (single node, parallel via joblib) or Ray Tune recommended; `SparkTrials` needs a `sparkContext`, so it requires classic Dedicated compute (not serverless/Spark Connect) | Teach `fmin`/`tpe`/`SparkTrials` (install `hyperopt`) labelled *exam-required, deprecated*; side-by-side Optuna equivalent as current practice | [Hyperparameter tuning](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/) |
| Delta Live Tables | Lakeflow pipelines (Spark Declarative Pipelines, formerly Delta Live Tables) (`from pyspark import pipelines as dp`, `@dp.materialized_view` / streaming tables) | Use current API; glossary maps DLT → Lakeflow; exam answer patterns use "DLT" | [Lakeflow pipelines](https://docs.databricks.com/aws/en/ldp/), [Transform data](https://docs.databricks.com/aws/en/ldp/transform), [Python ref](https://docs.databricks.com/aws/en/ldp/developer/python-ref) |
| Online tables / online feature tables | Online Feature Stores on Lakebase (`fe.create_online_store`, `fe.publish_table`) | Teach online vs offline concept (exam) with current API | [Online feature store](https://docs.databricks.com/aws/en/machine-learning/feature-store/online-feature-store) |
| Workspace model registry, stages | UC registry (default in MLflow 3, `databricks-uc`), aliases & tags; workspace registry legacy | UC as recommended path; stages shown only as pitfall | [Model lifecycle in UC](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/) |
| `FeatureStoreClient` | `FeatureEngineeringClient` (`databricks-feature-engineering`) | `FeatureEngineeringClient`; `FeatureStoreClient` as pitfall (cf. official sample Q1) | [Feature tables in UC](https://docs.databricks.com/aws/en/machine-learning/feature-store/uc/feature-tables-uc) |
| AutoML (built in) | Not built in from 18.0 ML; needs Dedicated compute | Concepts + UI walkthrough; install note; read-through alternative for serverless-only learners | [AutoML](https://docs.databricks.com/aws/en/machine-learning/automl/) |
| MLflow 2 `artifact_path` | MLflow 3 `name=`, LoggedModel entity | MLflow 3 API; note old parameter | [MLflow tracking](https://docs.databricks.com/aws/en/mlflow/tracking) |
| MLlib CV auto-logging | Deprecated; use `mlflow.pyspark.ml.autolog()` | Use autolog explicitly | Hyperparameter tuning page |

## R13. Key API facts confirmed for tutorials (docs snapshot)

- **Feature engineering**: `fe.create_table(name, primary_keys, schema|df, description, timeseries_columns=…)`; `fe.write_table(name, df, mode="merge")`; `FeatureLookup(table_name, feature_names, lookup_key, timestamp_lookup_key, …)`; `fe.create_training_set(df, feature_lookups, label, exclude_columns)` → `.load_df()`; `fe.log_model(model, artifact_path|name, flavor, training_set, registered_model_name)`; `fe.score_batch(model_uri, df)`. SQL alternative: `PRIMARY KEY` constraint on a UC Delta table. ([feature-tables-uc](https://docs.databricks.com/aws/en/machine-learning/feature-store/uc/feature-tables-uc), [train-models-with-feature-store](https://docs.databricks.com/aws/en/machine-learning/feature-store/train-models-with-feature-store))
- **Online**: `fe.create_online_store(name, capacity="CU_1")`, `fe.publish_table(online_store, source_table_name, online_table_name, publish_mode="TRIGGERED"|"CONTINUOUS"|"SNAPSHOT")`; requires PK and Change Data Feed for triggered/continuous. Not available on Free Edition (online tables unsupported). ([online-feature-store](https://docs.databricks.com/aws/en/machine-learning/feature-store/online-feature-store), [Free Edition limitations](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations))
- **Registry**: three-level names; `MlflowClient.set_registered_model_alias`, `delete_registered_model_alias`, `set_model_version_tag`, `delete_model_version_tag`, `set_registered_model_tag`, `delete_registered_model_tag`; load `models:/cat.sch.model@champion`. ([manage-model-lifecycle](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/))
- **Serving**: endpoint config `served_entities` (+ `entity_name`, `entity_version`, `workload_size`, `scale_to_zero_enabled`) and `traffic_config.routes[{served_model_name, traffic_percentage}]`; query `POST /serving-endpoints/<name>/invocations` with `dataframe_split` (recommended), `dataframe_records`, `instances`, `inputs`; `mlflow.deployments.get_deploy_client("databricks").predict(...)`; `ai_query` in SQL. ([serve-multiple-models](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-multiple-models-to-serving-endpoint), [score-custom-model-endpoints](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints))
- **Batch/streaming inference**: AI Functions/`ai_query` recommended for batch on Databricks data; `mlflow.pyfunc.spark_udf` inside a Lakeflow `@dp.materialized_view`/streaming table for pipeline inference. ([model-inference](https://docs.databricks.com/aws/en/machine-learning/model-inference/), [ldp/transform](https://docs.databricks.com/aws/en/ldp/transform))
- **MLOps**: "deploy code" is the default recommendation; "deploy models" when training is expensive/single workspace; environment ≈ UC catalog. ([deployment-patterns](https://docs.databricks.com/aws/en/machine-learning/mlops/deployment-patterns))
- **Data summaries**: `dbutils.data.summarize(df, precise=False)` (Public Preview; analyses whole DataFrame); notebook data profile tab; visualisation editor. ([databricks-utils](https://docs.databricks.com/aws/en/dev-tools/databricks-utils), [visualizations](https://docs.databricks.com/aws/en/notebooks/visualizations/))

## R13b. Exam answer conventions derived from the official sample questions

**Decision**: Encode these conventions in chapters and question authoring guidelines (spec FR-008 was reworded to match):
- **Model counting (S3.10)**: total = (product of grid sizes) × (number of folds); the final refit is *not* counted (sample Q4: 3 × 2 × 3 × 5 = 90). The book explains the refit as a nuance in a "Beyond the exam" note.
- **Feature table creation (S1.06)**: `FeatureEngineeringClient.create_table` is the expected answer (sample Q1); SQL `PRIMARY KEY` route is a valid alternative to teach, but fictional clauses such as `AS FEATURE STORE` are distractors.
- **Imputation (S2.05/S2.06)**: "inspect the distribution, then choose" beats blanket mean/mode (sample Q2).
- **Imbalance (S3.02)**: cost-sensitive learning / class weights directly mitigate bias (sample Q3).
- **Streaming inference (S4.04)**: a declarative pipeline (DLT/Lakeflow) applying the model as a Spark UDF meets autoscaling streaming needs; calling a serving endpoint from a UDF is the distractor (sample Q5).

**Rationale**: The official samples reveal how the exam scores ambiguous points; teaching against them would cost marks.

## R14. Workspace requirements per tutorial (edge case: free / serverless-only accounts)

**Decision**: Each tutorial's front matter declares `requires: [serverless | classic_ml_runtime | model_serving | online_store | lakeflow | unity_catalog]`. Free Edition is serverless-only, caps serving endpoints, has no online tables, one pipeline per type. Mapping:

| Chapters | Free Edition | Trial / paid workspace |
|---|---|---|
| 01 Platform, 04 MLflow, 06–09, 11 (data, foundations, pipelines, evaluation) | ✅ (serverless; Spark ML on serverless verified per tutorial during implementation) | ✅ |
| 10 Tuning | ⚠️ partial — Hyperopt `SparkTrials` needs a `sparkContext`, unavailable on serverless/Spark Connect; use Hyperopt `Trials` or Optuna + joblib there | ✅ classic Dedicated compute for `SparkTrials` |
| 02 AutoML | ❌ (needs Dedicated compute) → read-through with screenshots | ✅ classic ML runtime (cost note + terminate cluster) |
| 03 Feature engineering (offline) | ✅ | ✅ |
| 03 Online store | ❌ → read-through | ✅ (cost note + clean-up) |
| 05 Registry | ✅ | ✅ |
| 12 Batch & Lakeflow streaming | ✅ (1 pipeline) | ✅ |
| 13 Serving & traffic split | ⚠️ limited endpoints, CPU only | ✅ (cost note + clean-up) |

**Rationale**: FR-012, spec edge case "learner has no paid workspace". Exact Free Edition behaviour for each tutorial is verified during implementation and recorded in the tutorial front matter.

## R15. Time budget (SC-002)

**Decision**: Budget ≈ 50 h total: introduction + chapters 01–13 reading ≈ 22 h, tutorials ≈ 16 h, chapter reviews ≈ 4 h, two practice exams + review ≈ 6 h, cheat-sheet revision ≈ 2 h. 21-day plan ≈ 2.4 h/day; 14-day plan ≈ 3 h/day with lighter optional tutorials (marked "skim" for chapters a learner already knows). Each chapter states an estimated reading and tutorial time; `book check` sums them and fails if a plan exceeds its budget.

**Rationale**: Keeps the "2–3 weeks" promise verifiable rather than aspirational.
