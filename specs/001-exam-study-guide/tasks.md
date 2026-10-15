---

description: "Task list for the exam-aligned study guide and tutorial"
---

# Tasks: Exam-Aligned Study Guide and Tutorial for the Databricks ML Associate Exam

**Input**: Design documents from `specs/001-exam-study-guide/` — [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/), [quickstart.md](quickstart.md)

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: REQUIRED. Constitution Principle III mandates tests for every tutorial module, pitfall demonstrations asserted by tests (Principle V), and content validation gates. Write tests first and confirm they fail before implementing.

**Organization**: Tasks are grouped by user story. Chapters are numbered in exam-guide order (constitution); chapter blocks are *written* in learning order (08 → 09 → 11 → 10 → 06 → 07 → 01 → 04 → 02 → 03 → 05 → 12 → 13) because later chapters reuse earlier code. The learner's reading order is set by the study plans (US3).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story from spec.md (US1–US5)

## Conventions used by every chapter task (read once)

- **Chapter file** follows [contracts/content-formats.md](contracts/content-formats.md): front matter (`chapter`, `title`, `objectives` (≥ 1), `prerequisites`, `reading_minutes`, `tutorial`, `references`); H2 headings in order `Learning objectives` (≥ 3 items) → content sections each tagged `<!-- objectives: … -->` → `Choosing an approach` (when alternatives exist) → `Pitfalls` → `Summary` → `Review questions` with `<!-- questions: chNN -->`. Use callouts `> **Exam tip:**`, `> **Platform change:**`, `> **Beyond the exam:**`, `> **⚠️ Wrong approach:**`, `> **Requires:**`. All code blocks are synced snippets (`<!-- snippet: path#region -->`) or marked `<!-- illustrative -->`. Links to docs.databricks.com are plain inline Markdown links whose URL is registered in `book/references.yaml`. Only `book/NN-*.md` (01–13) are chapters; `book/introduction.md` and `book/appendix-*.md` are front/back matter and exempt from chapter rules.
- **Question files** `book/questions/chNN.yaml`: ≥ 8 questions with `pool: review`, original (no copying of official sample questions), scenario-based, `type: single|multiple`, 4–5 options, `answer` length 1 for single / ≥ 2 for multiple, an `explanation`, a `distractors` entry for every non-answer option, `chapter_ref`, `difficulty: easy|medium|hard`; every objective of the chapter has ≥ 1 question. Follow exam conventions in research R13b.
- **Code modules**: Python 3.12, typed, docstrings, ruff/mypy clean, deterministic seeds, snippet regions `# --8<-- [start:name]` / `# --8<-- [end:name]`. Spark code uses `pyspark` 4.1 APIs; pandas/scikit-learn code uses runtime pins in `runtime.toml`.
- **Pitfall demos** live in `src/db_ml_associate/chNN_*/pitfalls.py`: each exposes a `wrong_*` and `correct_*` function; tests assert the observable consequence of the wrong approach and that the correct approach avoids it.
- **"Verify against docs" step** (user instruction): re-read every page cited by the chapter on docs.databricks.com/aws/en, confirm each claim, update `claim`/`last_verified`/`status` in `book/references.yaml`, add any newly found relevant page, and record discrepancies with `status: discrepancy` + `note`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Align the project with Databricks Runtime 18 LTS ML (research R1–R4) and create the content skeleton.

- [X] T001 Change `.python-version` to `3.12` and in `pyproject.toml` set `requires-python = ">=3.12,<3.13"`, classifier `Programming Language :: Python :: 3.12`, `[tool.ruff] target-version = "py312"`, `[tool.mypy] python_version = "3.12"` (research R2)
- [X] T002 Update `pyproject.toml` dependencies per research R3: `[project].dependencies` = `pandas==2.2.3`, `numpy==2.1.3`, `scikit-learn==1.6.1`, `mlflow==3.8.1`, `pyyaml`, `rich`, `typer` (remove `textual`); add dependency groups `spark = ["pyspark==4.1.0"]`, `tuning = ["optuna==3.6.1", "hyperopt==0.2.7"]`, `databricks = ["databricks-sdk==0.67.0", "databricks-feature-engineering==0.13.0.1"]`, include all in `dev`; add pytest markers `spark`, `databricks`, `network`; run `uv lock` and commit `uv.lock`. If `uv lock` fails because `databricks-feature-engineering` conflicts with the `mlflow`/`pyspark` pins, remove it from the local `databricks` group, keep it in `runtime.toml` (workspace-only) and record the conflict in research R3; ch03 tests use a fake client and never import the package
- [X] T003 [P] Create `runtime.toml` at repo root exactly as in [contracts/content-formats.md](contracts/content-formats.md#runtimetoml) (18 LTS ML, Python 3.12.3, Spark 4.1.0, library pins, `docs_snapshot = 2026-10-14`)
- [X] T004 [P] Update `.devcontainer/Dockerfile` to `FROM mcr.microsoft.com/devcontainers/python:3.12` and install `default-jre-headless` (Java 17+ for local Spark) in the existing `apt-get` layer
- [X] T005 [P] Update `.github/workflows/smoke_test.yaml`: add `actions/setup-java` (Temurin 17) before tests and change both `uv sync` steps to `uv sync --all-groups` (the `book check` step and the scheduled docs job are added later, once those commands exist)
- [X] T006 [P] Create directory skeleton with `.gitkeep` where empty: `book/exam/`, `book/questions/`, `book/exams/`, `book/plans/`, `book/images/`, `notebooks/`, `src/db_ml_associate/cli/`, `src/db_ml_associate/content/`, `tests/cli/`, `tests/content/`
- [X] T007 [P] Add `# Databricks notebook source` handling to ruff config: per-file-ignores for `notebooks/*.py` (`F821` for `spark`/`dbutils`/`display`, `E402`, `T201`) in `pyproject.toml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Content registry, validators, CLI skeleton, datasets and test fixtures that every story depends on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

### Tests (write first)

- [X] T008 [P] Write `tests/conftest.py`: session-scoped `spark` fixture (`SparkSession.builder.master("local[2]")`, shuffle partitions 2, seed-friendly) that skips with reason "Java 17+ not available" when `java` is missing; auto-skip `databricks`-marked tests unless `DATABRICKS_HOST` or `DATABRICKS_RUNTIME_VERSION` is set (reason "no Databricks workspace configured"); auto-skip `network`-marked tests unless `--run-network` option or `RUN_NETWORK_TESTS=1` (reason "network tests disabled")
- [X] T009 [P] Write `tests/content/test_models.py`: tests for content entity validation rules from data-model.md — ExamObjective `id` matches `S<1-4>.<NN>`, exactly 48 objectives with section counts 18/9/15/6; Chapter `number` 1–13 and `objectives` ≥ 1; Question `answer` "Length 1 for `single`, ≥ 2 for `multiple`", 4–5 options, a distractor explanation for every non-answer option; PracticeExam "Exactly 48, all `pool: practice`, no reuse across exams"; Tutorial `cleanup` "MUST be true if `requires` contains `model_serving` or `online_store`"; Reference `last_verified` older than 90 days yields a warning
- [X] T010 [P] Write `tests/test_datasets.py`: `make_customer_churn(n_rows=5000, seed=42)` is deterministic, has unique `customer_id`, positive-class rate within 9–11 %, ≥ 1 numeric column with ~5 % nulls, a low-cardinality (≤ 5) and a high-cardinality (≥ 500) categorical, injected outliers beyond 3 σ; `make_house_prices(seed=42)` target skewness > 1 and `log1p(target)` skewness < 0.5; `load_customer_churn()`/`load_house_prices()` read the committed `data/*.parquet` files, and regenerating with the default seed and row count equals them (`pandas.testing.assert_frame_equal`)
- [X] T011 [P] Write `tests/cli/test_snippets.py`: region extraction from `# --8<-- [start:x]/[end:x]` (dedented), sync rewrites the fenced block after `<!-- snippet: path#x -->`, `--check` exits 1 on a stale block, `<!-- illustrative -->` blocks are ignored, missing region → error
- [X] T012 [P] Write `tests/cli/test_book_check.py` using a temporary fixture book: each failure condition listed in [contracts/cli.md](contracts/cli.md#book-check---strict) is detected with `file:line — message` (including a docs.databricks.com URL missing from `references.yaml` and a study plan scheduling a chapter before its `prerequisites` → warning, and a section whose share of all questions deviates > ±5 points from its share of objectives), exit code 1 on errors, 0 on a valid fixture; `--strict` turns stale-reference warnings into errors
- [X] T013 [P] Write `tests/test_runtime_pins.py`: every library in `runtime.toml [libraries]` that is also in `pyproject.toml` (dependencies or groups) has the identical `==` pin; `.python-version` major.minor equals `runtime.toml` python major.minor
- [X] T014 [P] Write `tests/content/test_loaders.py`: loaders return typed entities for fixture YAML/TOML/front matter, report `file:line` for malformed YAML and for a chapter section tag referencing an unknown objective id, and parse `<!-- objectives: S2.05, S2.06 -->` tags with their line numbers
- [X] T015 [P] Write `tests/cli/test_coverage.py`: on a fixture book, `book coverage` lists all objectives with chapter-section/tutorial-step/question counts, flags gaps, and `--write` replaces only the text between `<!-- coverage:start -->` and `<!-- coverage:end -->` (idempotent on second run)
- [X] T016 [P] Write `tests/cli/test_datasets_cli.py`: `datasets generate churn --rows 100 --seed 1 --out x.parquet` and `.csv` write files equal to `make_customer_churn(100, 1)`; unknown dataset name exits 2

### Implementation

- [X] T017 Implement content entities as frozen dataclasses with validation in `src/db_ml_associate/content/models.py` (ExamGuide, ExamObjective, Chapter, ChapterSection, Tutorial, Question, PracticeExam, Reference, PlatformChange, StudyPlan, StudyPlanDay, RuntimeTarget — fields and rules exactly as in data-model.md)
- [X] T018 Implement YAML/TOML/front-matter loaders in `src/db_ml_associate/content/loaders.py` (load objectives, questions, exams, references, platform changes, plans, runtime; parse chapter front matter and `<!-- objectives: … -->` section tags with line numbers)
- [X] T019 [P] Create `book/exam/objectives.yaml` with the exam-guide header (`version: 2025-03-01`, URL, `questions: 48`, `minutes: 90`, `checked_on: 2026-10-14`) and all 48 objectives verbatim from the exam guide PDF, IDs S1.01–S1.18, S2.01–S2.09, S3.01–S3.15, S4.01–S4.06 in guide order, each with `kind: conceptual|practical` (practical = create/write/train/score/log/register/set/promote/compute/remove/create visualizations/impute/use/develop/perform/parallelize/deploy/split)
- [X] T020 [P] Seed `book/references.yaml` with every docs URL cited in research.md R1, R11–R13 (ids e.g. `dbr-18-lts`, `dbr-18-lts-ml`, `dbr-18-0-ml`, `hyperparameter-tuning`, `automl`, `feature-tables-uc`, `train-with-feature-store`, `online-feature-store`, `model-lifecycle-uc`, `serve-multiple-models`, `query-custom-endpoints`, `model-inference`, `mlops-deployment-patterns`, `lakeflow-pipelines`, `lakeflow-transform`, `lakeflow-python-ref`, `dbutils`, `notebook-visualizations`, `mlflow-tracking`, `free-edition-limitations`) with `claim`, `objectives`, `chapters`, `last_verified: 2026-10-14`, `status: ok`
- [X] T021 [P] Create `book/exam/platform-changes.yaml` with the minimum set from research R12: Hyperopt, Delta Live Tables, online tables, workspace model registry & stages, FeatureStoreClient, AutoML packaging (not built in from 18.0 ML), MLflow 3 `name`/LoggedModel, MLlib CV auto-logging — each with `exam_term`, `current_term`, `change`, `exam_expects`, `objectives`, `references`
- [X] T022 [P] Implement deterministic synthetic datasets in `src/db_ml_associate/datasets.py` (`make_customer_churn`, `make_house_prices`, both returning pandas DataFrames, seeded via `numpy.random.default_rng`) satisfying `tests/test_datasets.py`
- [X] T023 [P] Implement snippet extraction/sync in `src/db_ml_associate/cli/snippets.py` satisfying `tests/cli/test_snippets.py`
- [X] T024 Implement `book check` validators in `src/db_ml_associate/cli/book_check.py` covering every row of the check table in [contracts/cli.md](contracts/cli.md) (objectives, coverage, chapters, questions incl. fuzzy match ≥ 0.85 against the five official sample stems stored in `src/db_ml_associate/content/official_samples.py`, exams, totals ≥ 200 and section proportions within ±5 points of objective shares (37.5/18.75/31.25/12.5 %), snippets, relative links/anchors, references incl. 90-day staleness and unregistered docs.databricks.com URLs, platform changes, plans incl. `prerequisites` ordering warning, runtime pins; chapter rules apply only to `book/NN-*.md` with NN 01–13); a chapter that does not exist yet is reported as "planned" open work, not an error, unless `--strict`
- [X] T025 Implement `book coverage [--write]` in `src/db_ml_associate/cli/coverage.py` (objective → chapter sections / tutorial steps / question counts; writes between `<!-- coverage:start -->` and `<!-- coverage:end -->` in `book/README.md`)
- [X] T026 Wire the typer app in `src/db_ml_associate/__main__.py` and `src/db_ml_associate/cli/__init__.py`: groups `book` (`check`, `sync-snippets`, `coverage`, `check-links`), `exam` (`take`, `render`), `datasets` (`generate`); exit codes 0/1/2 per contract; update `tests/test_main.py` (`main` now dispatches the app; use `typer.testing.CliRunner`)
- [X] T027 [P] Implement `datasets generate <churn|house_prices> [--rows] [--seed] [--out]` in `src/db_ml_associate/cli/datasets.py` writing CSV or Parquet by extension
- [X] T028 Generate and commit `data/customer_churn.parquet` (5,000 rows) and `data/house_prices.parquet` (3,000 rows) with `uv run db-ml-associate datasets generate` (default seed), plus `data/README.md` (columns and meaning, seed, row counts, regeneration command, "synthetic data, MIT licence like the repository"); add `load_customer_churn()`/`load_house_prices()` to `src/db_ml_associate/datasets.py` reading these files (depends on T016, T022, T027)
- [X] T029 Create `book/README.md`: title, how to use the book, table of contents linking `introduction.md`, chapters 01–13 (in exam-guide order) and appendices A–D, a "Recommended reading order" pointer to the study plans, runtime target (link `../runtime.toml`), exam-guide version, and empty `<!-- coverage:start -->`/`<!-- coverage:end -->` block
- [X] T030 Write `tests/test_content.py` that runs `book check` and `book sync-snippets --check` on the real `book/` and asserts exit code 0 (non-strict during development)
- [X] T031 Add a step `uv run db-ml-associate book check` to the `static_analysis` job in `.github/workflows/smoke_test.yaml` (after "Install the project")

**Checkpoint**: `uv run pytest` and `uv run db-ml-associate book check` pass on the empty book (chapters reported as planned work).

---

## Phase 3: User Story 1 — Learn every exam objective from one book (Priority: P1) 🎯 MVP

**Goal**: Introduction, chapters 01–13 (exam-guide order) + Appendix A covering all 48 objectives, with tested code snippets, pitfalls and ≥ 8 review questions per chapter.

**Independent Test**: `uv run db-ml-associate book check --strict` and `book coverage` show every objective mapped to a chapter section and a review question; each chapter has the required sections (spec US1 acceptance 1–3, SC-001, SC-006, SC-007).

### Chapter 08 — ML foundations (S3.01, S3.02, S3.15)

- [X] T032 [P] [US1] Write tests in `tests/ch08_foundations/test_imbalance.py` and `tests/ch08_foundations/test_bias_variance.py`: on churn data, class-weighted logistic regression has higher minority recall than unweighted; random over/undersampling applied only to the training split; train vs validation error curve over tree depth shows rising gap (variance) at high depth and both high at depth 1 (bias)
- [X] T033 [US1] Implement `src/db_ml_associate/ch08_foundations/imbalance.py` (class weights, `class_weight="balanced"`, Spark ML `weightCol`, random over/undersampling on the train split) and `bias_variance.py` (complexity sweep returning train/validation scores)
- [X] T034 [US1] Implement `src/db_ml_associate/ch08_foundations/pitfalls.py` + tests in `tests/ch08_foundations/test_pitfalls.py`: `wrong_resample_before_split` (leakage → inflated validation score) vs `correct_resample_after_split`; `wrong_accuracy_on_imbalanced` (≈ 90 % accuracy with zero minority recall) vs `correct_use_recall_f1`
- [X] T035 [US1] Write `book/08-ml-foundations.md`: algorithm selection by scenario (problem type, data size, interpretability, linear vs tree-based, table of scenarios → algorithm), imbalance mitigation (class weights/cost-sensitive — exam tip per R13b, resampling, threshold tuning, metrics), bias–variance trade-off and model complexity; Choosing an approach; Pitfalls; Summary
- [X] T036 [P] [US1] Write `book/questions/ch08.yaml` (≥ 8 review questions covering S3.01, S3.02, S3.15)
- [X] T037 [US1] Verify ch08 against docs (see Conventions) and run `uv run db-ml-associate book sync-snippets && uv run db-ml-associate book check`

### Chapter 09 — Pipelines (S3.03, S3.04)

- [X] T038 [P] [US1] Write tests in `tests/ch09_pipelines/test_pipelines.py` (marker `spark` for Spark parts): Spark ML `StringIndexer`/`OneHotEncoder`/`VectorAssembler`/`LogisticRegression` pipeline fits to a `PipelineModel` and transforms test data; transformers expose `transform` only, estimators `fit` → model; scikit-learn `Pipeline` with `ColumnTransformer` gives identical predictions after `mlflow.sklearn` save/load round trip
- [X] T039 [US1] Implement `src/db_ml_associate/ch09_pipelines/spark_pipeline.py` and `sklearn_pipeline.py` (build, fit, evaluate a churn training pipeline with snippet regions)
- [X] T040 [US1] Implement `src/db_ml_associate/ch09_pipelines/pitfalls.py` + `tests/ch09_pipelines/test_pitfalls.py`: `wrong_fit_scaler_on_full_data` (test statistics leak; assert scaler mean equals full-data mean) vs `correct_fit_inside_pipeline`; `wrong_transform_with_unfitted_estimator` (raises) vs correct
- [X] T041 [US1] Write `book/09-pipelines.md`: estimators vs transformers (Spark ML and scikit-learn terminology side by side), Pipeline/PipelineModel, building a training pipeline, Spark ML vs scikit-learn — when to choose each; Pitfalls; Summary
- [X] T042 [P] [US1] Write `book/questions/ch09.yaml` (≥ 8 questions covering S3.03, S3.04)
- [X] T043 [US1] Verify ch09 against docs (Spark ML pipelines page on docs.databricks.com; add reference) and run snippet sync + `book check`

### Chapter 11 — Evaluation (S3.11–S3.14)

- [X] T044 [P] [US1] Write tests in `tests/ch11_evaluation/test_metrics.py`: helper functions compute F1, log loss, ROC AUC, precision, recall, RMSE, MAE, R² equal to scikit-learn values; Spark `BinaryClassificationEvaluator`/`MulticlassClassificationEvaluator`/`RegressionEvaluator` agree with scikit-learn on the same predictions (marker `spark`); `choose_metric(scenario)` returns documented choices
- [X] T045 [US1] Implement `src/db_ml_associate/ch11_evaluation/metrics.py` (scikit-learn + Spark evaluator wrappers) and `metric_choice.py` (decision table: imbalanced → F1/PR-AUC, cost of FN vs FP → recall/precision, probability quality → log loss, outlier-sensitive regression → MAE vs RMSE)
- [X] T046 [US1] Implement `src/db_ml_associate/ch11_evaluation/pitfalls.py` + tests: `wrong_rmse_on_log_scale` (RMSE in log units, not comparable to dollars) vs `correct_exponentiate_before_metrics` (`expm1`/`exp` back-transform; assert values differ and correct one is in target units) — S3.14
- [X] T047 [US1] Write `book/11-evaluation.md`: classification metrics (confusion matrix, precision/recall/F1, log loss, ROC/AUC, PR-AUC), regression metrics (RMSE, MAE, R²), choosing a metric by objective, exponentiating log-transformed predictions; Choosing an approach; Pitfalls; Summary
- [X] T048 [P] [US1] Write `book/questions/ch11.yaml` (≥ 8 questions covering S3.11–S3.14)
- [X] T049 [US1] Verify ch11 against docs (MLflow evaluation / Spark ML evaluators pages) and run snippet sync + `book check`

### Chapter 10 — Validation and tuning (S3.05–S3.10)

- [X] T050 [P] [US1] Write tests in `tests/ch10_tuning/test_tuning.py`: `count_models(grid, folds)` returns product × folds without refit (exam convention, R13b; e.g. `{C:3, kernel:2, gamma:3}`, 5 folds → 90) and `count_models(..., include_refit=True)` adds 1; Spark `CrossValidator` with 2×2 grid and 3 folds fits 12 models (marker `spark`); `TrainValidationSplit` fits 4; Hyperopt `fmin` with `tpe.suggest`, `max_evals=10`, `rstate=np.random.default_rng(42)` returns a dict of best params; Optuna `TPESampler(seed=42)` study returns comparable best params; random search vs grid search trial counts
- [X] T051 [US1] Implement `src/db_ml_associate/ch10_tuning/cross_validation.py` (scikit-learn `cross_val_score`/`GridSearchCV`/`RandomizedSearchCV`; Spark `CrossValidator`, `TrainValidationSplit`, `ParamGridBuilder`; `count_models`)
- [X] T052 [US1] Implement `src/db_ml_associate/ch10_tuning/hyperopt_tuning.py` (`fmin`, `tpe.suggest`, `rand.suggest`, `hp.choice`/`hp.uniform`/`hp.quniform`/`hp.loguniform`, `Trials`, `SparkTrials(parallelism=…)` for single-node models with MLflow nested runs; module docstring and `Platform change` note that Hyperopt is not included after 16.4 LTS ML; docstring states `SparkTrials` needs a `sparkContext`, i.e. classic Dedicated compute, not serverless/Spark Connect) and `optuna_tuning.py` (`TPESampler`, `GridSampler`, `RandomSampler`, parallel via joblib `n_jobs`, MLflow callback)
- [X] T053 [US1] Implement `src/db_ml_associate/ch10_tuning/pitfalls.py` + tests: `wrong_tune_on_test_set` (optimistic score) vs `correct_tune_with_cv_then_test`; `wrong_fmin_maximise_metric` (Hyperopt minimises — returning accuracy picks the worst model) vs `correct_return_negative_metric`
- [X] T054 [US1] Write `book/10-validation-and-tuning.md`: train/validation/test, CV vs train–validation split (benefits/downsides), CV as part of fitting, grid vs random vs Bayesian (TPE) search, Hyperopt `fmin` (exam-required, deprecated — Platform change callout) with Optuna equivalent side by side, parallelising single-node tuning (`SparkTrials` with a `> **Requires:**` callout for classic Dedicated compute, Optuna + joblib as the serverless/Free Edition path, Ray Tune mention), counting models (section `#counting-models`, exam convention + refit nuance as Beyond the exam); Choosing an approach; Pitfalls; Summary
- [X] T055 [P] [US1] Write `book/questions/ch10.yaml` (≥ 8 questions covering S3.05–S3.10, including ≥ 2 model-counting questions)
- [X] T056 [US1] Verify ch10 against docs (`hyperparameter-tuning` and Hyperopt/Optuna/Ray Tune sub-pages, Spark ML tuning) and run snippet sync + `book check`

### Chapter 06 — Exploring data (S2.01, S2.03, S2.04)

- [X] T057 [P] [US1] Write tests in `tests/ch06_exploring/test_exploring.py` (marker `spark`): `df.summary()` returns count/mean/stddev/min/25%/50%/75%/max rows; `describe()` subset; categorical frequency table; two-continuous comparison returns Pearson and Spearman correlation; two-categorical comparison returns a contingency table and chi-square p-value (`pyspark.ml.stat.ChiSquareTest` and pandas/scipy agree)
- [X] T058 [US1] Implement `src/db_ml_associate/ch06_exploring/summary.py` (Spark `.summary()`, `.describe()`, pandas `describe()`; wrapper documenting `dbutils.data.summarize(df, precise=…)` as workspace-only call) and `compare.py` (correlation, crosstab, chi-square, box plot by category data preparation) and `plots.py` (matplotlib histogram/box plot/bar chart helpers returning figures)
- [X] T059 [US1] Write `book/06-exploring-data.md`: summary statistics via `.summary()`/`.describe()` vs `dbutils.data.summarize` vs notebook data profile tab (cost on large data, approximate vs precise), visualising categorical vs continuous features (bar, histogram, box plot, notebook visualisation editor), comparing two continuous (scatter, correlation) and two categorical (crosstab, chi-square) and mixed features; Choosing an approach; Pitfalls (e.g., correlation ≠ causation, Pearson on non-linear relations, summarize on huge tables); Summary
- [X] T060 [P] [US1] Write `book/questions/ch06.yaml` (≥ 8 questions covering S2.01, S2.03, S2.04)
- [X] T061 [US1] Verify ch06 against docs (`dbutils`, `notebook-visualizations`, EDA tutorial/data profile page) and run snippet sync + `book check`

### Chapter 07 — Cleaning and transforming (S2.02, S2.05–S2.09)

- [X] T062 [P] [US1] Write tests in `tests/ch07_cleaning/test_cleaning.py` (Spark parts marker `spark`): std-dev filter (|z| > 3) and IQR filter (1.5 × IQR via `approxQuantile` with relativeError 0) remove exactly the injected outliers of a fixture; `pyspark.ml.feature.Imputer` with strategy `mean`/`median`/`mode` fills nulls with the expected values; pandas `fillna` equivalents agree; `OneHotEncoder` after `StringIndexer` produces `numCategories − 1` (dropLast) vector size; `log1p` reduces skewness of `house_prices` target
- [X] T063 [US1] Implement `src/db_ml_associate/ch07_cleaning/outliers.py`, `impute.py`, `encoding.py`, `transforms.py` with snippet regions (Spark first, pandas/scikit-learn `SimpleImputer`/`OneHotEncoder` equivalents)
- [X] T064 [US1] Implement `src/db_ml_associate/ch07_cleaning/pitfalls.py` + tests: `wrong_mean_impute_skewed` (mean pulled by outliers; assert mean − median gap) vs `correct_median_impute`; `wrong_impute_before_split` (leakage) vs correct; `wrong_one_hot_high_cardinality_for_trees` (feature count explodes, assert width) vs `correct_index_or_target_encode`; `wrong_log_of_zero` (−inf) vs `correct_log1p`
- [X] T065 [US1] Write `book/07-cleaning-and-transforming.md`: outlier removal by standard deviation vs IQR (when each), imputation with mean/median/mode (compare and contrast; "inspect the distribution first" exam tip per R13b; indicator columns), one-hot encoding and when it is or is not appropriate (linear models vs tree-based, high cardinality, sparse vectors), log-scale transforms (right-skewed, multiplicative effects, non-negative data) and back-transformation link to ch11; Choosing an approach; Pitfalls; Summary
- [X] T066 [P] [US1] Write `book/questions/ch07.yaml` (≥ 8 questions covering S2.02, S2.05–S2.09)
- [X] T067 [US1] Verify ch07 against docs (Spark ML feature transformers / preprocessing pages) and run snippet sync + `book check`

### Chapter 01 — The Databricks ML platform (S1.01, S1.02, S1.16)

- [X] T068 [P] [US1] Write tests in `tests/ch01_platform/test_runtime_info.py`: `runtime_info()` returns `runtime.toml` values and, when `DATABRICKS_RUNTIME_VERSION` is set (monkeypatched), reports it
- [X] T069 [US1] Implement `src/db_ml_associate/ch01_platform/runtime_info.py` (reads `runtime.toml`, detects runtime env var, lists installed versions of pinned libraries)
- [X] T070 [US1] Write `book/01-databricks-ml-platform.md`: Databricks ML overview, ML runtimes and their advantages (pre-installed libraries, GPU variants, compatibility; standard vs ML runtime; serverless environments), Unity Catalog basics (catalog.schema.object, governance, lineage), MLOps strategy best practices (dev/staging/prod as catalogs, code in Git, CI/CD, monitoring), deploy code vs deploy models (default = deploy code; when deploy models is preferred) per `mlops-deployment-patterns`; Choosing an approach; Pitfalls; Summary
- [X] T071 [P] [US1] Write `book/questions/ch01.yaml` (≥ 8 questions covering S1.01, S1.02, S1.16)
- [X] T072 [US1] Verify ch01 against docs (`dbr-18-lts-ml`, `mlops-deployment-patterns`, MLOps workflows, Unity Catalog overview) and run snippet sync + `book check`

### Chapter 04 — Experiment tracking with MLflow (S1.11–S1.13)

- [X] T073 [P] [US1] Write tests in `tests/ch04_tracking/test_tracking.py` using a temporary SQLite tracking and registry URI (`sqlite:///{tmp_path}/mlflow.db`): manual logging of params, metrics (with steps), an artifact (PNG/CSV) and a model with signature and input example via `mlflow.sklearn.log_model(..., name=...)`; `find_best_run(client, experiment_id, metric, mode)` returns the run with max/min metric using `MlflowClient.search_runs(order_by=["metrics.f1 DESC"], max_results=1)`; nested child runs exist under the parent
- [X] T074 [US1] Implement `src/db_ml_associate/ch04_tracking/logging.py` (`mlflow.start_run`, `log_param(s)`, `log_metric(s)`, `log_artifact`, `log_figure`, `log_model` with `infer_signature`, autolog vs manual) and `best_run.py` (`MlflowClient.search_runs`, `mlflow.search_runs` DataFrame, filter strings)
- [X] T075 [US1] Implement `src/db_ml_associate/ch04_tracking/pitfalls.py` + tests: `wrong_log_model_without_signature` (no signature → assert `model_info.signature is None`; UC registry requires a signature) vs correct; `wrong_order_by_ascending_for_max_metric` vs correct
- [X] T076 [US1] Write `book/04-mlflow-tracking.md`: experiments vs runs (notebook vs workspace experiments), MLflow 3 LoggedModel, manual logging of metrics/params/artifacts/models, autologging, finding the best run with the Client API, what the MLflow UI shows (runs table, parameters, metrics charts, artifacts, model signature, source notebook/Git commit, comparison view) with screenshots in `book/images/ch04-*.png` (placeholder list if not yet captured); Platform change callout (`artifact_path` → `name`); Choosing an approach (autolog vs manual; UI vs API); Pitfalls; Summary
- [X] T077 [P] [US1] Write `book/questions/ch04.yaml` (≥ 8 questions covering S1.11–S1.13)
- [X] T078 [US1] Verify ch04 against docs (`mlflow-tracking`, MLflow 3 on Databricks, experiments UI pages) and run snippet sync + `book check`

### Chapter 02 — AutoML (S1.03, S1.04)

- [X] T079 [P] [US1] Write tests in `tests/ch02_automl/test_workspace.py`: `run_automl_classification(automl_module, df, target, timeout_minutes)` calls `databricks.automl.classify` with `dataset`, `target_col`, `primary_metric`, `timeout_minutes`, `exclude_cols` using a fake module; `summarize_automl(summary)` extracts `best_trial.mlflow_run_id`, `best_trial.metrics`, `experiment.experiment_id`; real-run test marked `databricks`
- [X] T080 [US1] Implement `src/db_ml_associate/ch02_automl/workspace.py` (injected `automl` module; classify/regress/forecast wrappers; result summary; `import_notebook` helper) with a module docstring stating AutoML requires Dedicated compute and is not built in from 18.0 ML (install `databricks-automl-runtime` per docs)
- [X] T081 [US1] Write `book/02-automl.md`: what AutoML does (data exploration notebook, trials with scikit-learn/XGBoost/LightGBM, best-trial notebook, MLflow experiment), how it facilitates model and feature selection (semantic type detection, imputation, feature importance/SHAP), advantages (glass-box notebooks, baseline speed, reproducibility), UI vs API, AutoML vs manual training (Choosing an approach), Platform change callout (packaging, compute), Requires callout; Pitfalls (treating AutoML as final model, data leakage columns, time column handling); Summary
- [X] T082 [P] [US1] Write `book/questions/ch02.yaml` (≥ 8 questions covering S1.03, S1.04)
- [X] T083 [US1] Verify ch02 against docs (`automl`, AutoML classification/regression/forecasting and API reference pages) and run snippet sync + `book check`

### Chapter 03 — Feature engineering in Unity Catalog (S1.05–S1.10)

- [X] T084 [P] [US1] Write tests in `tests/ch03_features/test_workspace.py` with a fake `FeatureEngineeringClient`: `create_customer_feature_table(fe, name, df)` calls `create_table(name=…, primary_keys=["customer_id"], df=…, description=…)`; `upsert_features` calls `write_table(name=…, df=…, mode="merge")`; `build_training_set` passes `FeatureLookup(table_name, feature_names, lookup_key="customer_id")`, `label`, `exclude_columns`; `log_feature_model` calls `fe.log_model(..., training_set=…, flavor=mlflow.sklearn, registered_model_name=…)`; `score` calls `score_batch(model_uri=…, df=…)`; `publish_online` calls `publish_table(online_store=…, source_table_name=…, online_table_name=…, publish_mode="TRIGGERED")`; real-workspace test marked `databricks`
- [X] T085 [P] [US1] Write tests in `tests/ch03_features/test_features.py`: `compute_customer_features(df)` (pandas and Spark, marker `spark`) produces one row per `customer_id`, no nulls in the key, deterministic aggregates
- [X] T086 [US1] Implement `src/db_ml_associate/ch03_features/features.py` (feature computation) and `workspace.py` (create/write/read/training set/log_model/score_batch/online store with injected client; SQL alternative strings `CREATE TABLE … CONSTRAINT … PRIMARY KEY` and `ALTER TABLE … ADD CONSTRAINT … PRIMARY KEY` as snippet regions)
- [X] T087 [US1] Implement `src/db_ml_associate/ch03_features/pitfalls.py` + tests: `wrong_feature_store_client` (legacy `FeatureStoreClient` — illustrative label, assert function documents deprecation) vs `correct_feature_engineering_client`; `wrong_nullable_primary_key` (validator rejects) vs correct; `wrong_modify_training_df_after_load` (features differ at inference) vs correct
- [X] T088 [US1] Write `book/03-feature-engineering-uc.md`: feature tables in UC vs workspace feature store (account-level governance, lineage, cross-workspace discovery — S1.05), creating a feature table (Python `create_table` = exam answer per R13b; SQL PK alternative), writing data (`mode="merge"`), training with `FeatureLookup`/`create_training_set`/`fe.log_model`, scoring with `score_batch` (and pyfunc), offline vs online feature tables (Lakebase Online Feature Store, `publish_table` modes, PK + Change Data Feed requirements — Platform change callout), Requires callout for online store cost/clean-up; Choosing an approach; Pitfalls; Summary
- [X] T089 [P] [US1] Write `book/questions/ch03.yaml` (≥ 8 questions covering S1.05–S1.10, one per objective minimum)
- [X] T090 [US1] Verify ch03 against docs (`feature-tables-uc`, `train-with-feature-store`, `online-feature-store`, feature engineering overview, Python API reference) and run snippet sync + `book check`

### Chapter 05 — Model lifecycle in Unity Catalog (S1.14, S1.15, S1.17, S1.18)

- [X] T091 [P] [US1] Write tests in `tests/ch05_registry/test_registry.py` with a fake `MlflowClient` (plus a real round trip against a temporary SQLite registry (`sqlite:///{tmp_path}/mlflow.db`) covering aliases and tags): `register(client, run_id, name)` calls `mlflow.register_model(f"runs:/{run_id}/model", "cat.sch.model")` or `client.create_model_version`; `set_tag`/`remove_tag` call `set_registered_model_tag`/`delete_registered_model_tag` and `set_model_version_tag`/`delete_model_version_tag`; `promote_challenger(client, name, metric)` compares `@champion` vs `@challenger` metrics and calls `set_registered_model_alias(name, "champion", version)` only when the challenger is better; three-level name validation rejects two-level names
- [X] T092 [US1] Implement `src/db_ml_associate/ch05_registry/registry.py` (`mlflow.set_registry_uri("databricks-uc")` note — default in MLflow 3, register, tags, aliases, load `models:/cat.sch.model@champion`, champion/challenger comparison)
- [X] T093 [US1] Implement `src/db_ml_associate/ch05_registry/pitfalls.py` + tests: `wrong_transition_stage` (stages unsupported in UC — illustrative, raises) vs `correct_set_alias`; `wrong_load_by_version_number_in_prod` vs `correct_load_by_alias`
- [X] T094 [US1] Write `book/05-model-lifecycle-uc.md`: registering with the MLflow Client API in UC (three-level names, signature required), UC registry vs workspace registry benefits (centralised ACLs, lineage, cross-workspace, audit), tags on models and versions (set/remove), aliases and champion/challenger promotion, Platform change callout (stages → aliases); Choosing an approach (UI vs API, alias vs version); Pitfalls; Summary
- [X] T095 [P] [US1] Write `book/questions/ch05.yaml` (≥ 8 questions covering S1.14, S1.15, S1.17, S1.18)
- [X] T096 [US1] Verify ch05 against docs (`model-lifecycle-uc`, upgrade-to-UC guide, MLflow Client API reference) and run snippet sync + `book check`

### Chapter 12 — Batch and streaming inference (S4.01, S4.03, S4.04)

- [X] T097 [P] [US1] Write tests in `tests/ch12_inference/test_batch.py`: `batch_predict_pandas(model_uri, pdf)` loads with `mlflow.pyfunc.load_model` and returns predictions aligned to input; `mapInPandas`/`mlflow.pyfunc.spark_udf` batch scoring on Spark matches pandas predictions (marker `spark`); `pipeline_source()` returns Lakeflow pipeline code using `from pyspark import pipelines as dp`, `@dp.table`/`@dp.materialized_view`, `spark.readStream`, and `mlflow.pyfunc.spark_udf` (string/AST check)
- [X] T098 [US1] Implement `src/db_ml_associate/ch12_inference/batch.py` (pandas, Spark UDF, `ai_query` SQL string example as snippet) and `streaming_pipeline.py` (Lakeflow declarative pipeline source applying the model as a Spark UDF to a streaming table; Structured Streaming `foreachBatch` alternative)
- [X] T099 [US1] Implement `src/db_ml_associate/ch12_inference/pitfalls.py` + tests: `wrong_load_model_per_row` (model loaded inside row loop; assert call count) vs `correct_load_once`; `wrong_column_order_mismatch` (schema enforcement error) vs correct
- [X] T100 [US1] Write `book/12-batch-and-streaming-inference.md`: batch vs streaming vs real-time (latency, cost, throughput, freshness — comparison table, S4.01), pandas batch inference, Spark UDF at scale, `ai_query`, streaming inference with Lakeflow pipelines (Platform change: formerly Delta Live Tables; exam tip per R13b sample Q5 — autoscaling pipeline applying model as Spark UDF), Structured Streaming alternative; Choosing an approach; Pitfalls; Summary
- [X] T101 [P] [US1] Write `book/questions/ch12.yaml` (≥ 8 questions covering S4.01, S4.03, S4.04)
- [X] T102 [US1] Verify ch12 against docs (`model-inference`, `lakeflow-pipelines`, `lakeflow-transform`, `lakeflow-python-ref`, batch inference pages) and run snippet sync + `book check`

### Chapter 13 — Real-time serving (S4.02, S4.05, S4.06)

- [X] T103 [P] [US1] Write tests in `tests/ch13_serving/test_serving.py` with a fake `WorkspaceClient`/deploy client: `endpoint_config(name, entities, split)` produces `served_entities` with `entity_name`, `entity_version`, `workload_size="Small"`, `scale_to_zero_enabled=True` and `traffic_config.routes` whose `traffic_percentage` sum to 100 (reject otherwise); `query_endpoint` sends `{"dataframe_split": {"columns": …, "data": …}}` and parses `predictions`; custom `mlflow.pyfunc.PythonModel` (e.g., churn model + business-rule post-processing) logs and predicts locally
- [X] T104 [US1] Implement `src/db_ml_associate/ch13_serving/custom_model.py` (PythonModel with `load_context`/`predict`, logged with signature) and `workspace.py` (create/update endpoint via `databricks-sdk` `w.serving_endpoints.create_and_wait` and REST equivalent; query via `mlflow.deployments.get_deploy_client("databricks").predict`; traffic split update; delete for clean-up)
- [X] T105 [US1] Implement `src/db_ml_associate/ch13_serving/pitfalls.py` + tests: `wrong_traffic_split_not_100` (validator raises) vs correct; `wrong_dataframe_records_column_order_assumption` vs `correct_dataframe_split`
- [X] T106 [US1] Write `book/13-real-time-serving.md`: Mosaic AI Model Serving overview (serverless, REST), deploying a custom (pyfunc) model to an endpoint (UI, SDK, REST), querying (REST `invocations`, input formats, MLflow deployments client, `ai_query`), splitting traffic between served models (A/B, canary; per-model invocation ignores split), Requires callout (cost, scale-to-zero, clean-up); Choosing an approach; Pitfalls; Summary
- [X] T107 [P] [US1] Write `book/questions/ch13.yaml` (≥ 8 questions covering S4.02, S4.05, S4.06)
- [X] T108 [US1] Verify ch13 against docs (`serve-multiple-models`, `query-custom-endpoints`, model serving overview, custom models page) and run snippet sync + `book check`

### Introduction, Appendix A and book assembly

- [X] T109 [US1] Write `book/introduction.md` (front matter, not a numbered chapter): exam facts (48 scored questions, 90 minutes, online proctored, no aids, 2-year validity; unscored items possible), how to register, how the book maps to the four sections (link coverage table), how to use chapters/tutorials/questions/plans, runtime target and docs snapshot date (`databricks_runtime`, `docs_snapshot` from `runtime.toml`), workspace options (Free Edition vs trial/paid, see research R14), exam-guide version and "check for a newer guide two weeks before your exam" (FR-021); recommended reading order (link to Appendix C)
- [X] T110 [US1] Implement Appendix A rendering in `src/db_ml_associate/cli/appendix.py` (`book render-appendix-a` writes the table from `platform-changes.yaml`) with tests in `tests/cli/test_appendix.py`, and generate `book/appendix-a-exam-vs-platform.md`
- [X] T111 [US1] Run `uv run db-ml-associate book coverage --write` to fill `book/README.md`; resolve every gap until all 48 objectives have ≥ 1 chapter section and ≥ 1 question (SC-001)
- [ ] T112 [US1] Run `uv run db-ml-associate book check --strict` and `uv run pytest`; fix all errors (SC-006, SC-007)

**Checkpoint**: The book alone is a complete study resource (MVP).

---

## Phase 4: User Story 2 — Practise hands-on with runnable tutorials (Priority: P2)

**Goal**: One Databricks notebook per chapter 01–13 that runs top to bottom on 18 LTS ML, with setup, test-runner and clean-up notebooks.

**Independent Test**: In a workspace on 18 LTS ML, run `notebooks/00_setup.py`, every `notebooks/chNN_*.py`, `notebooks/00_run_workspace_tests.py` and `notebooks/99_cleanup.py` without errors; locally `uv run pytest` passes with workspace tests skipped (spec US2, SC-004, SC-005).

- [ ] T113 [P] [US2] Write `tests/test_notebooks.py`: every `notebooks/*.py` starts with `# Databricks notebook source`; every chapter notebook `notebooks/ch[0-9][0-9]_*.py` except `ch12_pipeline_source.py` also has the header from [contracts/content-formats.md](contracts/content-formats.md#tutorial-notebook-header) (Objectives, Runtime, Requires, Free Edition, Estimated time, Clean-up); cells separated by `# COMMAND ----------`, compiles with `compile()` after stripping `# MAGIC` lines, and every chapter 01–13 front matter `tutorial:` path exists
- [ ] T114 [US2] Create `notebooks/00_setup.py`: widgets `catalog` (default `ml_associate`) and `schema` (default `tutorial`), `CREATE CATALOG/SCHEMA IF NOT EXISTS`, `%pip install -e` of the repo (or `sys.path` to `src`), `%pip install hyperopt==0.2.7` note, read `data/customer_churn.parquet` and `data/house_prices.parquet` from the Git folder (workspace files) and save them as `customer_churn` and `house_prices` Delta tables
- [ ] T115 [P] [US2] Create `notebooks/ch01_platform.py` (runtime info, Unity Catalog browsing, environment-as-catalog demo)
- [ ] T116 [P] [US2] Create `notebooks/ch06_exploring.py` (`.summary()`, `dbutils.data.summarize(df)`, data profile tab, visualisation editor instructions, feature comparisons)
- [ ] T117 [P] [US2] Create `notebooks/ch07_cleaning.py` (outliers by std-dev and IQR, imputation strategies, one-hot encoding, log transform, wrong-vs-correct pitfall cells)
- [ ] T118 [P] [US2] Create `notebooks/ch08_foundations.py` (imbalance mitigation, bias–variance sweep plots)
- [ ] T119 [P] [US2] Create `notebooks/ch09_pipelines.py` (Spark ML and scikit-learn training pipelines)
- [ ] T120 [P] [US2] Create `notebooks/ch11_evaluation.py` (metrics, evaluators, back-transforming log targets)
- [ ] T121 [P] [US2] Create `notebooks/ch10_tuning.py` (`%pip install hyperopt==0.2.7`, CrossValidator, Hyperopt `fmin` + `SparkTrials`, Optuna + joblib, model-count check against MLflow runs; detect serverless (no `spark.sparkContext`) and fall back to Hyperopt `Trials` / Optuna + joblib, printing why)
- [ ] T122 [P] [US2] Create `notebooks/ch04_tracking.py` (manual logging, autolog, MLflow UI tour checklist, best run via `MlflowClient.search_runs`)
- [ ] T123 [P] [US2] Create `notebooks/ch02_automl.py` (Requires: classic Dedicated compute; cost note (cluster DBUs for the experiment's `timeout_minutes`) + terminate-cluster clean-up; install note; `databricks.automl.classify` on churn, inspect data-exploration and best-trial notebooks; read-through section for Free Edition)
- [ ] T124 [P] [US2] Create `notebooks/ch03_features.py` (create/write feature table, training set, `fe.log_model` to `ml_associate.tutorial.churn_model`, `score_batch`; online store publish section with Requires + cost + clean-up)
- [ ] T125 [P] [US2] Create `notebooks/ch05_registry.py` (register, tags set/remove, `@champion`/`@challenger` aliases, promotion)
- [ ] T126 [P] [US2] Create `notebooks/ch12_inference.py` and `notebooks/ch12_pipeline_source.py` (pandas batch inference, Spark UDF, `ai_query`; Lakeflow pipeline source file plus instructions to create the pipeline and inspect the predictions streaming table)
- [ ] T127 [P] [US2] Create `notebooks/ch13_serving.py` (Requires: Model Serving; deploy custom pyfunc, query with `dataframe_split`, 80/20 traffic split between two versions, delete endpoint)
- [ ] T128 [US2] Create `notebooks/00_run_workspace_tests.py` (installs repo + test deps, runs `pytest -m databricks -p no:cacheprovider` against `tests/`, fails the notebook on non-zero exit) and `notebooks/99_cleanup.py` (delete serving endpoints, online tables via `WorkspaceClient().feature_store.delete_online_table`, online store, registered models, pipeline, schema — with confirmation widget)
- [ ] T129 [US2] Add the "Tutorial" box to each chapter 01–13 in `book/NN-*.md` (link to notebook, estimated time, Requires, Free Edition status, clean-up) and add `tutorial_minutes`/`requires`/`free_edition`/`cleanup`/`steps` metadata to chapter front matter per data-model.md Tutorial entity
- [ ] T130 [US2] Run all notebooks on a Databricks Runtime 18 LTS ML cluster (and on Free Edition serverless where applicable); confirm and record `SparkTrials` behaviour on serverless; record outcomes, actual runtimes and Free Edition status (`full|partial|read_through`) in `book/exam/tutorial-verification.md`; fix failures; mark any doc-vs-behaviour differences in `book/references.yaml` as `status: discrepancy` with a note
- [ ] T131 [US2] Capture UI screenshots referenced by chapters (MLflow UI, AutoML results, Catalog Explorer feature table, model aliases, serving endpoint traffic split) into `book/images/` and replace placeholders

**Checkpoint**: Book + runnable tutorials.

---

## Phase 5: User Story 3 — Follow a 2–3 week study plan (Priority: P2)

**Goal**: 14-day fast and 21-day standard plans with computed daily time budgets.

**Independent Test**: `book check` confirms every chapter, tutorial and both practice exams are scheduled, ≥ 2 days for practice exams + weak-area review, and every day is 90–180 minutes (spec US3, SC-002).

- [ ] T132 [P] [US3] Write `tests/content/test_plans.py`: plan minutes computed from chapter `reading_minutes`, `tutorial_minutes`, review-question time (2 min/question) and exam time (90 + 45 review); rules from data-model.md: "every chapter and tutorial appears ≥ 1 time", "both practice exams scheduled", "≥ 2 days for practice exams + weak-area review", "daily minutes within 90–180", fast ≤ 14 days, standard ≤ 21 days
- [ ] T133 [US3] Implement plan time computation and validation in `src/db_ml_associate/content/plans.py` and hook it into `book check`
- [ ] T134 [US3] Write `book/plans/standard-21.yaml` (≈ 2.4 h/day; reading order: introduction → 01 → 06 → 07 → 08 → 09 → 11 → 10 → 04 → 02 → 03 → 05 → 12 → 13 (respects every chapter's `prerequisites`), PE1 on day 18, weak-area review day 19, PE2 day 20, cheat-sheet revision day 21) with goals and self-checks per day
- [ ] T135 [P] [US3] Write `book/plans/fast-14.yaml` (≈ 3 h/day; same reading order as the standard plan; skim-able tutorials marked `optional: true` for chapters 08, 09 and 11; PE1 day 12, review + PE2 day 13, revision day 14)
- [ ] T136 [P] [US3] Write `tests/cli/test_plans_render.py`: `book render-plans` produces one table per track with a row per day (goals, linked items, computed minutes, self-check) and is idempotent
- [ ] T137 [US3] Implement `book render-plans` in `src/db_ml_associate/cli/plans.py` and generate `book/appendix-c-study-plans.md` (day table: goals, items with links, minutes, self-check)

**Checkpoint**: Learner can follow a scheduled path.

---

## Phase 6: User Story 4 — Assess readiness with practice exams (Priority: P3)

**Goal**: Two 48-question practice exams (18/9/15/6 per section), interactive timed CLI with per-section scoring, and Markdown renders.

**Independent Test**: `uv run db-ml-associate exam take PE1 --no-timer` presents 48 questions and reports per-section scores, 80 % readiness verdicts and chapter links for misses; `book check` confirms counts and originality (spec US4, FR-017–FR-019, SC-008).

- [ ] T138 [P] [US4] Write `tests/cli/test_exam.py`: scoring is all-or-nothing for `multiple` questions; per-section scores for sections 1–4; readiness verdict at 80 % per section; missed questions list `chapter_ref`; option shuffling deterministic for a given seed and answer mapping preserved; `--section N` filters; `render --answers` includes explanations and distractor reasons; simulated input via `CliRunner`
- [ ] T139 [US4] Implement `src/db_ml_associate/cli/exam.py` (`exam take` with rich prompts, countdown, `--no-timer`, `--section`, `--save` to `~/.db-ml-associate/attempts/<timestamp>.json`; `exam render [--answers]`)
- [ ] T140 [P] [US4] Write 48 practice questions for PE1 in `book/questions/practice-pe1.yaml` (ids `Q-PE1-001…048`, `pool: practice`, exactly one per objective S1.01–S4.06, ≥ 25 % `multiple` type, mixed difficulty)
- [ ] T141 [P] [US4] Write 48 practice questions for PE2 in `book/questions/practice-pe2.yaml` (ids `Q-PE2-001…048`, distinct scenarios from PE1 and chapter reviews, one per objective)
- [ ] T142 [US4] Create `book/exams/PE1.yaml` and `book/exams/PE2.yaml` (`minutes: 90`, ordered question lists referencing ids from `practice-pe1.yaml` / `practice-pe2.yaml`; section counts 18/9/15/6) and run `book check` (no reuse across exams, totals ≥ 200)
- [ ] T143 [US4] Generate `book/appendix-d-practice-exams.md` via `exam render` (both exams without answers, then answer keys with explanations and chapter links) and add a "score per section" worksheet table

**Checkpoint**: Learner can measure readiness.

---

## Phase 7: User Story 5 — Quick revision before the exam (Priority: P3)

**Goal**: Condensed cheat sheet linking every entry to its chapter.

**Independent Test**: `book/appendix-b-cheat-sheet.md` fits in ≈ 4 printed pages (≤ 1,600 words), and `book check` confirms every entry links to an existing chapter anchor (spec US5, FR-020).

- [ ] T144 [P] [US5] Write `tests/content/test_cheat_sheet.py`: word count ≤ 1,600; every table row contains a relative link to `book/NN-*.md#anchor` that resolves; includes sections "Key API calls", "Decision tables", "Formulas", "Exam vs current platform"
- [ ] T145 [US5] Write `book/appendix-b-cheat-sheet.md`: key API calls (FeatureEngineeringClient, MlflowClient registry/alias/tag, search_runs, fmin/SparkTrials, Optuna, CrossValidator, Imputer, OneHotEncoder, spark_udf, serving endpoint config/query), decision tables (imputation strategy, encoding by model type, metric by scenario, search strategy, serving mode, deploy code vs models, AutoML vs manual), formulas (precision/recall/F1, RMSE/MAE/R², IQR bounds, z-score, model count = grid × folds), renamed/deprecated list (from `platform-changes.yaml`), scenario lookup ("low-latency predictions with fresh features" → online store + Model Serving)

**Checkpoint**: All user stories complete.

---

## Phase 8: Polish & Cross-Cutting Concerns

- [ ] T146 Implement `book check-links [--update-verified]` in `src/db_ml_associate/cli/links.py` (urllib, HEAD then GET fallback, follow redirects, fail on 4xx/5xx or path change) with `network`-marked tests in `tests/cli/test_links.py` and offline unit tests using a fake opener; then add a `docs_links` job to `.github/workflows/smoke_test.yaml` on `schedule: cron "0 6 * * 1"` running `uv run pytest -m network --run-network`
- [ ] T147 Docs-coverage sweep: crawl each index page listed in [contracts/references.md](contracts/references.md) on docs.databricks.com/aws/en, record every sub-page in `book/exam/docs-coverage.md` as `covered` (chapter link) / `beyond-exam` / `out-of-scope` (reason) with `checked_on`; add newly relevant pages to `book/references.yaml` and open follow-up content edits in the affected chapters
- [ ] T148 Re-check the Databricks certification page for a newer ML Associate exam guide; update `book/exam/objectives.yaml` `checked_on` (and objectives + coverage if a new guide exists)
- [ ] T149 Run `uv run pytest -m network --run-network` / `book check-links`; fix moved URLs and update `last_verified`
- [ ] T150 [P] Rewrite `README.md`: project purpose (study guide + tutorials for the Databricks Certified ML Associate exam), quick start (devcontainer, `uv sync --all-groups`, `pytest`, `book check`, `exam take`), link to `book/README.md`, runtime target, remove template placeholder text and "First Steps When Using the Template Repo"
- [ ] T151 [P] Update `.specify/memory/constitution.md` Sync Impact Report follow-up TODOs: mark the Python-version mismatch and README placeholder as resolved (PATCH version bump via `/speckit-constitution` if required by governance)
- [ ] T152 [P] Add a "Runtime upgrade checklist" section to `docs/CONTRIBUTING.md`: (1) update `runtime.toml` and `pyproject.toml` pins from the new Runtime ML LTS release notes, (2) run `pytest` and `book check --strict`, (3) run all notebooks and `notebooks/00_run_workspace_tests.py` on the new runtime, (4) re-verify every `book/references.yaml` entry, (5) update `book/exam/platform-changes.yaml` and `book/exam/tutorial-verification.md` (constitution governance; SC-004)
- [ ] T153 Proofread all chapters for clarity (terms defined on first use, descriptive names, consistent exam-term vs current-term usage) and run `uv run db-ml-associate book sync-snippets --check`
- [ ] T154 Run full gates: `uv run pre-commit run --all-files`, `uv run pytest --cov=src`, `uv run bandit -c pyproject.toml -r src`, `uv run db-ml-associate book check --strict`
- [ ] T155 Execute [quickstart.md](quickstart.md) sections 1–5 end to end and record results in `book/exam/tutorial-verification.md`
- [ ] T156 Pilot walkthrough (quickstart section 6): one learner follows a plan, records actual minutes per day and practice-exam section scores; adjust chapter `reading_minutes`/`tutorial_minutes` and plans if actuals deviate > 20 %

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: none.
- **Foundational (Phase 2)**: depends on Setup; blocks all stories.
- **US1 (Phase 3)**: depends on Foundational. Chapter blocks are independent of each other except: ch11's back-transform section links ch07; ch02–05 and ch12–13 import the churn pipeline from ch09 and logging helpers from ch04, which is why ch09 and ch04 are written first; the introduction and Appendix A come last in the phase.
- **US2 (Phase 4)**: depends on Foundational and on the `src/` modules of the chapter whose notebook is written (US1 code tasks); chapter text not required.
- **US3 (Phase 5)**: depends on Foundational; final minute values need chapter front matter from US1/US2 (plans validate against whatever exists).
- **US4 (Phase 6)**: depends on Foundational; `chapter_ref` targets must exist (US1 chapter anchors) before `book check --strict` passes.
- **US5 (Phase 7)**: depends on US1 chapters (link targets).
- **Polish (Phase 8)**: after desired stories.

### Within each chapter block

Tests → code modules → pitfalls → chapter text (snippets synced from code) → questions → verify against docs + `book check`.

### Parallel Opportunities

- Phase 1: all `[P]` tasks after T001–T002 pins.
- Phase 2: all test-writing tasks in parallel (incl. loaders, coverage and datasets CLI tests); `objectives.yaml`, `references.yaml`, `platform-changes.yaml`, `datasets.py`, `snippets.py` in parallel.
- Phase 3: each chapter block is a separate set of files — different authors can take chapters in parallel after Foundational (ch09 and ch04 first if their helpers are needed); within a block, the tests task and questions task are `[P]`.
- Phase 4: all chapter notebooks `[P]` once `00_setup.py` exists.
- Phase 6: PE1 and PE2 question writing in parallel.

---

## Parallel Example: User Story 1

```text
Task: "Write tests in tests/ch08_foundations/test_imbalance.py and test_bias_variance.py"
Task: "Write tests in tests/ch09_pipelines/test_pipelines.py"
Task: "Write tests in tests/ch11_evaluation/test_metrics.py"
Task: "Write tests in tests/ch10_tuning/test_tuning.py"
Task: "Write book/questions/ch08.yaml"
Task: "Write book/questions/ch09.yaml"
```

## Parallel Example: User Story 2

```text
Task: "Create notebooks/ch06_exploring.py"
Task: "Create notebooks/ch07_cleaning.py"
Task: "Create notebooks/ch10_tuning.py"
Task: "Create notebooks/ch03_features.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1 Setup → Phase 2 Foundational.
2. Phase 3 (US1): write chapters in order 08 → 09 → 11 → 10 → 06 → 07 → 01 → 04 → 02 → 03 → 05 → 12 → 13, then the introduction + Appendix A.
3. **STOP and VALIDATE**: `book check --strict`, `book coverage` shows 48/48, `pytest` green.

### Incremental Delivery

1. Foundation → US1 (book MVP: learner can study all objectives).
2. US2 (hands-on tutorials verified on 18 LTS ML).
3. US3 (study plans) — makes the 2–3 week promise concrete.
4. US4 (practice exams) — readiness measurement.
5. US5 (cheat sheet) → Polish (docs sweep, link check, pilot).

### Continuous docs verification

Every chapter block ends with a "Verify against docs" task; Phase 8 adds the docs-coverage sweep, exam-guide re-check and link check; CI runs the link check weekly (user instruction).

---

## Notes

- `[P]` = different files, no dependency on incomplete tasks.
- Commit after each task or chapter block.
- Never present Hyperopt, DLT naming, workspace registry stages or `FeatureStoreClient` as the recommended path — always pair with the Platform change callout (Constitution II).
- Exam conventions from official samples (research R13b) override "technically more precise" answers in questions.
