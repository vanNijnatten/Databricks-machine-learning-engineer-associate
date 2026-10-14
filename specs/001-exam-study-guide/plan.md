# Implementation Plan: Exam-Aligned Study Guide and Tutorial for the Databricks ML Associate Exam

**Branch**: `001-exam-study-guide` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-exam-study-guide/spec.md`

## Summary

Write a 13-chapter Markdown study book (`book/`, chapters numbered in exam-guide order, plus an introduction and four appendices) covering all 48 objectives of the Databricks Certified ML Associate exam guide (2025-03-01), with a tested Python tutorial package (`src/db_ml_associate/`), Databricks source-format notebooks (`notebooks/`), a structured question bank (≥ 200 original questions), two 48-question practice exams with per-section scoring via a CLI, 14- and 21-day study plans, a cheat sheet, and an "exam vs. current platform" appendix. Everything targets **Databricks Runtime 18 LTS ML** (Python 3.12.3, Spark 4.1.0, MLflow 3.8.1) and is cross-checked against docs.databricks.com/aws/en through a references registry, a scheduled link check, a docs-coverage sweep, and a "verify against docs" step in every chapter task (user instruction). See [research.md](research.md).

## Technical Context

**Language/Version**: Python 3.12 (matches DBR 18 LTS ML's 3.12.3; project moves from 3.14 — research R2). Markdown (GFM) for the book; YAML for structured content.

**Primary Dependencies**: Pinned to the runtime (R1/R3): mlflow 3.8.1, scikit-learn 1.6.1, pandas 2.2.3, numpy 2.1.3, pyspark 4.1.0 (group `spark`), optuna 3.6.1 + hyperopt 0.2.7 (group `tuning`), databricks-feature-engineering 0.13.0.1 + databricks-sdk 0.67.0 (group `databricks`); CLI: typer, rich, pyyaml.

**Storage**: Files in the repo (Markdown, YAML, TOML, and small committed synthetic datasets in `data/*.parquet`, R9). In the workspace: Unity Catalog catalog/schema `ml_associate.tutorial` (Delta tables, feature tables, registered models), MLflow experiments, serving endpoints, Lakebase online store — all created and removed by notebooks.

**Testing**: pytest (+ pytest-mock, pytest-cov). Markers: `spark` (local Spark, skipped without Java), `databricks` (workspace; skipped unless `DATABRICKS_HOST`/`DATABRICKS_RUNTIME_VERSION`), `network` (docs link check; opt-in), plus existing `slow`, `data`, `configuration`, `user_interface`. Content tests call `book check`. Seeds fixed.

**Target Platform**: Databricks on AWS, Runtime 18 LTS ML (serverless where possible; classic Dedicated for AutoML); local Linux devcontainer with Java 17+.

**Project Type**: Documentation + teaching library + CLI (single project).

**Performance Goals**: Learner completes the material in ≈ 50 h (≤ 3 h/day, 14–21 days); local test suite < 5 min in CI; each tutorial notebook runs in ≤ 30 min of compute on a small cluster/serverless.

**Constraints**: No credentials or workspace IDs committed; small committed synthetic datasets only (R9); docs links are plain inline URLs registered in `references.yaml`; no deprecated API as the recommended path (Hyperopt/DLT taught as labelled exam-required items, R12); exam answer conventions from official samples (R13b); paid steps flagged with cost + clean-up.

**Scale/Scope**: introduction + 13 chapters (01–13) + 4 appendices; 13 tutorial notebooks + setup/test/cleanup notebooks; ≈ 13 chapter packages; ≥ 200 questions; ≈ 40–60 docs references.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle / Rule | How the plan complies | Status |
|---|---|---|
| I. Exam-aligned content | Objective registry with 48 IDs; chapter front matter maps objectives; `book check` enforces full coverage, learning objectives, summary, ≥ 8 review questions with explanations; "Beyond the exam" callout | ✅ |
| II. Runnable on current Databricks | `runtime.toml` single source (18 LTS ML); pins mirrored in `pyproject.toml` and tested; MLflow 3 + UC registry + FeatureEngineeringClient + Lakeflow pipelines as recommended paths; deprecated items labelled; every claim cites `references.yaml`; link check + re-verify cadence; Python 3.12 locally | ✅ (after R2 change) |
| III. Tested, high-quality code | ruff/format/mypy/bandit; typed + docstrings; tests mirror `src/`; workspace code behind `workspace.py` boundaries with fakes and `databricks` marker; snippet sync enforced; deterministic seeds | ✅ |
| IV. Clarity & multiple approaches | "Choosing an approach" sections (AutoML vs manual, Spark ML vs sklearn, grid/random/Bayesian, Hyperopt vs Optuna, batch/stream/real-time, deploy code vs models, UI vs API, Python vs SQL feature tables); `Exam tip` callout identifies expected approach | ✅ |
| V. Pitfalls explicit | Pitfalls section per chapter; `⚠️ Wrong approach` callouts; pitfall demos in `src/.../pitfalls.py` with tests asserting consequences | ✅ |
| Repo structure (`book/`, `src/db_ml_associate/`, `tests/`) | Followed; chapters `book/NN-*.md` numbered in exam-guide order (reading order lives in the study plans); introduction and appendices are unnumbered front/back matter; additions: `notebooks/`, `runtime.toml`, `data/` (justified below — not violations) | ✅ |
| Datasets rule | Tutorials use small, MIT-licensed synthetic files committed in `data/`; a test proves the generator reproduces them (R9) | ✅ |
| Markdown valid GFM, relative links | Plain inline links only; `book check` validates relative links/anchors and that docs URLs are registered | ✅ |
| Workflow gates | pre-commit, pytest in CI, chapter structure check, runtime verification step | ✅ |

**Pre-existing conflict resolved**: constitution follow-up TODO (Python ≥ 3.14 vs runtime 3.12) → R2 moves the project to 3.12. README placeholder text replaced as part of the book work.

**Post-design re-check (after Phase 1)**: data model and contracts encode every principle as a machine check (`book check` table in [contracts/cli.md](contracts/cli.md)); no new violations. ✅

**Re-check after `/speckit-analyze` (2026-10-14)**: resolved D1 (chapter order → exam-guide numbering), D2 (introduction is front matter, not a chapter), D3 (datasets committed as files), D4 (inline docs links). ✅

## Project Structure

### Documentation (this feature)

```text
specs/001-exam-study-guide/
├── plan.md              # This file
├── research.md          # Phase 0: decisions R1–R15
├── data-model.md        # Phase 1: content entities
├── quickstart.md        # Phase 1: validation guide
├── contracts/
│   ├── content-formats.md   # YAML/Markdown/snippet/notebook formats
│   ├── cli.md               # db-ml-associate commands
│   └── references.md        # docs references & continuous verification
├── checklists/requirements.md
└── tasks.md             # Phase 2 (/speckit-tasks)
```

### Source Code (repository root)

```text
runtime.toml                         # target runtime + library versions (single source)
data/                                # committed synthetic datasets (R9)
├── customer_churn.parquet  house_prices.parquet
└── README.md
book/
├── README.md                        # TOC + generated coverage table
├── introduction.md                  # front matter (not a numbered chapter)
├── 01-databricks-ml-platform.md     # Section 1
├── 02-automl.md
├── 03-feature-engineering-uc.md
├── 04-mlflow-tracking.md
├── 05-model-lifecycle-uc.md
├── 06-exploring-data.md             # Section 2
├── 07-cleaning-and-transforming.md
├── 08-ml-foundations.md             # Section 3
├── 09-pipelines.md
├── 10-validation-and-tuning.md
├── 11-evaluation.md
├── 12-batch-and-streaming-inference.md   # Section 4
├── 13-real-time-serving.md
├── appendix-a-exam-vs-platform.md
├── appendix-b-cheat-sheet.md
├── appendix-c-study-plans.md
├── appendix-d-practice-exams.md
├── exam/objectives.yaml | platform-changes.yaml | docs-coverage.md
├── questions/ch01.yaml … ch13.yaml, practice-pe1.yaml, practice-pe2.yaml
├── exams/PE1.yaml, PE2.yaml
├── plans/fast-14.yaml, standard-21.yaml
├── references.yaml
└── images/                          # UI screenshots (AutoML, MLflow UI, Serving)

notebooks/                           # Databricks source-format notebooks
├── 00_setup.py  00_run_workspace_tests.py  99_cleanup.py
└── ch01_platform.py … ch13_serving.py

src/db_ml_associate/
├── __init__.py  __main__.py         # typer app (existing entry point)
├── cli/  book_check.py  snippets.py  coverage.py  exam.py  links.py
├── content/                         # loaders + validators for YAML entities
├── datasets.py                      # generators + loaders for data/*.parquet
├── ch01_platform/   ch02_automl/    ch03_features/     ch04_tracking/
├── ch05_registry/   ch06_exploring/ ch07_cleaning/     ch08_foundations/
├── ch09_pipelines/  ch10_tuning/    ch11_evaluation/   ch12_inference/  ch13_serving/
│   └── (each: local logic modules, workspace.py boundary, pitfalls.py)

tests/
├── conftest.py                      # spark session fixture, marker auto-skip
├── test_content.py                  # runs book check, snippet sync, coverage
├── cli/  content/  test_datasets.py
└── ch01_platform/ … ch13_serving/   # mirror src/
```

Infra changes: `.python-version` → 3.12; `pyproject.toml` (requires-python, pins, groups, markers, ruff `target-version`); devcontainer base `python:3.12` + `default-jre-headless`; CI adds `setup-java`, then (once the commands exist) runs `book check` and a weekly scheduled `network` job.

**Structure Decision**: Single project. `book/` and `src/db_ml_associate/` follow the constitution's layout, chapter packages mirror chapter numbers, `notebooks/` holds the learner-facing tutorials that import the package, and `runtime.toml` satisfies "recorded in a single place".

## Implementation Phasing (input for /speckit-tasks)

1. **Foundation**: Python 3.12 + pins + groups + markers; devcontainer/CI Java; `runtime.toml`; objectives registry; content loaders/validators; `book check` skeleton; datasets generator; references registry seeded from research.
2. **US1 (P1) book core**: chapters 01–13 numbered in exam-guide order but *written* in dependency order 08 → 09 → 11 → 10 → 06 → 07 → 01 → 04 → 02 → 03 → 05 → 12 → 13, then the introduction and Appendix A; each chapter task = write → tutorial code + tests → notebook → review questions (≥ 8) → **verify against docs** → `book check`.
3. **US2 (P2) tutorials in workspace**: setup/cleanup/test-runner notebooks; run all on 18 LTS ML; record Free Edition status per tutorial.
4. **US3 (P2) study plans** with computed time budgets and the recommended reading order (respecting chapter `prerequisites`).
5. **US4 (P3) practice exams** (96 questions) + `exam take/render`.
6. **US5 (P3) cheat sheet** + Appendix A.
7. **Polish**: docs-coverage sweep, exam-guide recheck, link check, README rewrite, coverage table.

## Complexity Tracking

No constitution violations. Additions beyond the constitution's named directories, recorded for transparency:

| Addition | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| `notebooks/` directory | Learners run step-by-step tutorials in Databricks; exam APIs must be visible inline | Package-only code hides the exam-relevant calls across files |
| `data/` with committed Parquet files | Constitution datasets rule (files in the repository); identical locally and in the workspace | Generating at runtime is not "files in the repository" |
| `runtime.toml` | Principle II's "single place" for runtime/library versions, machine-checkable | Versions in prose drift from `pyproject.toml` |
| YAML content + CLI validators | Machine-verifiable coverage (SC-001/006/008) and practice-exam scoring (FR-018) | Pure Markdown cannot be validated or scored |
