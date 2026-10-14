# Data Model: Exam-Aligned Study Guide and Tutorial

**Feature**: `001-exam-study-guide` | **Date**: 2026-10-14

The "data" of this feature is the book's structured content. Entities are stored as YAML or Markdown front matter in `book/` and validated by `db-ml-associate book check` (see [contracts/cli.md](contracts/cli.md)). File formats are defined in [contracts/](contracts/).

```text
Introduction / Appendices (FrontMatterPage, not chapters)
ExamGuide 1─* ExamObjective *─* ChapterSection *─1 Chapter 1─1 Tutorial
                    │                                  │
                    *                                  *
                 Question *───────────────* PracticeExam (by id)
                    │
Reference *─* ExamObjective / Chapter        PlatformChange *─* ExamObjective
StudyPlan 1─* StudyPlanDay *─* Chapter / Tutorial / PracticeExam
RuntimeTarget (singleton) ← referenced by Tutorial, Introduction
Dataset ← loaded by Tutorial notebooks and tests
```

## ExamGuide

| Field | Type | Rules |
|---|---|---|
| `version` | date | Effective date of the guide (`2025-03-01`) |
| `url` | URL | Official PDF |
| `questions` | int | 48 |
| `minutes` | int | 90 |
| `checked_on` | date | Last date the certification page was checked for a newer guide |

Stored at the top of `book/exam/objectives.yaml`.

## ExamObjective

| Field | Type | Rules |
|---|---|---|
| `id` | string | `S<1-4>.<NN>`, unique, stable (e.g., `S3.10`) |
| `section` | int | 1–4 |
| `text` | string | Verbatim from the exam guide |
| `kind` | enum | `conceptual` (identify/describe/compare) or `practical` (create/write/perform/deploy) |

**Validation**: exactly 48 objectives (18/9/15/6). Every objective MUST be referenced by ≥ 1 ChapterSection and ≥ 1 Question; every `practical` objective MUST be referenced by ≥ 1 Tutorial step (SC-001).

## Chapter

| Field | Type | Rules |
|---|---|---|
| `number` | int | 1–13, numbered in exam-guide order; file `book/NN-<slug>.md` |
| `title` | string | |
| `objectives` | list[ExamObjective.id] | ≥ 1 |
| `prerequisites` | list[int] | Chapters to read first; `book check` warns if a study plan schedules this chapter before them |
| `learning_objectives` | list[string] | ≥ 3 |
| `reading_minutes` | int | > 0 |
| `tutorial` | Tutorial ref | Required for every chapter (01–13) |
| `references` | list[Reference.id] | ≥ 1 |

**Required body sections (in order)**: Exam mapping, Learning objectives, content sections, "Choosing an approach" (when FR-004 applies), Pitfalls, Summary, Review questions (rendered from Question bank). Each content section tagged with objective IDs via `<!-- objectives: S2.05, S2.06 -->`.

## FrontMatterPage

`book/introduction.md` and `book/appendix-{a,b,c,d}-*.md`. Fields: `slug`, `title`. Not chapters: exempt from Chapter rules (no required front matter, headings or review questions); still subject to link checks.

## ChapterSection

Heading within a chapter carrying an objectives tag. Identified by `chapter#anchor`. Used to build the coverage table.

## Tutorial

| Field | Type | Rules |
|---|---|---|
| `chapter` | int | 1:1 with Chapter (01–13) |
| `notebook` | path | `notebooks/chNN_<slug>.py` |
| `extra_files` | list[path] | Further notebooks or pipeline sources (e.g., `notebooks/ch12_pipeline_source.py`); must exist |
| `modules` | list[path] | `src/db_ml_associate/chNN_<slug>/…` |
| `requires` | list[enum] | `unity_catalog`, `serverless`, `classic_ml_runtime`, `model_serving`, `online_store`, `lakeflow` |
| `free_edition` | enum | `full`, `partial`, `read_through` |
| `tutorial_minutes` | int | > 0 |
| `cleanup` | bool | MUST be true if `requires` contains `model_serving` or `online_store` |
| `steps` | list[{id, objectives}] | Each step lists objectives it exercises |

## Question

| Field | Type | Rules |
|---|---|---|
| `id` | string | `Q-<chapter|PE1|PE2>-<NNN>`, unique |
| `objective` | ExamObjective.id | Exactly one primary objective |
| `type` | enum | `single` or `multiple` |
| `stem` | string | Scenario-based |
| `options` | map A–E → string | 4–5 options |
| `answer` | list[letter] | Length 1 for `single`, ≥ 2 for `multiple` |
| `explanation` | string | Why correct |
| `distractors` | map letter → string | Why each wrong option is wrong (all non-answer options) |
| `chapter_ref` | Chapter#anchor | Where to revisit |
| `pool` | enum | `review` (chapter) or `practice` |
| `difficulty` | enum | `easy`, `medium`, `hard` |

**Validation**: originality (no verbatim official sample question text — checked by fuzzy match against the 5 samples); ≥ 8 `review` questions per chapter 01–13; total ≥ 200 (SC-008).

## PracticeExam

| Field | Type | Rules |
|---|---|---|
| `id` | string | `PE1`, `PE2` |
| `minutes` | int | 90 |
| `questions` | list[Question.id] | Exactly 48, all `pool: practice`, no reuse across exams |

**Validation**: per-section counts equal 18/9/15/6 (proportional to objective counts, FR-017). **Scoring**: per section `correct/total`; `multiple` questions scored all-or-nothing (as on the real exam); readiness threshold 80 % per section (SC-003).

## Reference

| Field | Type | Rules |
|---|---|---|
| `id` | string | slug |
| `url` | URL | `https://docs.databricks.com/aws/en/...` (or official exam guide) |
| `title` | string | |
| `claim` | string | One-line fact this reference supports |
| `objectives` | list[id] | |
| `chapters` | list[int] | Chapters citing this reference (1–13); ≥ 1 unless cited only by the introduction or appendices |
| `last_verified` | date | Warn if > 90 days old (R11) |
| `status` | enum | `ok`, `moved`, `discrepancy` (with `note`) |

## PlatformChange

| Field | Type | Rules |
|---|---|---|
| `exam_term` | string | e.g., "Delta Live Tables" |
| `current_term` | string | e.g., "Lakeflow pipelines (Spark Declarative Pipelines)" |
| `change` | enum | `renamed`, `deprecated`, `replaced`, `repackaged` |
| `exam_expects` | string | What to answer on the exam |
| `objectives` | list[id] | |
| `references` | list[Reference.id] | ≥ 1 |

Stored in `book/exam/platform-changes.yaml`; rendered into Appendix A. Minimum set: Hyperopt, Delta Live Tables, online tables, workspace model registry/stages, FeatureStoreClient, AutoML packaging, MLflow 3 `name`/LoggedModel (SC-007).

## StudyPlan / StudyPlanDay

| Field | Type | Rules |
|---|---|---|
| `track` | enum | `fast` (14 days) or `standard` (21 days) |
| `days[].day` | int | 1..N, contiguous |
| `days[].goals` | list[string] | ≥ 1 |
| `days[].items` | list[{introduction|chapter|tutorial|exam|review|appendix, optional: bool}] | |
| `days[].minutes` | int | Computed from item estimates; 90–180 |
| `days[].self_check` | string | |

**Validation**: every chapter and tutorial appears ≥ 1 time (tutorials may be `optional` only on the fast track for chapters marked skim-able); both practice exams scheduled; ≥ 2 days for practice exams + weak-area review; daily minutes within 90–180 (FR-016, SC-002).

## Dataset

| Field | Type | Rules |
|---|---|---|
| `name` | enum | `customer_churn`, `house_prices` |
| `path` | path | `data/<name>.parquet`, committed, < 500 KB |
| `rows` / `seed` | int | 5,000 / 3,000 rows; default seed 42 |

**Validation**: regenerating with `rows`/`seed` equals the committed file.

## RuntimeTarget (singleton)

`runtime.toml` at repo root: `databricks_runtime = "18 LTS ML"`, `python`, `spark`, and a `[libraries]` table (see research R1), plus `docs_snapshot` date. Referenced by the introduction and every notebook header; a test asserts `pyproject.toml` pins equal `runtime.toml`.

## State transitions

- **Reference.status**: `ok` → `moved` (URL redirects/404 found by link check) → `ok` (URL updated); `ok` → `discrepancy` (docs vs observed behaviour differ; note required) → `ok`.
- **Chapter lifecycle** (tracked in `book/README.md` coverage table): `planned` → `drafted` → `verified` (tutorial run on target runtime + docs re-checked) → `reviewed`.
