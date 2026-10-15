# Contract: Content File Formats

Formats for the structured book content. Entity fields and validation rules are defined in [../data-model.md](../data-model.md); this file fixes the on-disk shape. All YAML is UTF-8, 2-space indent, validated by `db-ml-associate book check`.

## `book/exam/objectives.yaml`

```yaml
exam_guide:
  version: 2025-03-01
  url: https://www.databricks.com/sites/default/files/2025-02/databricks-certified-machine-learning-associate-exam-guide-1-mar-2025.pdf
  questions: 48
  minutes: 90
  checked_on: 2026-10-14
objectives:
  - id: S1.01
    section: 1
    kind: conceptual
    text: Identify the best practices of an MLOps strategy
  # ... 48 entries
```

## Chapter front matter (`book/NN-<slug>.md`, NN = 01–13 in exam-guide order)

`book/introduction.md` and `book/appendix-*.md` are front/back matter: no required front matter or headings.

```markdown
---
chapter: 10
title: Validation and hyperparameter tuning
objectives: [S3.05, S3.06, S3.07, S3.08, S3.09, S3.10]
prerequisites: [8, 9, 11]
reading_minutes: 120
tutorial:
  notebook: notebooks/ch10_tuning.py
  minutes: 75
  requires: [serverless]          # unity_catalog | serverless | classic_ml_runtime | model_serving | online_store | lakeflow
  free_edition: partial           # full | partial | read_through
  cleanup: false                  # must be true when requires model_serving or online_store
  extra_files: []
  steps:
    - id: grid-search-cv
      objectives: [S3.06, S3.09, S3.10]
references: [hyperparameter-tuning, spark-ml-tuning]
---

# 10. Validation and hyperparameter tuning

> **Exam mapping**: S3.05 … S3.10 · **Runtime**: see [runtime](../runtime.toml)

## Learning objectives
...
## Cross-validation vs. train/validation split
<!-- objectives: S3.08, S3.09 -->
...
## Choosing an approach
...
## Pitfalls
...
## Summary
...
## Review questions
<!-- questions: ch10 -->
```

- Required H2 headings, in order: `Learning objectives`, (content), `Pitfalls`, `Summary`, `Review questions`. `Choosing an approach` required when the chapter's objectives involve alternatives (FR-004).
- `<!-- questions: chNN -->` is replaced by rendered review questions (answers in a collapsible `<details>` block).
- Labels (blockquote callouts, GFM-compatible):
  - `> **Exam tip:**` — exam-expected approach.
  - `> **Platform change:**` — renamed/deprecated item (links Appendix A).
  - `> **Beyond the exam:**` — out-of-scope material (Principle I).
  - `> **⚠️ Wrong approach:**` — precedes pitfall code (Principle V).
  - `> **Requires:**` — paid/classic features, cost and clean-up (FR-012).

## Snippet markers

Source (`src/**/*.py`, `notebooks/*.py`):

```python
# --8<-- [start:impute_median]
...
# --8<-- [end:impute_median]
```

Book:

````markdown
<!-- snippet: src/db_ml_associate/ch07_cleaning/impute.py#impute_median -->
```python
...synced content...
```
````

- A fenced block immediately following a `snippet` comment MUST equal the marked region (dedented). `<!-- illustrative -->` exempts the next block.
- Region names unique per file.

## `book/questions/chNN.yaml`, `book/questions/practice-pe1.yaml` and `book/questions/practice-pe2.yaml`

```yaml
- id: Q-10-003
  objective: S3.10
  pool: review
  type: single
  difficulty: medium
  stem: >
    A data scientist runs GridSearchCV with 4 values of max_depth and
    3 values of n_estimators using 3-fold cross-validation.
    How many models are trained during the search?
  options:
    A: "12"
    B: "36"
    C: "7"
    D: "None of the above"
  answer: [B]
  explanation: >
    4 × 3 = 12 combinations, each trained once per fold: 12 × 3 = 36.
    This follows the exam's convention (official sample question: 18 × 5 = 90);
    the optional final refit on the full training set is not counted.
  distractors:
    A: Counts combinations only, ignoring folds.
    C: Adds grid sizes instead of multiplying them.
    D: Would only be right if the final refit were counted (37), which the exam does not do.
  chapter_ref: 10-validation-and-tuning.md#counting-models
```

## `book/exams/PE1.yaml`

```yaml
id: PE1
minutes: 90
questions: [Q-PE1-001, Q-PE1-002, ...]   # exactly 48
```

## `book/exam/platform-changes.yaml`

```yaml
- exam_term: Hyperopt fmin / SparkTrials
  current_term: Optuna (single node, joblib-parallel) or Ray Tune (distributed)
  change: deprecated
  exam_expects: Hyperopt fmin with tpe.suggest; SparkTrials to parallelise single-node models
  objectives: [S3.05, S3.06, S3.07]
  references: [hyperparameter-tuning]
```

## Docs links

Plain inline Markdown links to registered URLs, e.g. `[Hyperparameter tuning](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/)` (see [references.md](references.md)).

## `book/plans/standard-21.yaml` / `fast-14.yaml`

```yaml
track: standard
days:
  - day: 1
    goals: [Understand exam format, set up workspace]
    items:
      - introduction: true
      - chapter: 1
      - tutorial: notebooks/ch01_setup.py
    self_check: Can you explain deploy-code vs deploy-models in two sentences?
```

`minutes` is computed, not authored.

## `runtime.toml`

```toml
databricks_runtime = "18 LTS ML"
docs_snapshot = 2026-10-14
python = "3.12.3"
spark = "4.1.0"

[libraries]
mlflow = "3.8.1"
scikit-learn = "1.6.1"
pandas = "2.2.3"
numpy = "2.1.3"
optuna = "3.6.1"
hyperopt = "0.2.7"   # not preinstalled; %pip install
databricks-feature-engineering = "0.13.0.1"
databricks-sdk = "0.67.0"
```

## Tutorial notebook header (`notebooks/chNN_<slug>.py`)

Required for chapter notebooks only. Setup, test-runner, clean-up and pipeline-source files (`00_*.py`, `99_*.py`, `ch12_pipeline_source.py`) need only the `# Databricks notebook source` first line.

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter 10 tutorial: Validation and hyperparameter tuning
# MAGIC **Objectives**: S3.05–S3.10 · **Runtime**: 18 LTS ML · **Requires**: serverless or classic · **Free Edition**: full
# MAGIC **Estimated time**: 75 min · **Clean-up**: none
```
