<!--
Sync Impact Report
==================
Version change: (unversioned template) → 1.0.0
Bump rationale: initial ratification; all placeholders replaced with project-specific governance.

Modified principles (template placeholder → new title):
- [PRINCIPLE_1_NAME] → I. Exam-Aligned Study Content
- [PRINCIPLE_2_NAME] → II. Runnable on Current Databricks (NON-NEGOTIABLE)
- [PRINCIPLE_3_NAME] → III. Tested, High-Quality Code
- [PRINCIPLE_4_NAME] → IV. Clarity and Multiple Approaches
- [PRINCIPLE_5_NAME] → V. Pitfalls Made Explicit

Added sections:
- Content & Repository Structure (was [SECTION_2_NAME])
- Development Workflow & Quality Gates (was [SECTION_3_NAME])

Removed sections: none

Templates requiring updates:
- .specify/templates/plan-template.md ✅ no change needed (reads Constitution Check at runtime)
- .specify/templates/spec-template.md ✅ no change needed
- .specify/templates/tasks-template.md ✅ no change needed

Follow-up TODOs:
- RESOLVED 2026-10-15 (feature 001-exam-study-guide, T001/T003): Python version mismatch —
  pyproject.toml now requires Python >=3.12,<3.13 and .python-version is 3.12, matching
  Databricks Runtime 18 LTS ML (Python 3.12.3) recorded in runtime.toml.
- RESOLVED 2026-10-15 (feature 001-exam-study-guide, T150): README.md template placeholder text
  replaced with the project description and quick start.
No version bump: only this report changed; principles and sections are unchanged.
-->

# Databricks Machine Learning Associate Constitution

## Core Principles

### I. Exam-Aligned Study Content

The project exists to help a learner pass the Databricks Certified Machine Learning Associate
exam and become productive as an ML engineer on Databricks.

- Every chapter in `book/` and every tutorial module in `src/` MUST map to at least one domain
  or objective of the current official exam guide; the mapping MUST be stated at the top of the
  chapter or module.
- The book MUST cover every domain in the current exam guide; gaps MUST be tracked as open
  work, not silently omitted.
- Each chapter MUST state learning objectives up front and end with a summary and review
  questions (with answers and explanations) in the style of the exam.
- Content beyond the exam scope is allowed only when it is clearly marked as "beyond the exam"
  and directly supports practical ML engineering on Databricks.

**Rationale**: a study resource is only useful if a learner can see what is covered, why it
matters for the exam, and verify their own understanding.

### II. Runnable on Current Databricks (NON-NEGOTIABLE)

All code MUST work on the latest Databricks Runtime for Machine Learning (latest LTS by default)
and current Databricks platform features.

- The targeted Databricks Runtime ML version, its Python version, and key library versions
  (MLflow, Spark, scikit-learn, etc.) MUST be recorded in a single place in the repository and
  referenced from the book.
- Code MUST use current, non-deprecated APIs (e.g., Unity Catalog model registry, MLflow 3+
  APIs where applicable). Deprecated or legacy approaches MAY be shown only as a labelled
  pitfall or historical note, never as the recommended path.
- Tutorial code MUST NOT use Python language features or packages unavailable in the targeted
  runtime, regardless of the local development Python version.
- Claims about Databricks behavior MUST cite the official Databricks documentation; when the
  documentation and observed behavior differ, the discrepancy MUST be noted.
- When a new Databricks Runtime ML LTS is released, content MUST be re-verified and the
  recorded target version updated.

**Rationale**: outdated or broken examples actively harm learners and erode trust in the
material; the exam tests current platform behavior.

### III. Tested, High-Quality Code

Code in `src/` is teaching material and MUST be exemplary.

- All code MUST pass the configured linters, formatter, and type checker (ruff, ruff format,
  mypy, bandit) with no unjustified suppressions.
- Public functions MUST have type annotations and docstrings explaining purpose, parameters,
  and return values.
- Every tutorial module MUST have automated tests in `tests/`. Logic that can run locally
  (e.g., pandas, scikit-learn, pure Python, local Spark/MLflow) MUST be tested locally with
  pytest; logic that requires a Databricks workspace MUST be isolated behind clear boundaries,
  covered with mocks or fakes locally, and marked so it can be run against a real workspace.
- Code snippets embedded in `book/` MUST be sourced from, or kept in sync with, tested code in
  `src/`; untested snippets are not permitted unless explicitly marked as illustrative.
- Tests MUST be deterministic: random seeds fixed, no reliance on network access unless marked.

**Rationale**: learners copy what they see; untested or sloppy examples teach bad habits and
fail silently.

### IV. Clarity and Multiple Approaches

- Explanations MUST favor clarity over brevity: define terms on first use, prefer small focused
  examples, and build concepts incrementally.
- Where Databricks offers more than one way to accomplish a task (e.g., AutoML vs. manual
  training, pandas API on Spark vs. PySpark vs. pandas, Feature Engineering client vs. plain
  Delta tables, Hyperopt vs. Optuna, UI vs. API vs. SDK), the content MUST present the
  relevant alternatives side by side and state when to choose each, including trade-offs.
- The approach most likely to be expected on the exam MUST be identified explicitly.
- Names in code MUST be descriptive; clever or terse code is not acceptable in teaching
  material.

**Rationale**: the exam frequently asks which approach fits a scenario; understanding the
alternatives and their trade-offs is what distinguishes a competent engineer.

### V. Pitfalls Made Explicit

- Each chapter MUST include a "Pitfalls" section listing common mistakes, misconceptions, and
  exam traps for its topic (e.g., data leakage, fitting transformers before the split,
  forgetting to log signatures, mixing up model stages and aliases).
- Where practical, pitfalls MUST be demonstrated with code that shows the wrong approach, its
  observable consequence, and the corrected approach; tests MUST assert the demonstrated
  behavior so the pitfall stays accurate.
- Wrong-approach code MUST be clearly labelled so it cannot be mistaken for a recommendation.

**Rationale**: knowing what goes wrong, and why, makes knowledge durable and is heavily tested
by scenario-based exam questions.

## Content & Repository Structure

- `book/`: Markdown study guide. Chapters are numbered files (e.g., `book/01-<topic>.md`)
  ordered to follow the exam guide; a `book/README.md` MUST provide the table of contents and
  the exam-domain mapping.
- `src/db_ml_associate/`: the tutorial package. Modules MUST mirror the book's chapters so a
  learner can move between explanation and code. Code MUST be importable and runnable both
  locally (where possible) and in Databricks (as files in a Git folder or as a notebook
  importing the package).
- `tests/`: pytest suite mirroring `src/` structure. Tests that require a Databricks workspace
  MUST use a dedicated pytest marker and MUST be skipped automatically when no workspace is
  configured.
- Markdown MUST be valid CommonMark/GFM, use fenced code blocks with language tags, and use
  relative links between chapters and to source files.
- Datasets used in tutorials MUST be either Databricks-provided sample datasets
  (e.g., `/databricks-datasets`, `samples` catalog) or small, license-compatible files in the
  repository; no credentials, tokens, or workspace-specific identifiers may be committed.

## Development Workflow & Quality Gates

- All changes go through the Spec Kit flow (`/speckit-specify` → `/speckit-plan` →
  `/speckit-tasks` → `/speckit-implement`) for non-trivial content or code.
- Every plan MUST include a Constitution Check confirming compliance with Principles I–V.
- Before merging, the following gates MUST pass:
  1. `pre-commit` hooks (ruff check, ruff format, mypy) are clean.
  2. `pytest` passes locally and in CI, including coverage of new tutorial modules.
  3. New or changed chapters include learning objectives, exam-domain mapping, at least one
     comparison of approaches where applicable, a Pitfalls section, and review questions.
  4. Code and snippets are verified against the recorded target Databricks Runtime ML version.
- Reviewers MUST reject content that uses deprecated Databricks/MLflow APIs as the recommended
  path, contains untested code, or lacks the required chapter sections.

## Governance

- This constitution supersedes other project practices and guidelines. Where a template,
  CONTRIBUTING guide, or plan conflicts with it, the constitution wins.
- Amendments are made via `/speckit-constitution`, MUST include a Sync Impact Report, and MUST
  be reviewed in a pull request before merging.
- Versioning follows semantic versioning: MAJOR for removing or redefining principles, MINOR for
  adding principles or sections or materially expanding guidance, PATCH for clarifications and
  wording fixes.
- Compliance is reviewed on every pull request and during each `/speckit-plan` Constitution
  Check. Any deviation MUST be justified in the plan's Complexity Tracking section.
- A full content review against the current exam guide and latest Databricks Runtime ML LTS
  MUST be performed whenever either changes.

**Version**: 1.0.0 | **Ratified**: 2026-10-14 | **Last Amended**: 2026-10-14
