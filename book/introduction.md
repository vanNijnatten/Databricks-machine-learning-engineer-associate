# Introduction

This book prepares you for the **Databricks Certified Machine Learning Associate** exam in two to
three weeks. It explains every objective of the official exam guide, shows each technique in tested
code that runs on Databricks, warns you about the traps exam questions like to set, and lets you
measure your readiness with more than 200 practice questions.

## The exam

From the [official exam guide](https://www.databricks.com/sites/default/files/2025-02/databricks-certified-machine-learning-associate-exam-guide-1-mar-2025.pdf)
(version of 2025-03-01) and the
[certification page](https://www.databricks.com/learn/certification/machine-learning-associate):

| | |
|---|---|
| Questions | 48 scored multiple-choice or multiple-selection questions (plus possibly unscored items) |
| Time | 90 minutes |
| Fee | USD 200 |
| Delivery | Proctored, online or at a test centre; English, Japanese, Portuguese (BR), Korean |
| Aids | None |
| Validity | 2 years; recertify by taking the current exam |
| Registration | Through Databricks' exam delivery platform, linked from the certification page |
| Passing score | Not published |

The guide divides the exam into four sections:

| Section | Objectives | Chapters |
|---|---|---|
| 1. Databricks Machine Learning | 18 | [1](01-databricks-ml-platform.md), [2](02-automl.md), [3](03-feature-engineering-uc.md), [4](04-mlflow-tracking.md), [5](05-model-lifecycle-uc.md) |
| 2. Data Processing | 9 | [6](06-exploring-data.md), [7](07-cleaning-and-transforming.md) |
| 3. Model Development | 15 | [8](08-ml-foundations.md), [9](09-pipelines.md), [10](10-validation-and-tuning.md), [11](11-evaluation.md) |
| 4. Model Deployment | 6 | [12](12-batch-and-streaming-inference.md), [13](13-real-time-serving.md) |

The guide publishes no weights per section; with 18 objectives Section 1 is the largest. The
[coverage table](README.md#exam-objective-coverage) maps all 48 objectives to chapter sections,
tutorial steps and questions.

> **Check before you book:** this book targets the exam guide of **2025-03-01**, still the current
> version when the book was last checked (2026-10-14). Databricks asks candidates to check for a newer
> guide **two weeks before the exam** — if the objectives changed, compare them with
> [`exam/objectives.yaml`](exam/objectives.yaml).

## The platform this book targets

All code targets **Databricks Runtime 18 LTS for Machine Learning** (Python 3.12.3, Apache Spark
4.1.0, MLflow 3.8.1) and the Databricks documentation as of **2026-10-14**. The exact versions are in
[`runtime.toml`](../runtime.toml); every claim about Databricks links to the documentation page it
comes from.

The platform has moved on since the exam guide was written: Hyperopt is no longer preinstalled, Delta
Live Tables is now Lakeflow pipelines, online tables became Online Feature Stores, and AutoML is a
separate package. The book teaches what the exam expects **and** what you would use today; every such
difference is marked with a **Platform change** box and collected in
[Appendix A](appendix-a-exam-vs-platform.md).

## How each chapter works

- **Exam mapping** — the section and objective IDs the chapter covers.
- **Learning objectives** — what you can do after reading it.
- **Content** — explanations, tables and code. Code blocks are copied automatically from tested source
  files in [`src/`](../src/db_ml_associate/), so they always run.
- **Choosing an approach** — Databricks often offers several ways to do something; this section says
  which to pick when, and which one the exam expects.
- **Pitfalls** — common mistakes and exam traps, many demonstrated with a wrong and a correct version
  of the code and the measured consequence.
- **Summary** and **review questions** — answer them before opening the answers; each explanation
  says why every wrong option is wrong.

Call-outs you will meet:

> **Exam tip:** what the exam is most likely to ask or expect.

> **Platform change:** where the exam's wording and today's platform differ.

> **Requires:** a feature that needs classic compute, Model Serving or other paid capabilities, with
> cost and clean-up notes.

> **Beyond the exam:** useful background the exam does not test.

## Setting up a workspace

The tutorials (`notebooks/`) run in a Databricks workspace with **Unity Catalog**. You need to be able
to create a catalog and schema (the default is `ml_associate.tutorial`).

| Workspace | What works |
|---|---|
| **Trial or company workspace with classic compute** | Everything. Create compute with the **Machine learning** checkbox and Databricks Runtime **18 LTS ML** (Dedicated access mode). |
| **Free Edition** (serverless only) | Most tutorials, using serverless notebooks with the **ML** base environment: data exploration, cleaning, foundations, pipelines, evaluation, MLflow, the model registry, offline feature tables and batch inference. Not available: AutoML (needs classic compute), online feature stores, and Hyperopt `SparkTrials` (use `Trials` or Optuna instead). Model Serving is limited to a few CPU endpoints ([Free Edition limitations](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)). |

Each tutorial notebook starts with a header that lists what it requires, whether it works on Free
Edition, its estimated time and the clean-up it needs.

To set up:

1. Add this repository to your workspace as a **Git folder**.
2. Run `notebooks/00_setup.py`. It creates the catalog and schema and loads the two tutorial datasets
   ([`data/README.md`](../data/README.md)) into Delta tables.
3. Follow the chapters' tutorials; finish with `notebooks/99_cleanup.py` to delete endpoints, online
   stores and registered models so nothing keeps costing money.

To run the tested code locally, open the repository in its devcontainer and run `uv sync --all-groups`
and `uv run pytest`.

## How to study

You do not have to read the chapters in numeric order. Chapters are numbered like the exam guide's
sections so you can find things, but Section 1 assumes you can already train and evaluate a model.
The [study plans](appendix-c-study-plans.md) give the **recommended reading order** — foundations
first:

> Introduction → 1 → 6 → 7 → 8 → 9 → 11 → 10 → 4 → 2 → 3 → 5 → 12 → 13 → practice exams

Pick the **21-day plan** (about 2½ hours a day) or the **14-day fast track** (about 3 hours a day).
Then:

1. Take both [practice exams](appendix-d-practice-exams.md) under exam conditions (90 minutes, no
   notes). The `db-ml-associate exam take` command times you and scores each section.
2. Revisit the chapters linked to every question you missed.
3. Revise with the [cheat sheet](appendix-b-cheat-sheet.md) and [Appendix A](appendix-a-exam-vs-platform.md).
4. Book the exam when you score at least **80 % in every section** of both practice exams — the
   official pass mark is not published, so leave a margin.

## Summary

- 48 questions, 90 minutes, four sections; the book maps every objective to chapters, tutorials and
  questions.
- The code targets Databricks Runtime 18 LTS ML; platform changes since the exam guide are flagged.
- Follow a study plan, run the tutorials, and book the exam once you score 80 % per section in the
  practice exams.
