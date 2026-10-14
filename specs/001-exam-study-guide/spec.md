# Feature Specification: Exam-Aligned Study Guide and Tutorial for the Databricks ML Associate Exam

**Feature Branch**: `001-exam-study-guide`

**Created**: 2026-10-14

**Status**: Draft

**Input**: User description: "I would like you to analyse the study guide for the databricks Machine Learning Associate exam on `https://www.databricks.com/sites/default/files/2025-02/databricks-certified-machine-learning-associate-exam-guide-1-mar-2025.pdf`. Find which sections match the sections that are neccessary to pass the exam. Find the most recent documentation on `https://docs.databricks.com/aws/en`, and determine what is neccessary to write a book or extensive study guide and tutorial to pass the exam. It should be possible to pass the exam in about two or three weeks time."

## Context: Exam Analysis Summary

The official exam guide (live version as of 2025-03-01, still the most recent official guide on 2026-10-14) describes:

- **Format**: 48 scored multiple-choice / multiple-selection questions, 90 minutes, online proctored, no test aids, valid 2 years.
- **Recommended background**: working knowledge of Python, scikit-learn, Spark ML, Unity Catalog and Delta Live Tables; Databricks Academy courses on data preparation, model development, deployment and MLOps.
- **Four exam sections** containing **48 objectives** in total (Section 1: 18, Section 2: 9, Section 3: 15, Section 4: 6). The guide does not publish section weights; third-party sources disagree, so the guide MUST treat Section 1 and Section 3 as the largest by objective count.

### Exam objective → documentation area map

Every exam objective falls into one of these study areas, which together define the minimum content of the guide:

| Exam section | Objectives (summarised) | Current documentation area (docs.databricks.com/aws/en) |
|---|---|---|
| 1. Databricks Machine Learning | MLOps best practices; promoting code vs. promoting models | MLOps workflows / "deploy code" vs "deploy models" patterns |
| 1 | Advantages of ML runtimes | Databricks Runtime for Machine Learning (release notes, included libraries) |
| 1 | How AutoML facilitates model/feature selection; AutoML advantages | AutoML (classification, regression, forecasting; generated notebooks; MLflow trials) |
| 1 | Feature tables in Unity Catalog vs. workspace; create, write, train with, score with feature tables; online vs. offline | Feature engineering in Unity Catalog; online feature stores (Lakebase) |
| 1 | Find best run with MLflow Client; manually log metrics/artifacts/models; MLflow UI information | MLflow tracking (experiments, runs, Client API, UI) |
| 1 | Register a model in Unity Catalog via MLflow Client; UC registry benefits vs. workspace registry; set/remove model tags; champion/challenger aliases | Manage model lifecycle in Unity Catalog |
| 2. Data Processing | Summary statistics (`.summary()`, dbutils data summaries); visualisations; comparing two features | Exploratory data analysis, notebook data profiles and visualisations |
| 2 | Outlier removal (standard deviation, IQR) | Spark DataFrame operations; general ML foundations |
| 2 | Imputation (mean / median / mode) and when to choose each | Spark ML / scikit-learn feature preprocessing |
| 2 | One-hot encoding and when it is (in)appropriate; log-scale transforms | Spark ML / scikit-learn feature preprocessing; general ML foundations |
| 3. Model Development | Algorithm selection; data imbalance mitigation; bias–variance trade-off | General ML foundations |
| 3 | Estimators vs. transformers; training pipelines | Spark ML pipelines; scikit-learn pipelines |
| 3 | Hyperopt `fmin`; random / grid / Bayesian search; parallelising single-node tuning | Hyperparameter tuning (Hyperopt — deprecated; Optuna/Ray Tune recommended) |
| 3 | Cross-validation vs. train–validation split; CV in fitting; counting models trained in grid search × CV | Spark ML tuning (CrossValidator, TrainValidationSplit); scikit-learn model selection |
| 3 | Classification and regression metrics; choosing a metric; exponentiating log-transformed targets | Model evaluation (MLflow evaluate, scikit-learn / Spark ML evaluators) |
| 4. Model Deployment | Batch vs. real-time vs. streaming serving | Model inference overview |
| 4 | Batch inference with pandas | Batch inference (pandas, Spark UDFs from MLflow models) |
| 4 | Streaming inference with Delta Live Tables | Lakeflow pipelines (Spark Declarative Pipelines, formerly Delta Live Tables) |
| 4 | Deploy a custom model to an endpoint; deploy and query for real-time inference; split traffic between served models | Mosaic AI Model Serving (custom model endpoints, served entities, traffic split, querying) |

### Drift between the exam guide and the current platform

The guide was written for the platform as of early 2025. As of 2026-10-14 the documentation shows these changes, which the study guide MUST reconcile explicitly:

- **Hyperopt** is deprecated and no longer pre-installed in recent Databricks Runtime ML versions; Databricks recommends Optuna (single node) or Ray Tune (distributed). The exam still names Hyperopt `fmin`.
- **Delta Live Tables** has been renamed to Lakeflow pipelines (Spark Declarative Pipelines, formerly Delta Live Tables). The exam still uses the old name.
- **Online tables** for feature serving have been replaced by Online Feature Stores backed by Lakebase.
- **AutoML** is no longer a built-in library starting with Databricks Runtime 18.0 ML and requires dedicated (classic) compute.
- **MLflow 3** is the current major version and defaults the model registry to Unity Catalog.
- **Latest Databricks Runtime ML LTS** is 18 LTS (released 2026-06-10, MLflow 3.8.x).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Learn every exam objective from one book (Priority: P1)

A learner with basic Python and ML knowledge, preparing for the exam, reads the book chapter by chapter. Each chapter tells them which exam objectives it covers, explains the concepts with current Databricks behaviour, highlights the alternatives and exam-preferred approach, lists pitfalls, and ends with exam-style review questions with explained answers.

**Why this priority**: The written guide is the core value. Without full objective coverage the learner cannot rely on it to pass, regardless of any tutorials or plans.

**Independent Test**: Take the 48 objectives from the exam guide and confirm each one is addressed in a named chapter section with explanation, at least one example, and at least one review question.

**Acceptance Scenarios**:

1. **Given** the official exam outline, **When** a reviewer looks up any objective in the book's coverage table, **Then** they find the chapter and section where it is taught.
2. **Given** a chapter, **When** the learner reaches its end, **Then** they find a summary, a Pitfalls section, and at least eight exam-style questions with answers and explanations of why the wrong options are wrong.
3. **Given** a topic where the exam wording differs from the current platform (Hyperopt, Delta Live Tables, online tables, workspace registry), **When** the learner reads that topic, **Then** both the exam terminology and the current recommended approach are explained, and the learner is told which one the exam expects.

---

### User Story 2 - Practise hands-on with runnable tutorials (Priority: P2)

The learner follows a tutorial for each chapter in a Databricks workspace, running the code that accompanies the book: exploring and cleaning a dataset, building feature tables, tracking experiments, tuning and evaluating models, registering them in Unity Catalog, promoting with aliases, and deploying for batch, streaming and real-time inference.

**Why this priority**: The exam expects hands-on familiarity (APIs, UI, method names). Practical exercises make the theory stick, but the book alone still delivers value without them.

**Independent Test**: A learner with a suitable workspace runs each chapter's tutorial end to end on the recorded target runtime and obtains the results described in the book; locally-runnable parts pass the automated test suite.

**Acceptance Scenarios**:

1. **Given** a workspace on the recorded target runtime, **When** the learner runs a chapter's tutorial in order, **Then** every step completes and produces the outputs shown in the book.
2. **Given** the repository on a developer machine without a workspace, **When** the test suite runs, **Then** all local tests pass and workspace-dependent tests are skipped with a clear reason.
3. **Given** a tutorial step that needs paid or classic-only features (e.g., model serving endpoints, AutoML on dedicated compute, online feature store), **When** the learner reads it, **Then** the requirement and the approximate cost/clean-up steps are stated before the step.

---

### User Story 3 - Follow a 2–3 week study plan (Priority: P2)

A learner with limited time follows a day-by-day plan that sequences reading, hands-on work, review and practice exams over 14 days (fast track) or 21 days (standard track), at roughly 1.5–3 hours per day.

**Why this priority**: The user's stated goal is to be exam-ready in two to three weeks; a plan turns the content into a schedule. It depends on the book existing.

**Independent Test**: Sum the estimated time of each plan day's activities and confirm every chapter, tutorial and practice exam is scheduled and the total fits within the chosen track.

**Acceptance Scenarios**:

1. **Given** the 14-day or 21-day track, **When** the learner follows it, **Then** every chapter and tutorial is scheduled at least once and at least two days are reserved for full practice exams and weak-area review.
2. **Given** a day in the plan, **When** the learner reads it, **Then** they see its goals, the chapters/tutorials to complete, the estimated time, and a self-check to confirm the day is done.

---

### User Story 4 - Assess readiness with practice exams (Priority: P3)

Before booking the exam, the learner takes full-length timed practice exams that mirror the real format and get a per-section score showing which areas to revisit.

**Why this priority**: Practice under exam conditions measures readiness and directs final review, but it builds on the chapters and their questions.

**Independent Test**: Take a practice exam; confirm it has 48 questions in 90 minutes, its question mix matches the objective counts per section, and each answer links back to the chapter covering it.

**Acceptance Scenarios**:

1. **Given** a practice exam, **When** the learner completes it, **Then** they can compute a score per exam section and see which chapter to revisit for each missed question.
2. **Given** the practice question bank, **When** a reviewer checks it, **Then** every exam objective is covered by at least one practice question and no question duplicates an official sample question verbatim.

---

### User Story 5 - Quick revision before the exam (Priority: P3)

On the last days before the exam the learner uses a condensed cheat sheet: key API calls and their purpose, decision tables (which approach for which scenario), formulas (metrics, model-count calculations), and the list of renamed/deprecated terms.

**Why this priority**: A useful accelerator for the final review, but only valuable once the full material is learned.

**Independent Test**: Confirm the cheat sheet fits in a few printed pages and each entry links to the chapter where it is explained.

**Acceptance Scenarios**:

1. **Given** the cheat sheet, **When** the learner looks up a scenario (e.g., "low-latency predictions with fresh features"), **Then** they find the recommended approach and the chapter reference.

---

### Edge Cases

- **Exam guide is updated** after the book is written: the book records the exam-guide version it targets, and a new guide version triggers a coverage review (new/removed objectives tracked as open work).
- **Exam asks about deprecated tools** (Hyperopt, Delta Live Tables naming, workspace model registry, online tables): content teaches what the exam expects, labels it as deprecated/renamed, and shows the current replacement.
- **Learner has no paid workspace** (e.g., only a free/trial account with serverless-only compute): the book states which tutorials work there and which need classic compute or paid features, and provides read-through alternatives (screenshots/expected output) for those that do not.
- **Runtime changes** (e.g., a newer LTS removes a library used in a tutorial): the target runtime is recorded in one place; tutorials that need extra libraries state the install step.
- **Learner lacks ML fundamentals**: chapters on general ML (metrics, bias–variance, imputation, encoding) are self-contained so a learner can start there without prior Databricks knowledge.
- **Ambiguous third-party information** (question counts, section weights, pass mark): the book uses only the official guide's figures and labels anything else as unofficial.

## Requirements *(mandatory)*

### Functional Requirements

**Coverage and structure**

- **FR-001**: The book MUST cover all 48 objectives of the current official exam guide, and include a coverage table mapping each objective to the chapter section(s), tutorial(s) and practice questions that address it.
- **FR-002**: The book MUST be organised into numbered chapters following the four exam sections in exam-guide order, preceded by an unnumbered introduction (exam format, registration, how to use the book, workspace setup, recommended reading order) and followed by appendices (exam vs. current platform, cheat sheet, study plans, practice exams). The introduction and appendices are not chapters and are exempt from FR-003.
- **FR-003**: Each chapter MUST begin with its learning objectives and the exam objectives it maps to, and MUST end with a summary, a Pitfalls section, and at least eight exam-style review questions with answers and explanations.
- **FR-004**: Where Databricks offers more than one approach to an objective (e.g., AutoML vs. manual training; Spark ML vs. scikit-learn; grid vs. random vs. Bayesian search; batch vs. streaming vs. real-time serving; promoting code vs. promoting models; UI vs. API), the book MUST compare the alternatives, state when to choose each, and identify the approach the exam is most likely to expect.
- **FR-005**: For every term or tool where the exam guide differs from current Databricks documentation, the book MUST name both, explain the relationship (renamed, deprecated, replaced), and state which one the exam expects; a consolidated "exam vs. current platform" reference MUST be included.
- **FR-006**: Every factual claim about Databricks behaviour MUST cite the relevant page of the official Databricks documentation.
- **FR-007**: General ML foundations required by the exam (algorithm selection, data imbalance, imputation strategies, one-hot encoding suitability, log transforms and back-transformation, outlier removal by standard deviation and IQR, cross-validation, classification and regression metrics, bias–variance trade-off) MUST be explained with worked examples, independent of any Databricks feature.
- **FR-008**: The book MUST explain how to count models trained in combined grid-search and cross-validation scenarios as combinations × folds (the convention used by the official sample question), and explain the optional final refit as a beyond-the-exam nuance, with worked examples.

**Tutorials**

- **FR-009**: Each chapter (01–13) MUST have an accompanying hands-on tutorial that exercises the chapter's practical objectives, using Databricks-provided sample data or small, licence-compatible dataset files committed in the repository.
- **FR-010**: Tutorials MUST run on the latest Databricks Runtime ML LTS recorded in the repository (currently 18 LTS), and the target runtime and key library versions MUST be recorded in a single place referenced by the book.
- **FR-011**: Tutorials MUST cover, at minimum: summary statistics and data profiles; outlier removal; imputation; one-hot encoding; feature table creation, writing, training and batch scoring in Unity Catalog; publishing to an online store; AutoML experiment and inspection of its outputs; manual MLflow logging and finding the best run via the Client API; model registration in Unity Catalog, tags and champion/challenger aliases; training pipelines with cross-validation and hyperparameter search (including Hyperopt `fmin` and parallelised single-node tuning); evaluation metrics; batch inference with pandas; streaming inference in a declarative pipeline; deploying, querying and traffic-splitting a model serving endpoint.
- **FR-012**: Each tutorial step that requires features beyond a basic workspace (classic/dedicated compute, model serving, online feature store, AutoML) MUST state the requirement, expected cost implications, and clean-up steps before the step is run.
- **FR-013**: Parts of tutorials that can run without a workspace MUST be verifiable by an automated local test suite; workspace-dependent parts MUST be skippable automatically when no workspace is configured.
- **FR-014**: Code shown in the book MUST be identical to, or kept in sync with, tested tutorial code; illustrative-only snippets MUST be labelled as such.
- **FR-015**: Pitfalls that can be demonstrated in code (e.g., data leakage from fitting transformers before the split, forgetting to exponentiate log-transformed predictions, one-hot encoding high-cardinality features for tree models) MUST show the wrong approach, its observable consequence, and the corrected approach, with tests asserting the behaviour.

**Study plan, practice and revision**

- **FR-016**: The book MUST include a 14-day fast-track plan and a 21-day standard plan, each listing daily goals, chapters and tutorials, estimated time (1.5–3 hours per day), and a daily self-check.
- **FR-017**: The book MUST include at least two full-length practice exams of 48 questions each, timed at 90 minutes, with the number of questions per section proportional to that section's share of exam objectives, and answer explanations linked to chapters.
- **FR-018**: The book MUST let the learner compute a score per exam section for each practice exam and recommend chapters to revisit based on missed questions.
- **FR-019**: Practice and review questions MUST be original (not copies of official or third-party questions) and MUST follow the official style: scenario-based, single- or multiple-selection, with plausible distractors.
- **FR-020**: The book MUST include a condensed revision cheat sheet with key API calls, decision tables, formulas, and the exam-vs-current-platform terminology list, each entry linking to its chapter.
- **FR-021**: The book MUST state the exam-guide version and Databricks documentation snapshot date it targets, and advise the learner to check for a newer exam guide two weeks before the exam.

### Key Entities

- **Exam Objective**: one bullet from the official exam outline; belongs to exactly one exam section; maps to one or more chapter sections, tutorials and questions.
- **Chapter**: a unit of the book covering a coherent set of objectives; has learning objectives, body, comparisons of approaches, pitfalls, summary and review questions.
- **Tutorial**: hands-on, runnable exercise paired with a chapter; has prerequisites (compute type, features), steps, expected outputs, clean-up and tests.
- **Question**: an exam-style item with scenario, options, correct answer(s), explanation and the objective it assesses; used in chapter reviews and practice exams.
- **Practice Exam**: a set of 48 questions with a time limit and per-section scoring.
- **Study Plan Day**: a day in the 14- or 21-day plan with goals, assigned chapters/tutorials, time estimate and self-check.
- **Platform Change Note**: a record of a difference between exam terminology and the current platform (old name, new name/replacement, status, which the exam expects).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of the 48 exam objectives are mapped to at least one chapter section, one review or practice question, and — where the objective is practical — one tutorial step.
- **SC-002**: A learner with basic Python and ML knowledge can complete all chapters, tutorials and both practice exams within 21 days at no more than 3 hours per day (approx. 45–60 hours total), and within 14 days on the fast track.
- **SC-003**: Learners who complete the plan score at least 80% on each practice exam before booking the real exam (the official pass mark is not published; 80% provides a safety margin).
- **SC-004**: 100% of tutorials run to completion on the recorded target runtime, verified at least once per runtime LTS change.
- **SC-005**: 100% of local tutorial tests pass; workspace-only tests are skipped (not failed) when no workspace is configured.
- **SC-006**: Every chapter contains learning objectives, exam mapping, a Pitfalls section and at least eight review questions with explanations (checked per chapter).
- **SC-007**: Every renamed or deprecated item identified in the exam-vs-platform drift list (at least Hyperopt, Delta Live Tables, online tables, workspace model registry, AutoML packaging) has an explicit explanation in the book.
- **SC-008**: Total question bank of at least 200 original questions (chapter reviews plus practice exams), with every exam section represented in proportion to its objective count.

## Assumptions

- **Target exam version**: the 2025-03-01 exam guide is the current official version as of 2026-10-14; no newer official guide was found. If a newer guide is published, coverage is re-reviewed (Edge Cases).
- **Target platform**: Databricks on AWS documentation (`docs.databricks.com/aws/en`) and Databricks Runtime 18 LTS for Machine Learning are the reference; content applies to Azure/GCP with minor UI differences, which are out of scope.
- **Learner profile**: comfortable with Python and basic pandas; familiar with basic ML concepts but not necessarily with Spark or Databricks; can study 1.5–3 hours per day.
- **Workspace access**: the learner has (or can create) a Databricks workspace with Unity Catalog and the ability to start classic compute with the ML runtime (e.g., a trial or company workspace). Steps requiring paid serverless features (model serving, online feature stores) are clearly marked; a free/serverless-only account can complete most data-processing, MLflow and model-development tutorials, with read-through alternatives for the rest.
- **Deprecated features on the exam**: because the exam still tests Hyperopt and Delta Live Tables, the book teaches them as exam-required, labelled as deprecated/renamed, alongside the current replacement (Optuna/Ray Tune; Lakeflow Declarative Pipelines). Hyperopt may require an explicit install step on the target runtime.
- **Format and language**: the book is written in English as Markdown in the repository, with tutorials as code in the repository; publishing to other formats (PDF, website) is out of scope for this feature.
- **Question originality**: automated checks compare questions only against the official sample questions; originality against third-party material is ensured by author review.
- **Out of scope**: generative AI / LLM topics, ML Professional exam content, and Databricks Academy course material (referenced, not reproduced).
- **Constitution**: the project constitution (v1.0.0) governs structure (`book/`, `src/`, `tests/`), testing, pitfall and chapter-section requirements; this spec does not repeat those rules beyond what is needed for testability.
