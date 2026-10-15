---
chapter: 1
title: The Databricks ML platform
objectives: [S1.01, S1.02, S1.16]
prerequisites: []
reading_minutes: 70
tutorial:
  notebook: notebooks/ch01_platform.py
  minutes: 20
  requires: [unity_catalog]
  free_edition: full
  cleanup: false
  extra_files: []
  steps:
    - id: runtime-info
      objectives: [S1.02]
    - id: unity-catalog-namespace
      objectives: [S1.01]
    - id: environments-as-catalogs
      objectives: [S1.01, S1.16]
references:
  [databricks-runtime-ml, dbr-18-lts-ml, train-models, serverless-dependencies, unity-catalog, mlops-workflow, mlops-deployment-patterns]
---

# 1. The Databricks ML platform

> **Exam mapping**: Section 1 — Databricks Machine Learning · objectives S1.01, S1.02, S1.16 ·
> **Runtime**: Databricks Runtime 18 LTS ML · **No prerequisites** — read this right after the
> introduction.

This chapter is the map for everything else in Section 1: the compute you train on (the ML
runtime), where data and models live (Unity Catalog), and how work moves from a notebook to
production (MLOps). The exam asks about the *advantages* of ML runtimes, the *best practices* of an
MLOps strategy, and when to promote *code* versus *models*.

> **Tutorial:** [`notebooks/ch01_platform.py`](../notebooks/ch01_platform.py) · about 20 minutes · **Requires:** Unity Catalog · **Free Edition:** yes · **Clean-up:** none. Run [`notebooks/00_setup.py`](../notebooks/00_setup.py) once first.

## Learning objectives

- List the advantages of Databricks Runtime for Machine Learning over the standard runtime and
  know how to create compute with it.
- Explain Unity Catalog's three-level namespace and why governance matters for ML assets.
- Describe the best practices of an MLOps strategy on Databricks.
- Decide when to promote code and when to promote models between environments.

## The Databricks ML toolbox

<!-- objectives: S1.02 -->

| Need | Databricks feature | Chapter |
|---|---|---|
| Compute with ML libraries | Databricks Runtime for Machine Learning | 1 |
| Governed data, features and models | Unity Catalog | 1, 3, 5 |
| A quick baseline model | AutoML | 2 |
| Reusable features for training and serving | Feature engineering in Unity Catalog | 3 |
| Experiment tracking | MLflow (managed) | 4 |
| Model versions, aliases, tags | Models in Unity Catalog | 5 |
| Batch and streaming inference | Spark UDFs, Lakeflow pipelines, AI Functions | 12 |
| Real-time inference | Mosaic AI Model Serving | 13 |

## Databricks Runtime for Machine Learning

<!-- objectives: S1.02 -->

A **Databricks Runtime** is the software image on a compute resource: operating system, Spark,
Python and libraries. The **ML variant** adds a curated machine-learning stack
([Databricks Runtime for Machine Learning](https://docs.databricks.com/aws/en/machine-learning/databricks-runtime-ml)).

**Advantages of the ML runtime:**

- **Preinstalled, tested libraries** — scikit-learn, XGBoost, LightGBM, PyTorch, TensorFlow/Keras,
  Optuna, MLflow, the feature engineering client and the Databricks SDK are installed and tested to
  work together, so there is no dependency wrangling. Runtime 18 LTS ML ships, among others,
  mlflow-skinny 3.8.1, scikit-learn 1.6.1, xgboost 3.1.1, lightgbm 4.6.0, optuna 3.6.1 and
  databricks-feature-engineering 0.13.0.1
  ([release notes](https://docs.databricks.com/aws/en/release-notes/runtime/18ml)).
- **Optimised and regularly updated** — "top-tier" libraries get extra testing and optimisation and
  are updated with each runtime release.
- **GPU support** — choose a GPU instance type and the runtime includes the GPU libraries (CUDA,
  GPU builds of PyTorch).
- **Reproducibility** — every cluster on the same runtime version has identical library versions;
  MLflow records the runtime used to train a model.
- **Integrated tooling** — AutoML (installable, chapter 2), MLflow tracking and Unity Catalog
  integration work out of the box.

**Creating ML compute:** in the create-compute UI tick **Machine learning**. This sets the access
mode to **Dedicated** (assigned to one user or group), which is required for Unity Catalog data
with the ML runtime. For deep-learning training on GPUs, Databricks now recommends **AI Runtime**, a
serverless GPU offering. **Serverless** notebooks have no runtime to pick; instead you choose a
**base environment** in the notebook's Environment side pane. The **ML** base environment has the
Python packages of Databricks Runtime ML preinstalled, and you add further dependencies in the same
pane ([Manage serverless dependencies](https://docs.databricks.com/aws/en/compute/serverless/dependencies)).

| | Standard runtime | ML runtime |
|---|---|---|
| Spark, Delta, Python | ✅ | ✅ |
| ML / DL libraries preinstalled | Few | Many, tested together |
| GPU builds | No | Yes (GPU instance types) |
| Typical use | ETL, SQL, BI | Training and tuning models |

The tutorials target **Databricks Runtime 18 LTS ML** (Python 3.12.3, Spark 4.1.0); the exact versions
are recorded in [`runtime.toml`](../runtime.toml):

<!-- snippet: src/db_ml_associate/ch01_platform/runtime_info.py#runtime_info -->
```python
target = tomllib.loads(runtime_file.read_text(encoding="utf-8"))
runtime_version = os.environ.get("DATABRICKS_RUNTIME_VERSION")  # e.g. "18.0" on Databricks
libraries = {str(name): str(version) for name, version in target["libraries"].items()}
```

> **Exam tip:** "Why use the ML runtime?" → preinstalled, compatible ML libraries (no manual
> installs, no version conflicts), GPU support and built-in integration with MLflow/AutoML. "LTS"
> means long-term support: choose LTS versions for production.

## Unity Catalog in one page

<!-- objectives: S1.02, S1.01 -->

**Unity Catalog** is the governance layer for data *and* AI assets
([What is Unity Catalog?](https://docs.databricks.com/aws/en/data-governance/unity-catalog/)):

- A three-level namespace: **`catalog.schema.object`**, e.g. `ml_associate.tutorial.customer_churn`.
- Securable objects include **tables, views, volumes, functions and models**.
- Access is granted with **privileges** (`GRANT SELECT ON TABLE …`), row/column filters and
  workspace bindings.
- **Lineage** links source tables to features, models and endpoints; **audit logs** record access;
  **Catalog Explorer** lets users discover assets.

Because feature tables (chapter 3) and registered models (chapter 5) live in Unity Catalog, they
share this governance across every workspace attached to the same metastore.

## MLOps strategy best practices

<!-- objectives: S1.01 -->

**MLOps** applies software-engineering discipline (version control, testing, automation,
monitoring) to ML code, data and models. Databricks' recommended workflow
([MLOps workflows on Databricks](https://docs.databricks.com/aws/en/machine-learning/mlops/mlops-workflow)):

| Environment | Who | What happens | Unity Catalog catalog |
|---|---|---|---|
| **Development** | Data scientists | Explore data, build training/validation/deployment pipelines, read-only access to production data where possible | `dev` |
| **Staging** | CI system | Unit tests on pull requests; integration tests run all pipelines on an environment that mirrors production | `staging` |
| **Production** | ML engineers (automated jobs) | Train on production data, validate, deploy, serve, monitor, retrain | `prod` |

Production pipeline steps: **train** → **validate** (performance on slices, compliance; passing
models get the **`Challenger`** alias) → **deploy** (compare Challenger with the current
**`Champion`**; the better one becomes Champion) → **serve** (Model Serving, or batch/streaming jobs
that load `@Champion`) → **monitor** (drift, performance, alerts) → **retrain** (start scheduled, then
trigger on drift).

**Best practices:**

1. **Separate environments** (dev / staging / prod) — typically one Unity Catalog catalog each, with
   clear promotion steps.
2. **Everything in Git** — pipelines and code are version-controlled; CI runs tests on every pull
   request.
3. **Deploy code, not models** (default) — see the next section.
4. **Unified governance** — Delta tables for data and features, MLflow for experiments, Unity
   Catalog for models and access control.
5. **Least privilege in production** — data scientists get read-only visibility, not write or
   compute access.
6. **Automate, with human checkpoints where needed** — approvals via CI/CD or notifications.
7. **Aliases decouple deployment from inference** — jobs load `models:/catalog.schema.model@Champion`,
   so promoting a new version needs no code change (chapter 5).
8. **Monitor and retrain** — track data drift and model quality in production.

> **Exam tip:** MLOps best-practice answers mention separate environments, version control,
> automated testing (CI/CD), model registry with governance, monitoring and reproducibility.
> "Train in a notebook and copy the model file to production by hand" is never the answer.

## Promoting code vs promoting models

<!-- objectives: S1.16 -->

How does a model reach production? Two patterns
([Model deployment patterns](https://docs.databricks.com/aws/en/machine-learning/mlops/deployment-patterns)):

**Deploy code (the default recommendation).** The *training code* moves dev → staging → prod
through Git and CI/CD; the model is **retrained in each environment**, finally on production data.

**Deploy models.** The *model artifact* is trained once (in development) and the same artifact is
promoted through staging to production.

| | Deploy code | Deploy models |
|---|---|---|
| What moves between environments | Training code (Git) | Model artifact (registry) |
| Where the production model is trained | In production, on production data | In development |
| Prefer when | Production data is restricted (dev cannot read it); automated retraining is needed; code must pass review and tests | Training is very expensive or not reproducible (e.g. large deep-learning models); all work happens in one workspace; no external CI/CD |
| Advantages | Production models come from reviewed, tested code; retraining is easy to automate | Train once; simpler handoff for data scientists |
| Trade-offs | Training cost in several environments; data scientists must hand code to engineers | Dev must access production-like data; ancillary code (features, inference, monitoring) still has to be deployed separately; retraining harder |

With Unity Catalog, an environment usually corresponds to a **catalog**, and "deploy models" means
copying a model version from, say, `staging.ml.churn` to `prod.ml.churn`.

> **Exam tip:** "production data cannot be accessed from development" or "the model must be
> retrained regularly on fresh data" → **promote code**. "Training takes days on GPUs and the exact
> artifact must be reused" → **promote models**.

## Choosing an approach

| Situation | Choose |
|---|---|
| Training and tuning classic ML models on clusters | ML runtime (Dedicated access mode) |
| ML in serverless notebooks (e.g. Free Edition) | Serverless with the **ML** base environment |
| ETL, SQL, dashboards | Standard runtime or serverless |
| GPU deep-learning training | AI Runtime (serverless GPU) or GPU ML runtime |
| Isolating dev/staging/prod data and models | One Unity Catalog catalog per environment |
| Moving work to production by default | Deploy code |
| Very expensive, hard-to-reproduce training | Deploy models |

## Pitfalls

1. **Installing ML libraries by hand on a standard runtime** — version conflicts and unreproducible
   environments; use the ML runtime (or pin versions in a serverless environment).
2. **Choosing a non-LTS runtime for production** — it reaches end of support sooner.
3. **One catalog for everything** — dev experiments can overwrite production tables and models.
4. **Promoting models when production data is off-limits to developers** — the model never sees
   production data before release; deploy code instead.
5. **Hard-coding model version numbers in inference jobs** — use aliases (`@Champion`) so promotion
   needs no code change.

## Summary

- The ML runtime gives preinstalled, tested ML libraries, GPU support and MLflow/AutoML integration;
  create it with the *Machine learning* checkbox (Dedicated access mode).
- Unity Catalog governs data, features and models with a `catalog.schema.object` namespace,
  privileges, lineage and audit logs.
- MLOps best practices: separate environments, Git + CI/CD, governance, least privilege, automation,
  monitoring, aliases.
- Deploy code by default; deploy models when training is expensive or everything happens in one
  workspace.

## Review questions

<!-- questions: ch01 -->

### Question 1

A team spends hours on every new cluster installing scikit-learn, XGBoost and MLflow and fixing version conflicts between them. Which change addresses this most directly?

- **A.** Use Databricks Runtime for Machine Learning, which preinstalls a tested set of ML libraries
- **B.** Use a larger driver node
- **C.** Store the libraries in a Delta table
- **D.** Switch the cluster to a non-LTS standard runtime

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — The ML runtime ships popular ML libraries preinstalled and tested together, removing manual installs and version conflicts.

Why the other options are wrong:

- **B**: Node size does not install libraries.
- **C**: Delta tables store data, not Python packages.
- **D**: A standard runtime still lacks the ML libraries, and non-LTS versions are less stable.

Objective S1.02 · [Revisit](01-databricks-ml-platform.md#databricks-runtime-for-machine-learning)

</details>

### Question 2

Which TWO are advantages of Databricks Runtime for Machine Learning over the standard Databricks Runtime?

- **A.** GPU-enabled builds of deep-learning libraries when a GPU instance type is selected
- **B.** Popular ML libraries such as scikit-learn, XGBoost and PyTorch are preinstalled and tested together
- **C.** Models are deployed to production automatically
- **D.** Spark is no longer needed
- **E.** Unity Catalog permissions are bypassed for faster training

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — The ML runtime adds a curated, compatible ML/DL library stack, including GPU builds.

Why the other options are wrong:

- **C**: Deployment is a separate step (Model Serving, jobs).
- **D**: The ML runtime still includes Spark.
- **E**: Governance is never bypassed; the ML runtime uses Dedicated access mode for Unity Catalog data.

Objective S1.02 · [Revisit](01-databricks-ml-platform.md#databricks-runtime-for-machine-learning)

</details>

### Question 3

A data scientist ticks the "Machine learning" checkbox when creating classic compute that must read Unity Catalog tables. Which access mode does the compute use?

- **A.** Dedicated (assigned to a single user or group)
- **B.** No isolation shared
- **C.** Standard with legacy table ACLs
- **D.** Serverless

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Selecting the ML runtime sets the access mode to Dedicated, which is required to use Unity Catalog data with Databricks Runtime ML.

Why the other options are wrong:

- **B**: No isolation shared does not support Unity Catalog.
- **C**: Legacy table ACLs are not the ML runtime's access mode.
- **D**: Serverless is a different compute type without runtime selection.

Objective S1.02 · [Revisit](01-databricks-ml-platform.md#databricks-runtime-for-machine-learning)

</details>

### Question 4

A company is designing its MLOps strategy on Databricks. Which TWO practices are recommended?

- **A.** Separate development, staging and production environments, for example one Unity Catalog catalog each
- **B.** Keep pipeline code in Git and run automated tests in CI before promotion
- **C.** Give every data scientist write access to production tables to speed up fixes
- **D.** Retrain production models manually from a personal notebook
- **E.** Store model files on a laptop and upload them to production when ready

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — Isolated environments with clear promotion steps and version-controlled, tested code are core MLOps best practices.

Why the other options are wrong:

- **C**: Production follows least privilege; data scientists typically have read-only access.
- **D**: Retraining should be an automated, reviewed pipeline.
- **E**: Models belong in a governed registry, not on laptops.

Objective S1.01 · [Revisit](01-databricks-ml-platform.md#mlops-strategy-best-practices)

</details>

### Question 5

A nightly batch-scoring job must always use the currently approved production model, and the team does not want to edit the job each time a new model version is promoted. What should the job load?

- **A.** models:/prod.ml.churn/7, updated by hand after each release
- **B.** models:/prod.ml.churn@Champion
- **C.** The latest run in the data scientist's experiment
- **D.** A pickle file copied to DBFS

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — Aliases decouple deployment from inference; moving the Champion alias to a new version makes the next job run use it without code changes.

Why the other options are wrong:

- **A**: Version numbers must be edited for every promotion.
- **C**: The latest experiment run is not necessarily validated or approved.
- **D**: Ungoverned copies bypass the registry, lineage and access control.

Objective S1.01 · [Revisit](01-databricks-ml-platform.md#mlops-strategy-best-practices)

</details>

### Question 6

Security rules prevent the development environment from reading production customer data, and the churn model must be retrained monthly on fresh production data. Which deployment pattern fits?

- **A.** Deploy models — train once in development and copy the artifact to production
- **B.** Deploy code — promote the reviewed training code and train the model in production
- **C.** Deploy notebooks by exporting HTML files
- **D.** No deployment pattern is needed; serve the development model directly

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: B** — When development cannot access production data and regular retraining is needed, promoting code lets the production environment train on production data with reviewed code.

Why the other options are wrong:

- **A**: A model trained in development would never see production data.
- **C**: HTML exports are not executable deployment artifacts.
- **D**: Serving a development model skips validation and governance.

Objective S1.16 · [Revisit](01-databricks-ml-platform.md#promoting-code-vs-promoting-models)

</details>

### Question 7

A team fine-tunes a large model on GPUs for four days; reproducing the exact training run is impractical. All work happens in a single workspace and there is no external CI/CD. Which pattern is most appropriate?

- **A.** Deploy models — promote the trained artifact through staging to production
- **B.** Deploy code — retrain the model separately in staging and production
- **C.** Retrain the model in every environment every day
- **D.** Keep the model only in the development catalog

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Deploy models suits expensive or hard-to-reproduce training and single-workspace setups: train once and promote the same artifact.

Why the other options are wrong:

- **B**: Retraining a four-day job per environment is costly and may not reproduce the same model.
- **C**: Daily retraining multiplies the cost problem.
- **D**: Production use requires promotion and governance.

Objective S1.16 · [Revisit](01-databricks-ml-platform.md#promoting-code-vs-promoting-models)

</details>

### Question 8

Which statement is a recognised trade-off of the "deploy models" pattern compared with "deploy code"?

- **A.** Ancillary code such as feature and inference pipelines must still be deployed separately, and automated retraining is harder
- **B.** Models can never be registered in Unity Catalog
- **C.** It always costs more compute than deploy code
- **D.** It requires production data to be copied into the development catalog every day

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Only the model artifact moves; supporting pipelines still need their own deployment, and retraining in production is not part of the flow.

Why the other options are wrong:

- **B**: Deploy models typically uses the Unity Catalog registry to promote versions.
- **C**: Deploy models trains once and usually costs less training compute.
- **D**: Data copying is not inherent to the pattern, although development needs access to suitable data.

Objective S1.16 · [Revisit](01-databricks-ml-platform.md#promoting-code-vs-promoting-models)

</details>

### Question 9

A data scientist works in serverless notebooks and wants the same Python ML packages that Databricks Runtime ML provides, without managing a cluster. What should they do?

- **A.** Select the ML base environment in the notebook's Environment side pane
- **B.** Install PySpark with %pip to get MLlib
- **C.** Attach the notebook to a SQL warehouse
- **D.** It is impossible; ML packages require classic compute

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Serverless notebooks offer an ML base environment with the Python packages of Databricks Runtime ML preinstalled.

Why the other options are wrong:

- **B**: Databricks warns not to install PySpark on serverless notebooks.
- **C**: SQL warehouses run SQL, not Python notebooks with ML libraries.
- **D**: The ML base environment exists precisely for this case.

Objective S1.02 · [Revisit](01-databricks-ml-platform.md#databricks-runtime-for-machine-learning)

</details>
<!-- /questions -->
