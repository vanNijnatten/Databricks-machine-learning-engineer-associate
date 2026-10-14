---
chapter: 5
title: Model lifecycle in Unity Catalog
objectives: [S1.14, S1.15, S1.17, S1.18]
prerequisites: [1, 4]
reading_minutes: 70
references: [model-lifecycle-uc, mlflow-3, mlops-workflow, unity-catalog, workspace-feature-store]
---

# 5. Model lifecycle in Unity Catalog

> **Exam mapping**: Section 1 — Databricks Machine Learning · objectives S1.14, S1.15, S1.17,
> S1.18 · **Runtime**: Databricks Runtime 18 LTS ML (MLflow 3.8) · **Read after**: chapters 1 and 4.

A logged model is just an artifact of a run. To deploy it you **register** it: it becomes a version
of a named, governed **registered model** in Unity Catalog. Versions are labelled with **tags** and
promoted with **aliases** such as `champion` and `challenger`. Jobs and endpoints refer to the
alias, so promotion is a single API call.

## Learning objectives

- Register a model in the Unity Catalog registry with the MLflow API.
- Explain the benefits of the Unity Catalog registry over the legacy workspace registry.
- Set and remove tags on registered models and model versions.
- Promote a challenger to champion with aliases, and load models by alias.

## Registering a model in Unity Catalog

<!-- objectives: S1.14 -->

A registered model in Unity Catalog has a **three-level name** `catalog.schema.model`, for example
`ml_associate.tutorial.churn_model`. The catalog usually encodes the environment (`dev`, `staging`,
`prod`), not the deployment status. Each registration creates a new, immutable **version** (1, 2,
3, …) ([Manage model lifecycle in Unity Catalog](https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/)).

<!-- snippet: src/db_ml_associate/ch05_registry/registry.py#register_model -->
```python
# mlflow.set_registry_uri("databricks-uc")  # default in MLflow 3 on Databricks
version = mlflow.register_model(model_uri=model_uri, name=name)  # name = catalog.schema.model
# alternative: mlflow.sklearn.log_model(model, name="model", registered_model_name=name, ...)
```

Three ways to register:

| Way | When |
|---|---|
| `mlflow.register_model("runs:/<run_id>/model", name)` or `mlflow.register_model("models:/<model_id>", name)` | Register an already-logged model, e.g. the best run found with `search_runs` (chapter 4) |
| `log_model(..., registered_model_name=name)` | Log and register in one step |
| Catalog Explorer / the run's **Register model** button | UI |

Requirements:

- **A signature** — "new ML model versions in UC must have a model signature". Log one with
  `signature=infer_signature(...)` or an `input_example` (chapter 4); autologging adds it.
- **Permissions** — `USE CATALOG` and `USE SCHEMA`, plus `CREATE MODEL` on the schema for a new
  model; to add versions, own the model or hold `CREATE MODEL VERSION`; to load, `EXECUTE` on the
  model.
- **Registry URI** — MLflow 3 defaults to `databricks-uc` on Databricks
  ([MLflow 3](https://docs.databricks.com/aws/en/mlflow/mlflow-3-install)); with MLflow 2 call
  `mlflow.set_registry_uri("databricks-uc")`.

> **Exam tip:** "register using the MLflow Client API in the Unity Catalog registry" →
> `mlflow.register_model(model_uri, "catalog.schema.model")` (or `MlflowClient().create_model_version`)
> with the registry URI set to `databricks-uc`.

## Unity Catalog registry vs workspace registry

<!-- objectives: S1.15 -->

| | Models in Unity Catalog | Workspace Model Registry (legacy) |
|---|---|---|
| Scope | Account-level metastore: discoverable and usable from **every workspace** attached to it | One workspace |
| Access control | Unity Catalog privileges (`EXECUTE`, `CREATE MODEL VERSION`, …), centrally managed | Workspace-local permissions |
| Lineage | From source tables and feature tables to models and endpoints | — |
| Auditing | Unity Catalog audit logs | Limited |
| Promotion | **Aliases** (`@champion`) and environment catalogs | **Stages** (`Staging`, `Production`, `Archived`) |
| Names | `catalog.schema.model` | Flat name |
| Copy between environments | `copy_model_version` | Manual |

The benefits Databricks lists for Unity Catalog: **centralised access control, auditing, lineage and
model discovery across workspaces**, while staying compatible with the open-source MLflow client.

> **Platform change:** stages are **not supported** in Unity Catalog. Use catalogs for
> environments and aliases for deployment status. Workspaces without Unity Catalog fall back to the
> legacy workspace registry.

## Tags

<!-- objectives: S1.17 -->

**Tags** are key–value pairs for searching and organising — ownership, validation status, data
version. Registered models and individual versions have separate tags:

<!-- snippet: src/db_ml_associate/ch05_registry/registry.py#model_tags -->
```python
client.set_registered_model_tag(name, key, value)  # e.g. ("team", "retention")
```

<!-- snippet: src/db_ml_associate/ch05_registry/registry.py#remove_model_tag -->
```python
client.delete_registered_model_tag(name, key)
```

<!-- snippet: src/db_ml_associate/ch05_registry/registry.py#version_tags -->
```python
client.set_model_version_tag(name, version, key, value)  # e.g. ("validation_status", "approved")
```

<!-- snippet: src/db_ml_associate/ch05_registry/registry.py#remove_version_tag -->
```python
client.delete_model_version_tag(name, version, key)
```

| Target | Set | Remove |
|---|---|---|
| Registered model | `set_registered_model_tag(name, key, value)` | `delete_registered_model_tag(name, key)` |
| Model version | `set_model_version_tag(name, version, key, value)` | `delete_model_version_tag(name, version, key)` |

Tags can also be edited in Catalog Explorer. Unlike aliases, a tag does not have to be unique: many
versions can carry `validation_status=approved`.

## Aliases: champion and challenger

<!-- objectives: S1.18 -->

An **alias** is a mutable, named pointer to **exactly one** version of a registered model. Moving an
alias is instant, and everything that loads `models:/<name>@<alias>` follows it:

<!-- snippet: src/db_ml_associate/ch05_registry/registry.py#set_alias -->
```python
client.set_registered_model_alias(name, alias, version)  # moves the alias if it already exists
```

<!-- snippet: src/db_ml_associate/ch05_registry/registry.py#load_by_alias -->
```python
model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")  # e.g. models:/prod.ml.churn@champion
```

The **champion/challenger** pattern from Databricks' MLOps workflow
([MLOps workflows](https://docs.databricks.com/aws/en/machine-learning/mlops/mlops-workflow)):

1. The production model has the alias **`champion`**; batch jobs and endpoints load `@champion`.
2. A newly trained version passes validation and receives **`challenger`**.
3. The challenger is compared with the champion — offline on held-out data, or online with a traffic
   split (chapter 13).
4. If it is better, **`champion` moves to the challenger's version** and the `challenger` alias is
   removed. The next job run uses the new model automatically.

<!-- snippet: src/db_ml_associate/ch05_registry/registry.py#promote_challenger -->
```python
champion = client.get_model_version_by_alias(name, "champion")
challenger = client.get_model_version_by_alias(name, "challenger")
champion_score = _metric(client, champion, metric)
challenger_score = _metric(client, challenger, metric)
better = challenger_score > champion_score if higher_is_better else challenger_score < champion_score
if better:
    client.set_registered_model_alias(name, "champion", str(challenger.version))  # instant switch
    client.delete_registered_model_alias(name, "challenger")
```

Useful client calls: `get_model_version_by_alias(name, alias)` returns the version an alias points
to; `delete_registered_model_alias(name, alias)` removes an alias. For the *deploy models* pattern
(chapter 1), `copy_model_version` copies a version to another catalog:

<!-- snippet: src/db_ml_associate/ch05_registry/registry.py#copy_model_version -->
```python
return client.copy_model_version(src_model_uri=f"models:/{source_name}/{version}", dst_name=target_name)
```

> **Exam tip:** "promote a challenger to champion" → `set_registered_model_alias(name, "champion",
> challenger_version)`. Not `transition_model_version_stage` (stages), not re-registering, not
> editing jobs.

## Choosing an approach

| Need | Use |
|---|---|
| Mark which version production uses | Alias (`champion`) |
| Record approvals, owners, data versions | Tags |
| Separate dev/staging/prod | Catalog in the three-level name |
| Promote between catalogs without retraining | `copy_model_version` (deploy models) |
| Retrain in each environment | Register the newly trained version there (deploy code) |
| Pin an exact version for reproducible analysis | `models:/<name>/<version>` |

## Pitfalls

**1. Hard-coding version numbers in production jobs.** A job that loads `models:/…/1` keeps using
version 1 after the champion moves to version 2:

> **⚠️ Wrong approach:** `mlflow.pyfunc.load_model("models:/ml_associate.tutorial.churn_model/1")`
> in a scheduled job.

The chapter's test registers two versions, points `champion` at version 1, writes the URI with
version 1 into a "job", promotes version 2, and shows that the job still loads version 1 while
`models:/…@champion` loads version 2.

**2. Using stages.** `transition_model_version_stage(..., "Production")` is the legacy workspace
registry; stages do not exist in Unity Catalog.

**3. Two-level names.** `tutorial.churn_model` registers nowhere in Unity Catalog; use
`catalog.schema.model`.

**4. Missing signature.** Registration fails; log the model with a signature (chapter 4).

**5. Missing `EXECUTE` privilege.** Jobs that load a model by alias need `EXECUTE` on it, plus
`USE CATALOG` and `USE SCHEMA`.

## Summary

- Register with `mlflow.register_model(uri, "catalog.schema.model")` or `registered_model_name`; every
  version needs a signature.
- Unity Catalog adds cross-workspace discovery, central access control, lineage and auditing; stages
  are replaced by aliases and environment catalogs.
- Tags: `set_/delete_registered_model_tag` and `set_/delete_model_version_tag`.
- Promote with aliases: move `champion` to the challenger's version; consumers load `@champion`.

## Review questions

<!-- questions: ch05 -->

### Question 1

A data scientist found the best run with run_id best_id; its model was logged with name="model". Which call registers it as a new version of a Unity Catalog model?

- **A.** mlflow.register_model(f"runs:/{best_id}/model", "prod.churn.churn_model")
- **B.** mlflow.register_model(f"runs:/{best_id}/model", "churn_model")
- **C.** mlflow.log_param("registered", True)
- **D.** MlflowClient().transition_model_version_stage("churn_model", 1, "Production")

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — register_model takes the logged model's URI and a three-level Unity Catalog name and creates a new model version.

Why the other options are wrong:

- **B**: A one-level name is not a Unity Catalog model name.
- **C**: Logging a parameter registers nothing.
- **D**: Stages belong to the legacy workspace registry.

Objective S1.14 · [Revisit](05-model-lifecycle-uc.md#registering-a-model-in-unity-catalog)

</details>

### Question 2

Registering a scikit-learn model in Unity Catalog fails with an error about a missing signature. What is the most direct fix?

- **A.** Re-log the model with signature=infer_signature(X_sample, model.predict(X_sample)) or an input_example, then register it
- **B.** Register it under a two-level name
- **C.** Grant the user EXECUTE on the catalog
- **D.** Register the model in the workspace registry with a stage

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — New model versions in Unity Catalog must have a signature; logging one fixes the error.

Why the other options are wrong:

- **B**: Two-level names are not valid in Unity Catalog and do not add a signature.
- **C**: Permissions are not the cause of a signature error.
- **D**: The legacy registry avoids, rather than fixes, the problem and loses UC governance.

Objective S1.14 · [Revisit](05-model-lifecycle-uc.md#registering-a-model-in-unity-catalog)

</details>

### Question 3

Which TWO are benefits of registering models in Unity Catalog instead of the workspace model registry?

- **A.** Models are discoverable and usable across all workspaces attached to the same metastore
- **B.** Centralised access control, auditing and lineage
- **C.** Support for the Staging and Production stages
- **D.** Models no longer need versions
- **E.** Models are automatically deployed to a serving endpoint

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — Unity Catalog brings centralised governance, auditing, lineage and cross-workspace discovery to models.

Why the other options are wrong:

- **C**: Stages are not supported in Unity Catalog; aliases replace them.
- **D**: Every registration still creates a version.
- **E**: Serving endpoints are created separately.

Objective S1.15 · [Revisit](05-model-lifecycle-uc.md#unity-catalog-registry-vs-workspace-registry)

</details>

### Question 4

A team migrating from the workspace registry asks how to express "this version is in production" in Unity Catalog. What is the recommended approach?

- **A.** Use a catalog per environment and an alias such as champion for the deployed version
- **B.** Use transition_model_version_stage(..., "Production")
- **C.** Add "_prod" to the model version number
- **D.** Delete all other versions

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Stages are not supported in Unity Catalog; environments are catalogs and deployment status is an alias.

Why the other options are wrong:

- **B**: Stages are a workspace-registry feature.
- **C**: Version numbers are assigned automatically and cannot be renamed.
- **D**: Deleting versions destroys history and rollback options.

Objective S1.15 · [Revisit](05-model-lifecycle-uc.md#unity-catalog-registry-vs-workspace-registry)

</details>

### Question 5

After validation, a data scientist wants to mark version 4 of prod.churn.churn_model with validation_status=approved without affecting other versions. Which call does this?

- **A.** MlflowClient().set_model_version_tag("prod.churn.churn_model", "4", "validation_status", "approved")
- **B.** MlflowClient().set_registered_model_tag("prod.churn.churn_model", "validation_status", "approved")
- **C.** mlflow.set_tag("validation_status", "approved")
- **D.** MlflowClient().set_registered_model_alias("prod.churn.churn_model", "approved", "4")

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — set_model_version_tag tags one specific version.

Why the other options are wrong:

- **B**: A registered-model tag applies to the model as a whole, not one version.
- **C**: mlflow.set_tag tags the active run, not the model version.
- **D**: An alias is a pointer to one version, not a status label for any number of versions.

Objective S1.17 · [Revisit](05-model-lifecycle-uc.md#tags)

</details>

### Question 6

A registered model carries the tag owner=team-a, but team B now owns it. The data scientist wants to remove the old tag before setting a new one. Which call removes the tag?

- **A.** MlflowClient().delete_registered_model_tag("prod.churn.churn_model", "owner")
- **B.** MlflowClient().delete_model_version("prod.churn.churn_model", "1")
- **C.** MlflowClient().delete_registered_model_alias("prod.churn.churn_model", "owner")
- **D.** mlflow.delete_tag("owner")

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — delete_registered_model_tag removes a tag from the registered model.

Why the other options are wrong:

- **B**: This deletes a model version, not a tag.
- **C**: This removes an alias, and owner is a tag.
- **D**: mlflow.delete_tag removes a tag from the active run.

Objective S1.17 · [Revisit](05-model-lifecycle-uc.md#tags)

</details>

### Question 7

Version 7 of prod.churn.churn_model has the alias challenger and beat the current champion (version 5) in an offline comparison. Which call promotes it so that batch jobs loading models:/prod.churn.churn_model@champion use version 7 on their next run?

- **A.** MlflowClient().set_registered_model_alias("prod.churn.churn_model", "champion", "7")
- **B.** MlflowClient().transition_model_version_stage("prod.churn.churn_model", "7", "Production")
- **C.** Register version 7 again under the name champion
- **D.** Edit every batch job to load version 7

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Moving the champion alias to version 7 switches every consumer that loads by alias, with no code change.

Why the other options are wrong:

- **B**: Stages are not supported in Unity Catalog.
- **C**: Re-registering creates a new version under a different model.
- **D**: Editing jobs is exactly what aliases avoid.

Objective S1.18 · [Revisit](05-model-lifecycle-uc.md#aliases-champion-and-challenger)

</details>

### Question 8

Which TWO statements about model aliases in Unity Catalog are correct?

- **A.** An alias points to exactly one version of a registered model at a time
- **B.** Loading models:/<name>@<alias> returns whichever version the alias currently points to
- **C.** An alias can point to versions of two different registered models
- **D.** Aliases are immutable once created
- **E.** A version can only be loaded by alias, never by number

*Select 2 answers.*

<details>
<summary>Answer</summary>

**Correct: A, B** — Aliases are mutable, unique pointers within a registered model, and consumers that load by alias follow it.

Why the other options are wrong:

- **C**: An alias belongs to a single registered model.
- **D**: Aliases are mutable; moving them is how promotion works.
- **E**: models:/<name>/<version> loads a specific version.

Objective S1.18 · [Revisit](05-model-lifecycle-uc.md#aliases-champion-and-challenger)

</details>

### Question 9

A job's service principal can see the catalog and schema but fails to load models:/prod.churn.churn_model@champion with a permission error. Which privilege is most likely missing on the registered model?

- **A.** EXECUTE
- **B.** MODIFY
- **C.** CREATE MODEL
- **D.** SELECT

*Select one answer.*

<details>
<summary>Answer</summary>

**Correct: A** — Loading a model version by alias or number requires EXECUTE on the registered model, plus USE CATALOG and USE SCHEMA.

Why the other options are wrong:

- **B**: MODIFY applies to tables, not to loading models.
- **C**: CREATE MODEL is needed to create models, not load them.
- **D**: SELECT applies to tables and views.

Objective S1.14 · [Revisit](05-model-lifecycle-uc.md#registering-a-model-in-unity-catalog)

</details>
<!-- /questions -->
