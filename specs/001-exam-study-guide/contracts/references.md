# Contract: Documentation References and Continuous Docs Verification

Implements the user instruction "keep on checking the documents website to ensure everything is covered" and constitution Principle II.

## `book/references.yaml`

```yaml
- id: hyperparameter-tuning
  url: https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/
  title: Hyperparameter tuning
  claim: Hyperopt is not included in Databricks Runtime ML after 16.4 LTS ML; Optuna or Ray Tune recommended.
  objectives: [S3.05, S3.06, S3.07]
  chapters: [10]
  last_verified: 2026-10-14
  status: ok
```

- Every link to docs.databricks.com in `book/` MUST be a plain inline Markdown link (`[Hyperparameter tuning](https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/)`) whose URL exactly matches a `url` in this file. `book check` reports unregistered docs URLs as errors. Reference-style links are not used (they render as literal text on GitHub without a definition).
- Statuses: `ok`, `moved`, `discrepancy` (requires `note:` describing the difference between docs and observed behaviour).

## `book/exam/docs-coverage.md`

A table produced by the docs-coverage sweep: every sub-page under these docs index pages, with a decision.

| Index page | URL |
|---|---|
| Machine learning | `/aws/en/machine-learning/` |
| MLflow | `/aws/en/mlflow/` |
| Feature store | `/aws/en/machine-learning/feature-store/` |
| AutoML | `/aws/en/machine-learning/automl/` |
| Hyperparameter tuning | `/aws/en/machine-learning/automl-hyperparam-tuning/` |
| Model serving | `/aws/en/machine-learning/model-serving/` |
| Model inference | `/aws/en/machine-learning/model-inference/` |
| MLOps | `/aws/en/machine-learning/mlops/` |
| Lakeflow pipelines | `/aws/en/ldp/` |
| Runtime ML release notes | `/aws/en/release-notes/runtime/` |
| Notebooks / visualisations / dbutils | `/aws/en/notebooks/`, `/aws/en/dev-tools/databricks-utils` |

Columns: `page`, `url`, `decision` (`covered` → chapter link · `beyond-exam` → chapter link · `out-of-scope` → reason), `checked_on`.

## Verification cadence

| When | Action | Who/What |
|---|---|---|
| Every chapter task | Re-read cited pages; update `claim`, `last_verified`; add newly discovered relevant pages | Author task step "Verify against docs" |
| Before release / every 4 weeks | Docs-coverage sweep; re-check certification page for a newer exam guide (update `exam_guide.checked_on`) | Task in `tasks.md` |
| Weekly (CI `schedule`) | `db-ml-associate book check-links` | GitHub Actions job, marker `network` |
| New Runtime ML LTS | Update `runtime.toml`, re-run workspace tests, re-verify all references | Constitution governance |
