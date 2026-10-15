# Docs coverage sweep

Every sub-page linked from the documentation index pages in `specs/001-exam-study-guide/contracts/references.md`, checked on **2026-10-15** (paths relative to `https://docs.databricks.com/aws/en`). Decisions: **covered** — the book teaches the topic (chapter link); **beyond-exam** — mentioned for context but not tested; **out-of-scope** — not needed for the exam, with the reason. Cited pages are registered in [`references.yaml`](../references.yaml).

The MLOps index moved from `/machine-learning/mlops/` (now 404) to `/machine-learning/mlops/mlops-workflow`; the contract's list is updated.

Totals over all index pages (a page listed under two indexes counts twice): beyond-exam 18, covered 66, out-of-scope 109.

## Machine learning

Index: `/machine-learning/`

| Page | Decision | Where / reason | Checked on |
|---|---|---|---|
| Develop agents on Databricks (`/agents`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| AI Gateway (`/ai-gateway/overview-serving-endpoints`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Unity Catalog (`/data-governance/unity-catalog`) | covered | [ch. 1](../01-databricks-ml-platform.md), [ch. 3](../03-feature-engineering-uc.md), [ch. 5](../05-model-lifecycle-uc.md) | 2026-10-15 |
| Anomaly detection (`/data-governance/unity-catalog/data-quality-monitoring/anomaly-detection`) | out-of-scope | Data quality monitoring: not part of the exam guide | 2026-10-15 |
| Data profiling (`/data-governance/unity-catalog/data-quality-monitoring/data-profiling`) | beyond-exam | [ch. 1](../01-databricks-ml-platform.md) | 2026-10-15 |
| Declarative Automation Bundles (`/dev-tools/bundles`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Get started: Build your first machine learning model on Databricks (`/getting-started/ml-get-started`) | covered | [ch. 1](../01-databricks-ml-platform.md), [ch. 4](../04-mlflow-tracking.md) | 2026-10-15 |
| Lakeflow Jobs (`/jobs`) | beyond-exam | [ch. 1](../01-databricks-ml-platform.md) | 2026-10-15 |
| AI Runtime (`/machine-learning/ai-runtime`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Concepts: Data science and machine learning on Databricks (`/machine-learning/concepts`) | covered | [ch. 1](../01-databricks-ml-platform.md) | 2026-10-15 |
| Databricks Runtime for ML (`/machine-learning/databricks-runtime-ml`) | covered | [ch. 1](../01-databricks-ml-platform.md) | 2026-10-15 |
| Feature Store (`/machine-learning/feature-store`) | covered | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| Foundation model APIs (`/machine-learning/foundation-model-apis`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Models in Unity Catalog (`/machine-learning/manage-model-lifecycle`) | covered | [ch. 5](../05-model-lifecycle-uc.md) | 2026-10-15 |
| MLOps workflows (`/machine-learning/mlops/mlops-workflow`) | covered | [ch. 1](../01-databricks-ml-platform.md), [ch. 5](../05-model-lifecycle-uc.md) | 2026-10-15 |
| Batch inference (`/machine-learning/model-inference`) | covered | [ch. 12](../12-batch-and-streaming-inference.md) | 2026-10-15 |
| Model Serving (`/machine-learning/model-serving`) | covered | [ch. 13](../13-real-time-serving.md) | 2026-10-15 |
| Ray on Databricks (`/machine-learning/ray`) | beyond-exam | [ch. 10](../10-validation-and-tuning.md) | 2026-10-15 |
| Distributed training examples (`/machine-learning/train-model/distributed-training`) | out-of-scope | Deep learning and distributed training: not part of the exam guide | 2026-10-15 |
| DL best practices (`/machine-learning/train-model/dl-best-practices`) | out-of-scope | Deep learning and distributed training: not part of the exam guide | 2026-10-15 |
| Model training examples (`/machine-learning/train-model/training-examples`) | out-of-scope | Advanced training or feature technique not covered by the exam objectives | 2026-10-15 |
| MLflow for Models (`/mlflow`) | covered | [ch. 4](../04-mlflow-tracking.md) | 2026-10-15 |
| MLflow tracking (`/mlflow/tracking`) | covered | [ch. 4](../04-mlflow-tracking.md) | 2026-10-15 |
| Databricks notebooks (`/notebooks`) | covered | [ch. 6](../06-exploring-data.md) | 2026-10-15 |
| Use Genie Code for data science (`/notebooks/ds-agent`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |

## MLflow

Index: `/mlflow/`

| Page | Decision | Where / reason | Checked on |
|---|---|---|---|
| Databricks disaster recovery (`/admin/disaster-recovery`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |
| Custom Agents (`/agents`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Databricks Free Edition (`/getting-started/free-edition`) | covered | [ch. 1](../01-databricks-ml-platform.md) | 2026-10-15 |
| Machine learning (`/machine-learning`) | covered | index page | 2026-10-15 |
| Manage models using Databricks Unity Catalog (`/machine-learning/manage-model-lifecycle`) | covered | [ch. 5](../05-model-lifecycle-uc.md) | 2026-10-15 |
| Model Serving using Databricks (`/machine-learning/model-serving`) | covered | [ch. 13](../13-real-time-serving.md) | 2026-10-15 |
| Train AI models (`/machine-learning/train-model`) | covered | [ch. 1](../01-databricks-ml-platform.md), [ch. 8](../08-ml-foundations.md) | 2026-10-15 |
| Deployment jobs (`/mlflow/deployment-job`) | out-of-scope | Production operations detail beyond the exam's deploy/query/split objectives | 2026-10-15 |
| Logged Models (`/mlflow/logged-model`) | covered | [ch. 4](../04-mlflow-tracking.md) | 2026-10-15 |
| Get started with MLflow 3 for models (`/mlflow/mlflow-3-install`) | covered | [ch. 4](../04-mlflow-tracking.md), [ch. 5](../05-model-lifecycle-uc.md) | 2026-10-15 |
| Model Registry improvements with MLflow 3 (`/mlflow/model-registry-3`) | covered | [ch. 5](../05-model-lifecycle-uc.md) | 2026-10-15 |
| Experiment tracking using Databricks (`/mlflow/tracking`) | covered | [ch. 4](../04-mlflow-tracking.md) | 2026-10-15 |
| Agent Evaluation (`/mlflow3/genai/eval-monitor`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Genie Code for agent observability and evaluation (`/mlflow3/genai/getting-started/genie-code`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Databricks enterprise-grade security (`/security`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |
| customer-managed keys (`/security/keys`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |

## Feature store

Index: `/machine-learning/feature-store/`

| Page | Decision | Where / reason | Checked on |
|---|---|---|---|
| Machine learning (`/machine-learning`) | covered | index page | 2026-10-15 |
| Model Serving with automatic feature lookup (`/machine-learning/feature-store/automatic-feature-lookup`) | covered | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| Databricks Feature Store overview and glossary (`/machine-learning/feature-store/concepts`) | covered | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| Explore Feature Views in Unity Catalog (`/machine-learning/feature-store/explore-feature-views`) | beyond-exam | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| Serve feature tables (`/machine-learning/feature-store/feature-function-serving`) | beyond-exam | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| Example: Deploy and query a feature serving endpoint (`/machine-learning/feature-store/feature-serving-tutorial`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Feature Views (`/machine-learning/feature-store/feature-views`) | beyond-exam | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| Feature governance and lineage (`/machine-learning/feature-store/lineage`) | covered | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| Materialize Feature Views (`/machine-learning/feature-store/materialized-features`) | beyond-exam | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| On-demand feature computation (`/machine-learning/feature-store/on-demand-features`) | out-of-scope | Advanced training or feature technique not covered by the exam objectives | 2026-10-15 |
| Online Feature Store (`/machine-learning/feature-store/online-feature-store`) | covered | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| Purge feature values for entities (`/machine-learning/feature-store/purge-feature-entities`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Feature Engineering Python API (`/machine-learning/feature-store/python-api`) | covered | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| Example: use features with structured RAG applications (`/machine-learning/feature-store/rag`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Point-in-time feature joins (`/machine-learning/feature-store/time-series`) | beyond-exam | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| Train models with feature tables (`/machine-learning/feature-store/train-models-with-feature-store`) | covered | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| Feature tables (`/machine-learning/feature-store/uc/feature-tables-uc`) | covered | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| Explore feature tables in Unity Catalog (`/machine-learning/feature-store/uc/ui-uc`) | covered | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| Workspace Feature Store (deprecated) (`/machine-learning/feature-store/workspace-feature-store`) | covered | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| Real-time mode (`/structured-streaming/real-time/concepts`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |

## AutoML

Index: `/machine-learning/automl/`

| Page | Decision | Where / reason | Checked on |
|---|---|---|---|
| Create a workspace (`/admin/workspace`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |
| compute access mode (`/compute/configure`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |
| Machine learning (`/machine-learning`) | covered | index page | 2026-10-15 |
| Python API (`/machine-learning/automl/automl-api-reference`) | covered | [ch. 2](../02-automl.md) | 2026-10-15 |
| classification (`/machine-learning/automl/classification`) | covered | [ch. 2](../02-automl.md) | 2026-10-15 |
| Feature store integration (`/machine-learning/automl/feature-store-integration`) | beyond-exam | [ch. 2](../02-automl.md), [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| forecasting (`/machine-learning/automl/forecasting`) | covered | [ch. 2](../02-automl.md) | 2026-10-15 |
| regression (`/machine-learning/automl/regression`) | covered | [ch. 2](../02-automl.md) | 2026-10-15 |
| Python API (`/machine-learning/automl/regression-train-api`) | covered | [ch. 2](../02-automl.md) | 2026-10-15 |
| Databricks Runtime ML (`/machine-learning/databricks-runtime-ml`) | covered | [ch. 1](../01-databricks-ml-platform.md) | 2026-10-15 |
| Train models (`/machine-learning/train-model`) | covered | [ch. 1](../01-databricks-ml-platform.md), [ch. 8](../08-ml-foundations.md) | 2026-10-15 |
| Forecasting (serverless) (`/machine-learning/train-model/serverless-forecasting`) | beyond-exam | [ch. 2](../02-automl.md) | 2026-10-15 |

## Hyperparameter tuning

Index: `/machine-learning/automl-hyperparam-tuning/`

| Page | Decision | Where / reason | Checked on |
|---|---|---|---|
| Machine learning (`/machine-learning`) | covered | index page | 2026-10-15 |
| Use distributed training algorithms with Hyperopt (`/machine-learning/automl-hyperparam-tuning/hyperopt-distributed-ml`) | out-of-scope | Advanced training or feature technique not covered by the exam objectives | 2026-10-15 |
| Hyperparameter tuning with Optuna (`/machine-learning/automl-hyperparam-tuning/optuna`) | covered | [ch. 10](../10-validation-and-tuning.md) | 2026-10-15 |
| Databricks Runtime ML (`/machine-learning/databricks-runtime-ml`) | covered | [ch. 1](../01-databricks-ml-platform.md) | 2026-10-15 |
| What is Ray on Databricks? (`/machine-learning/ray`) | beyond-exam | [ch. 10](../10-validation-and-tuning.md) | 2026-10-15 |
| RayTune (`/machine-learning/ray/ray-mlflow`) | beyond-exam | [ch. 10](../10-validation-and-tuning.md) | 2026-10-15 |
| Train models (`/machine-learning/train-model`) | covered | [ch. 1](../01-databricks-ml-platform.md), [ch. 8](../08-ml-foundations.md) | 2026-10-15 |
| Model training examples (`/machine-learning/train-model/training-examples`) | out-of-scope | Advanced training or feature technique not covered by the exam objectives | 2026-10-15 |
| Databricks Autologging (`/mlflow/databricks-autologging`) | covered | [ch. 4](../04-mlflow-tracking.md) | 2026-10-15 |

## Model serving

Index: `/machine-learning/model-serving/`

| Page | Decision | Where / reason | Checked on |
|---|---|---|---|
| Monitor model serving costs (`/admin/system-tables/model-serving-cost`) | beyond-exam | [ch. 13](../13-real-time-serving.md) | 2026-10-15 |
| Develop agents on Databricks (`/agents`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Deploy an agent for AI applications (Model Serving) (`/agents/custom-agents/model-serving/deploy-agent`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| AI Gateway (`/ai-gateway`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| AI Search (`/ai-search/ai-search`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Unity Catalog (`/data-governance/unity-catalog`) | covered | [ch. 1](../01-databricks-ml-platform.md), [ch. 3](../03-feature-engineering-uc.md), [ch. 5](../05-model-lifecycle-uc.md) | 2026-10-15 |
| serverless compute (`/getting-started/high-level-architecture`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |
| AI Functions (`/large-language-models/ai-functions`) | covered | [ch. 12](../12-batch-and-streaming-inference.md) | 2026-10-15 |
| AI Playground (`/large-language-models/ai-playground`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Get started querying LLMs on Databricks (`/large-language-models/llm-serving-intro`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Machine learning (`/machine-learning`) | covered | index page | 2026-10-15 |
| Databricks Feature Store (`/machine-learning/feature-store`) | covered | [ch. 3](../03-feature-engineering-uc.md) | 2026-10-15 |
| Foundation Model APIs (`/machine-learning/foundation-model-apis`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| External models (`/machine-learning/foundation-models/external-models`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Tutorial: Create external model endpoints to query OpenAI models (`/machine-learning/foundation-models/external-models-tutorial`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Workspace Model Registry (`/machine-learning/manage-model-lifecycle/workspace-model-registry`) | covered | [ch. 5](../05-model-lifecycle-uc.md) | 2026-10-15 |
| Custom models (`/machine-learning/model-serving/custom-models`) | covered | [ch. 13](../13-real-time-serving.md) | 2026-10-15 |
| Foundation models (`/machine-learning/model-serving/foundation-model-overview`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Migrate to Model Serving (`/machine-learning/model-serving/migrate-model-serving`) | out-of-scope | Production operations detail beyond the exam's deploy/query/split objectives | 2026-10-15 |
| Tutorial: Deploy and query a custom model (`/machine-learning/model-serving/model-serving-intro`) | covered | [ch. 13](../13-real-time-serving.md) | 2026-10-15 |
| Model Serving limits and regions (`/machine-learning/model-serving/model-serving-limits`) | covered | [ch. 13](../13-real-time-serving.md) | 2026-10-15 |
| Optimize Model Serving endpoints for production (`/machine-learning/model-serving/production-optimization`) | out-of-scope | Production operations detail beyond the exam's deploy/query/split objectives | 2026-10-15 |
| Databricks Geos (`/resources/databricks-geos`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |
| Databricks Designated Service (`/resources/designated-services`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |
| Serving endpoint ACLs (`/security/auth/access-control`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |
| Manage entitlements (`/security/auth/entitlements`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |
| Manage network policies for serverless egress control (`/security/network/serverless-network-security/manage-network-policies`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |

## Model inference

Index: `/machine-learning/model-inference/`

| Page | Decision | Where / reason | Checked on |
|---|---|---|---|
| task-specific AI functions (`/large-language-models/ai-functions`) | covered | [ch. 12](../12-batch-and-streaming-inference.md) | 2026-10-15 |
| model types and the associated models (`/large-language-models/ai-query`) | covered | [ch. 12](../12-batch-and-streaming-inference.md) | 2026-10-15 |
| Deploy batch inference pipelines (`/large-language-models/batch-inference-pipelines`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Machine learning (`/machine-learning`) | covered | index page | 2026-10-15 |
| Deploy models using Model Serving (`/machine-learning/model-serving`) | covered | [ch. 13](../13-real-time-serving.md) | 2026-10-15 |
| Public Preview (`/release-notes/release-types`) | out-of-scope | Another runtime or release channel; the book targets Databricks Runtime 18 LTS ML | 2026-10-15 |

## MLOps

Index: `/machine-learning/mlops/mlops-workflow`

| Page | Decision | Where / reason | Checked on |
|---|---|---|---|
| Delta tables (`/delta`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Developer tools use cases (`/dev-tools`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Add notifications on a job (`/jobs/notifications`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| task values (`/jobs/task-values`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Automate jobs with schedules and triggers (`/jobs/triggers`) | beyond-exam | [ch. 1](../01-databricks-ml-platform.md) | 2026-10-15 |
| lakehouse architecture (`/lakehouse`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Machine learning (`/machine-learning`) | covered | index page | 2026-10-15 |
| AutoML (`/machine-learning/automl`) | covered | [ch. 2](../02-automl.md) | 2026-10-15 |
| Models in Unity Catalog (`/machine-learning/manage-model-lifecycle`) | covered | [ch. 5](../05-model-lifecycle-uc.md) | 2026-10-15 |
| Model deployment patterns (`/machine-learning/mlops/deployment-patterns`) | covered | [ch. 1](../01-databricks-ml-platform.md) | 2026-10-15 |
| MLflow (`/mlflow`) | covered | [ch. 4](../04-mlflow-tracking.md) | 2026-10-15 |
| MLflow tracking (`/mlflow/tracking`) | covered | [ch. 4](../04-mlflow-tracking.md) | 2026-10-15 |
| Databricks Git folders (`/repos`) | beyond-exam | [ch. 1](../01-databricks-ml-platform.md) | 2026-10-15 |

## Lakeflow pipelines

Index: `/ldp/`

| Page | Decision | Where / reason | Checked on |
|---|---|---|---|
| Data engineering (`/data-engineering`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Best practices (`/ldp/best-practices`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Pipeline concepts (`/ldp/concepts`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| How to use Lakeflow pipelines (`/ldp/concepts/how-to-use-pipelines`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Develop pipelines (`/ldp/concepts/pipelines`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Configure pipelines (`/ldp/configure-pipeline`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Standalone pipelines (`/ldp/dbsql/dbsql-for-ldp`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Developers (`/ldp/developer`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Pipeline limitations (`/ldp/limitations`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Monitor pipelines (`/ldp/observability`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Tutorials (`/ldp/tutorials`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |

## Runtime release notes

Index: `/release-notes/runtime/`

| Page | Decision | Where / reason | Checked on |
|---|---|---|---|
| Photon (`/compute/photon`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Databricks Runtime 14.3 LTS (`/release-notes/runtime/14.3lts`) | out-of-scope | Another runtime or release channel; the book targets Databricks Runtime 18 LTS ML | 2026-10-15 |
| Databricks Runtime 14.3 LTS for Machine Learning (`/release-notes/runtime/14.3lts-ml`) | out-of-scope | Another runtime or release channel; the book targets Databricks Runtime 18 LTS ML | 2026-10-15 |
| Databricks Runtime 15.4 LTS (`/release-notes/runtime/15.4lts`) | out-of-scope | Another runtime or release channel; the book targets Databricks Runtime 18 LTS ML | 2026-10-15 |
| Databricks Runtime 15.4 LTS for Machine Learning (`/release-notes/runtime/15.4lts-ml`) | out-of-scope | Another runtime or release channel; the book targets Databricks Runtime 18 LTS ML | 2026-10-15 |
| Databricks Runtime 16.4 LTS (`/release-notes/runtime/16.4lts`) | out-of-scope | Another runtime or release channel; the book targets Databricks Runtime 18 LTS ML | 2026-10-15 |
| Databricks Runtime 16.4 LTS for Machine Learning (`/release-notes/runtime/16.4lts-ml`) | out-of-scope | Another runtime or release channel; the book targets Databricks Runtime 18 LTS ML | 2026-10-15 |
| Databricks Runtime 17.3 LTS (`/release-notes/runtime/17.3lts`) | out-of-scope | Another runtime or release channel; the book targets Databricks Runtime 18 LTS ML | 2026-10-15 |
| Databricks Runtime 17.3 LTS for Machine Learning (`/release-notes/runtime/17.3lts-ml`) | out-of-scope | Another runtime or release channel; the book targets Databricks Runtime 18 LTS ML | 2026-10-15 |
| Databricks Runtime 18 LTS (`/release-notes/runtime/18`) | covered | [ch. 1](../01-databricks-ml-platform.md) | 2026-10-15 |
| Databricks Runtime 18 LTS for Machine Learning (`/release-notes/runtime/18ml`) | covered | [ch. 1](../01-databricks-ml-platform.md), [ch. 10](../10-validation-and-tuning.md) | 2026-10-15 |
| Databricks Runtime 19 (`/release-notes/runtime/19`) | out-of-scope | Another runtime or release channel; the book targets Databricks Runtime 18 LTS ML | 2026-10-15 |
| Databricks Runtime 19 for Machine Learning (`/release-notes/runtime/19ml`) | out-of-scope | Another runtime or release channel; the book targets Databricks Runtime 18 LTS ML | 2026-10-15 |
| Databricks Runtime support lifecycles (`/release-notes/runtime/databricks-runtime-ver`) | beyond-exam | [ch. 1](../01-databricks-ml-platform.md) | 2026-10-15 |
| End-of-support Databricks Runtime release notes (`/release-notes/runtime/eos`) | out-of-scope | Another runtime or release channel; the book targets Databricks Runtime 18 LTS ML | 2026-10-15 |
| Databricks Runtime maintenance updates (`/release-notes/runtime/maintenance-updates`) | out-of-scope | Another runtime or release channel; the book targets Databricks Runtime 18 LTS ML | 2026-10-15 |
| Serverless compute release notes (`/release-notes/serverless`) | out-of-scope | Another runtime or release channel; the book targets Databricks Runtime 18 LTS ML | 2026-10-15 |

## Notebooks

Index: `/notebooks/`

| Page | Decision | Where / reason | Checked on |
|---|---|---|---|
| Import and visualize CSV data from a notebook (`/getting-started/import-visualize-data`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Query and visualize data from a notebook (`/getting-started/quick-start`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| End-to-end classic ML models (`/mlflow/end-to-end-example`) | covered | [ch. 4](../04-mlflow-tracking.md), [ch. 5](../05-model-lifecycle-uc.md) | 2026-10-15 |
| Basic editing (`/notebooks/basic-editing`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Best practices (`/notebooks/best-practices`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Code help using Genie Code (`/notebooks/code-assistant`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| Dashboards in notebooks (`/notebooks/dashboards`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Debug notebooks (`/notebooks/debugger`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Use the Data Science Agent (`/notebooks/ds-agent`) | out-of-scope | Generative AI, agents or foundation models: not part of the ML Associate exam guide | 2026-10-15 |
| EDA techniques using Databricks notebooks (`/notebooks/eda-tutorial`) | covered | [ch. 6](../06-exploring-data.md) | 2026-10-15 |
| Import and export notebooks (`/notebooks/notebook-export-import`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Notebook outputs and results (`/notebooks/notebook-outputs`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Orchestrate notebooks and modularize code (`/notebooks/notebook-workflows`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Develop code in notebooks (`/notebooks/notebooks-code`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Collaborate using notebooks (`/notebooks/notebooks-collaborate`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Run notebooks (`/notebooks/run-notebook`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Unit testing (`/notebooks/testing`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Databricks widgets (`/notebooks/widgets`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |

## dbutils

Index: `/dev-tools/databricks-utils`

| Page | Decision | Where / reason | Checked on |
|---|---|---|---|
| credential passthrough (`/archive/credential-passthrough/iam-passthrough`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |
| Library utility (dbutils.library) (legacy) (`/archive/dev-tools/dbutils-library`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |
| Connect to Amazon S3 (`/archive/storage/amazon-s3`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |
| What is DBFS? (`/dbfs`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Databricks Utilities for Scala, with Java (`/dev-tools/sdk-java`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Databricks Utilities for Python (`/dev-tools/sdk-python`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Sample datasets (`/discover/databricks-datasets`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| workspace files (`/files/workspace`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| parameters for notebook tasks (`/jobs/task-parameters`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Read values from a For each task's iterations (`/jobs/task-values`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Restart the Python process on Databricks (`/libraries/restart-python-process`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Orchestrate Databricks notebooks and modularize code (`/notebooks/notebook-workflows`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| magic commands (`/notebooks/notebooks-code`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Databricks widgets (`/notebooks/widgets`) | out-of-scope | General platform, developer-tool or data-engineering feature beyond what the exam objectives need | 2026-10-15 |
| Public Preview (`/release-notes/release-types`) | out-of-scope | Another runtime or release channel; the book targets Databricks Runtime 18 LTS ML | 2026-10-15 |
| deprecated (`/release-notes/runtime/databricks-runtime-ver`) | beyond-exam | [ch. 1](../01-databricks-ml-platform.md) | 2026-10-15 |
| Secret management (`/security/secrets`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |
| Step 3: Use the secrets in a notebook (`/security/secrets/example-secret-workflow`) | out-of-scope | Account administration, security or platform architecture: not part of the exam guide | 2026-10-15 |
| structured streaming (`/structured-streaming/production`) | beyond-exam | [ch. 12](../12-batch-and-streaming-inference.md) | 2026-10-15 |
