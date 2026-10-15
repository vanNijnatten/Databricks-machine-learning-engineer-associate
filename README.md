# Databricks Machine Learning Associate

![Smoke Test](https://github.com/vanNijnattenDataCraft/db-ml-associate/actions/workflows/smoke_test.yaml/badge.svg)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

A study guide and hands-on tutorial for the **Databricks Certified Machine Learning Associate** exam.
It covers all 48 objectives of the exam guide (version 2025-03-01) in 13 chapters, with tested code,
a tutorial notebook per chapter, 121 review questions, two 48-question practice exams, 14- and 21-day
study plans and a cheat sheet.

**Start reading: [book/README.md](book/README.md).**

| | |
|---|---|
| Target runtime | Databricks Runtime 18 LTS for Machine Learning (Python 3.12, Spark 4.1, MLflow 3.8) — see [`runtime.toml`](runtime.toml) |
| Book | [`book/`](book/) — introduction, chapters 01–13, appendices A–D |
| Tutorials | [`notebooks/`](notebooks/) — Databricks notebooks; run `00_setup` first and `99_cleanup` last |
| Tested code | [`src/db_ml_associate/`](src/db_ml_associate/) — every code block in the book is copied from here |
| Datasets | [`data/`](data/) — two small synthetic datasets ([description](data/README.md)) |

## Quick start

**In a Databricks workspace:** add this repository as a Git folder, open
`notebooks/00_setup.py` and run it on Databricks Runtime 18 LTS ML (or a serverless notebook), then
follow the chapter tutorials. The [introduction](book/introduction.md#setting-up-a-workspace) explains
which tutorials work on Free Edition.

**Locally** (devcontainer included; needs Java 17 for the Spark tests):

```bash
uv sync --all-groups                       # Python 3.12 and the runtime's library versions
uv run pytest                              # tests, including local runs of the tutorial notebooks
uv run db-ml-associate book check          # validate the book: coverage, structure, questions, links
uv run db-ml-associate exam take PE1       # timed practice exam in the terminal, scored per section
```

Other commands: `book sync-snippets`, `book coverage --write`, `book check-links` (network),
`book render-plans`, `book render-appendix-a`, `book render-appendix-d`, `exam render`,
`datasets generate`. Run `uv run db-ml-associate --help` for details.

## Keeping the book current

The book cites the Databricks documentation for every platform claim
([`book/references.yaml`](book/references.yaml)); a weekly CI job checks that every cited page still
exists, and [`book/exam/docs-coverage.md`](book/exam/docs-coverage.md) records which documentation pages
are covered. When a new Runtime ML LTS is released, follow the runtime upgrade checklist in
[CONTRIBUTING](docs/CONTRIBUTING.md#runtime-upgrade-checklist).

The questions are original and are not taken from the official exam. This project is not affiliated
with or endorsed by Databricks.

# Code of Conduct

The Databricks Machine Learning Associate is an open source project developed by a lonely developer in need of contributors.
- Please read the [Code of Conduct](./docs/CODE_OF_CONDUCT.md) for guidance on how to interact with others in a way that makes a community thrive.
- Please also read how to [Contribute](./docs/CONTRIBUTING.md), if you would like to contribute.


# Call for Contributions

Writing code isn’t the only way to contribute. You can also:
- review pull requests and report errors in the book or the questions
- help us stay on top of new and old issues
- develop tutorials, presentations, and other educational materials
- translate content
- help with outreach and onboard new contributors

If you are new to contributing to open source, [this guide](https://opensource.guide/how-to-contribute/) helps explain why, what, and how to successfully get involved.
