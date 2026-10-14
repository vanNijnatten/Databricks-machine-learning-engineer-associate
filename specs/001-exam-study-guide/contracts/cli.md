# Contract: `db-ml-associate` CLI

Entry point already declared in `pyproject.toml` (`db-ml-associate = "db_ml_associate:main"`). Implemented with `typer`; output with `rich`. All commands run from the repository root, need no network unless stated, and exit `0` on success, `1` on validation failure, `2` on usage error.

## `book check [--strict]`

Validates all content against [content-formats.md](content-formats.md) and [../data-model.md](../data-model.md):

| Check | Failure condition |
|---|---|
| objectives | ≠ 48 objectives or section counts ≠ 18/9/15/6 |
| coverage | objective without chapter section or question; practical objective without tutorial step |
| chapters | in `book/NN-*.md` (01–13 only; introduction and appendices exempt): missing required headings/front matter, `objectives` empty; < 8 review questions |
| questions | schema error; duplicate id; answer/type mismatch; missing distractor explanation; near-duplicate of an official sample question |
| exams | ≠ 48 questions; section counts ≠ 18/9/15/6; question reused across exams or with review pool |
| totals | question bank < 200, or any section's share of all questions more than ±5 percentage points from its share of objectives (37.5 / 18.75 / 31.25 / 12.5 %) |
| snippets | stale synced block or missing source region |
| links | broken relative link or anchor in `book/` |
| references | a docs.databricks.com URL in `book/` not present in `references.yaml`; `last_verified` > 90 days (warning; error with `--strict`) |
| platform changes | fewer than the required minimum set (SC-007) |
| plans | chapter scheduled before one of its `prerequisites` (warning); chapter/tutorial/exam not scheduled; day outside 90–180 min; fast > 14 days or standard > 21 days |
| runtime | `pyproject.toml` pins ≠ `runtime.toml` |

Output: a table of failures (`file:line — message`), then a summary line `N errors, M warnings`.

## `book sync-snippets [--check]`

Rewrites snippet blocks from source regions. `--check` reports stale blocks without writing (exit 1 if any).

## `book coverage [--write]`

Prints the objective → chapter section / tutorial step / question count table; `--write` replaces the block between `<!-- coverage:start -->` and `<!-- coverage:end -->` in `book/README.md`.

## `book check-links` *(network)*

Fetches every `references.yaml` URL (and the exam guide URL). Reports HTTP status and final URL; fails on 4xx/5xx or redirects to a different path. `--update-verified` sets `last_verified` to today for URLs returning 200 at the same path (content re-reading remains a manual task step).

## `exam take <PE1|PE2> [--no-timer] [--section N]`

Interactive terminal practice exam: shuffled option order (seeded per attempt), countdown of `minutes`, single/multiple selection input. At the end prints per-section score, overall score, 80 % readiness verdict per section, and for each missed question the `chapter_ref` to revisit (FR-018). Results optionally written to `~/.db-ml-associate/attempts/<timestamp>.json` (`--save`).

## `exam render <PE1|PE2> [--answers]`

Writes a Markdown version to stdout (questions only, or answer key with explanations), for paper practice and the `book/appendix-d-*.md` files.

## `datasets generate <churn|house_prices> [--rows N] [--seed S] [--out PATH]`

Writes the deterministic synthetic dataset as CSV/Parquet. With default `--rows`/`--seed` and `--out data/<name>.parquet` it reproduces the committed file (used to regenerate `data/`).

## Python API (importable in notebooks)

```python
from db_ml_associate.datasets import load_customer_churn, load_house_prices  # committed data/*.parquet -> pandas.DataFrame
from db_ml_associate.datasets import make_customer_churn, make_house_prices  # generators
from db_ml_associate.ch07_cleaning import impute, outliers, encoding           # Spark + pandas helpers
...
```

Public functions are typed and documented (Principle III); signatures are defined during implementation per chapter.
