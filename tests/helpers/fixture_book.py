"""A small but valid book used to exercise ``book check`` and ``book coverage``."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from db_ml_associate.cli.snippets import sync_book

SECTION_COUNTS = {1: 18, 2: 9, 3: 15, 4: 6}
PRACTICAL_EVERY = 3  # every third objective is practical


def objective_ids() -> list[str]:
    return [f"S{section}.{n:02d}" for section, total in SECTION_COUNTS.items() for n in range(1, total + 1)]


def practical_ids() -> list[str]:
    return objective_ids()[::PRACTICAL_EVERY]


DOCS_URL = "https://docs.databricks.com/aws/en/machine-learning/"

CHAPTER_TEMPLATE = """\
---
chapter: 1
title: The platform
objectives: [{objectives}]
prerequisites: []
reading_minutes: 60
references: [ml-overview]
tutorial:
  notebook: notebooks/ch01_platform.py
  minutes: 30
  requires: [serverless]
  free_edition: full
  cleanup: false
  steps:
    - id: everything
      objectives: [{practical}]
---

# 1. The platform

> **Exam mapping**: all objectives in this fixture.

## Learning objectives

- Understand one.
- Understand two.
- Understand three.

## Everything in one section

<!-- objectives: {objectives} -->

See the [ML docs]({docs_url}) and the [summary](#summary).

<!-- snippet: src/pkg/demo.py#demo -->
```python
value = 1
```

## Pitfalls

- Do not do this.

## Summary

Short summary.

## Review questions

<!-- questions: ch01 -->
"""

DEMO_SOURCE = """\
def demo() -> int:
    # --8<-- [start:demo]
    value = 1
    # --8<-- [end:demo]
    return value
"""

PLATFORM_CHANGES = [
    {
        "exam_term": "Hyperopt fmin / SparkTrials",
        "current_term": "Optuna",
        "change": "deprecated",
    },
    {
        "exam_term": "Delta Live Tables",
        "current_term": "Lakeflow pipelines",
        "change": "renamed",
    },
    {
        "exam_term": "Online tables",
        "current_term": "Online Feature Store",
        "change": "replaced",
    },
    {
        "exam_term": "Workspace model registry and stages",
        "current_term": "UC aliases",
        "change": "replaced",
    },
    {
        "exam_term": "FeatureStoreClient",
        "current_term": "FeatureEngineeringClient",
        "change": "replaced",
    },
    {
        "exam_term": "AutoML built into the ML runtime",
        "current_term": "databricks-automl-runtime",
        "change": "repackaged",
    },
    {
        "exam_term": "MLflow 2 artifact_path",
        "current_term": "MLflow 3 name / LoggedModel",
        "change": "renamed",
    },
]


def question(question_id: str, objective: str, pool: str = "review") -> dict[str, object]:
    return {
        "id": question_id,
        "objective": objective,
        "pool": pool,
        "type": "single",
        "difficulty": "easy",
        "stem": f"A data scientist working on scenario {question_id} must pick an approach for {objective}.",
        "options": {"A": "First", "B": "Second", "C": "Third", "D": "Fourth"},
        "answer": ["A"],
        "explanation": "First is right.",
        "distractors": {"B": "No.", "C": "No.", "D": "No."},
        "chapter_ref": "01-platform.md#everything-in-one-section",
    }


@dataclass
class FixtureBook:
    """Paths and helpers for a generated fixture book."""

    root: Path

    @property
    def book(self) -> Path:
        return self.root / "book"

    @property
    def chapter(self) -> Path:
        return self.book / "01-platform.md"

    def read_yaml(self, relative: str) -> object:
        return yaml.safe_load((self.book / relative).read_text())

    def write_yaml(self, relative: str, data: object) -> None:
        path = self.book / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    def edit_chapter(self, old: str, new: str) -> None:
        text = self.chapter.read_text()
        assert old in text, old
        self.chapter.write_text(text.replace(old, new, 1))


def build_fixture_book(root: Path, today: date) -> FixtureBook:
    """Write a complete, valid miniature book under ``root``."""
    fixture = FixtureBook(root)
    ids = objective_ids()
    fixture.write_yaml(
        "exam/objectives.yaml",
        {
            "exam_guide": {
                "version": date(2025, 3, 1),
                "url": "https://example.com/guide.pdf",
                "questions": 48,
                "minutes": 90,
                "checked_on": today,
            },
            "objectives": [
                {
                    "id": objective_id,
                    "section": int(objective_id[1]),
                    "kind": "practical" if objective_id in practical_ids() else "conceptual",
                    "text": f"Objective {objective_id}",
                }
                for objective_id in ids
            ],
        },
    )
    fixture.chapter.write_text(
        CHAPTER_TEMPLATE.format(
            objectives=", ".join(ids),
            practical=", ".join(practical_ids()),
            docs_url=DOCS_URL,
        )
    )
    fixture.write_yaml(
        "questions/ch01.yaml",
        [question(f"Q-01-{n:03d}", objective) for n, objective in enumerate(ids, start=1)],
    )
    fixture.write_yaml(
        "references.yaml",
        [
            {
                "id": "ml-overview",
                "url": DOCS_URL,
                "title": "Machine learning",
                "claim": "Overview page.",
                "objectives": ["S1.01"],
                "chapters": [1],
                "last_verified": today,
                "status": "ok",
            }
        ],
    )
    fixture.write_yaml(
        "exam/platform-changes.yaml",
        [
            {
                **change,
                "exam_expects": "x",
                "objectives": ["S1.01"],
                "references": ["ml-overview"],
            }
            for change in PLATFORM_CHANGES
        ],
    )
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "demo.py").write_text(DEMO_SOURCE)
    (root / "notebooks").mkdir()
    (root / "notebooks" / "ch01_platform.py").write_text("# Databricks notebook source\n")
    (root / "runtime.toml").write_text(
        'databricks_runtime = "18 LTS ML"\ndocs_snapshot = 2026-10-14\npython = "3.12.3"\nspark = "4.1.0"\n\n'
        '[libraries]\npandas = "2.2.3"\n'
    )
    (root / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["pandas==2.2.3"]\n')
    (root / ".python-version").write_text("3.12\n")
    sync_book(root)  # render the review questions, as an author would
    return fixture
