"""Tutorial notebooks are valid Databricks source files and match their chapters' tutorial metadata."""

import re
from pathlib import Path

import pytest

from db_ml_associate.content.loaders import load_book

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.py"))
CHAPTER_NOTEBOOK = re.compile(r"^ch\d{2}_[a-z0-9_]+\.py$")
HEADER_EXEMPT = {"ch12_pipeline_source.py"}
NOTEBOOK_SOURCE = "# Databricks notebook source"
CELL_SEPARATOR = "# COMMAND ----------"
HEADER_FIELDS = ("Objectives", "Runtime", "Requires", "Free Edition", "Estimated time", "Clean-up")
HEADER_LINES = 8
BOOK = load_book(ROOT)


def _chapter_notebooks() -> list[Path]:
    return [path for path in NOTEBOOKS if CHAPTER_NOTEBOOK.match(path.name) and path.name not in HEADER_EXEMPT]


def _header(path: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[:HEADER_LINES])


def _header_value(header: str, name: str) -> str:
    match = re.search(rf"\*\*{re.escape(name)}\*\*:\s*([^·\n]+)", header)
    assert match, f"header field {name} missing"
    return match.group(1).strip()


def test_there_is_a_notebook_for_every_chapter() -> None:
    numbers = {int(path.name[2:4]) for path in _chapter_notebooks()}
    assert numbers == set(range(1, 14))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_is_databricks_source(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == NOTEBOOK_SOURCE
    if path.name not in HEADER_EXEMPT:
        assert CELL_SEPARATOR in text, "cells must be separated by '# COMMAND ----------'"
    code = "\n".join(line for line in text.splitlines() if not line.startswith("# MAGIC"))
    compile(code, str(path), "exec")


@pytest.mark.parametrize("path", _chapter_notebooks(), ids=lambda path: path.name)
def test_chapter_notebook_has_header(path: Path) -> None:
    header = _header(path)
    assert re.search(r"^# MAGIC # Chapter \d+ tutorial: ", header, re.MULTILINE)
    for name in HEADER_FIELDS:
        _header_value(header, name)


@pytest.mark.parametrize("chapter", BOOK.chapters, ids=lambda chapter: chapter.name)
def test_chapter_tutorial_metadata_matches_notebook(chapter: object) -> None:
    from db_ml_associate.content.models import Chapter

    assert isinstance(chapter, Chapter)
    tutorial = chapter.tutorial
    assert tutorial is not None, "chapter front matter needs a tutorial"
    for file in (tutorial.notebook, *tutorial.extra_files):
        assert (ROOT / file).is_file(), f"{file} does not exist"
    assert tutorial.steps, "tutorial needs at least one step"
    assert not tutorial.problems()
    header = _header(ROOT / tutorial.notebook)
    assert _header_value(header, "Free Edition") == tutorial.free_edition
    assert _header_value(header, "Estimated time") == f"{tutorial.minutes} min"
