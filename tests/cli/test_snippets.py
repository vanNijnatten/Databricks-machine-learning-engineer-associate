"""Book code blocks are kept in sync with marked regions of tested source files."""

from pathlib import Path

import pytest

from db_ml_associate.cli.snippets import (
    SnippetError,
    extract_region,
    sync_book,
    sync_markdown,
)

SOURCE = """\
import pandas as pd


def impute_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
    # --8<-- [start:impute_median]
    median = df[column].median()
    return df.fillna({column: median})
    # --8<-- [end:impute_median]
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    source = tmp_path / "src" / "pkg" / "impute.py"
    source.parent.mkdir(parents=True)
    source.write_text(SOURCE)
    (tmp_path / "book").mkdir()
    return tmp_path


def chapter(body: str) -> str:
    return f"# Chapter\n\nSome text.\n\n{body}\n\nMore text.\n"


SYNCED_BLOCK = (
    "<!-- snippet: src/pkg/impute.py#impute_median -->\n"
    "```python\n"
    "median = df[column].median()\n"
    "return df.fillna({column: median})\n"
    "```"
)


def test_extract_region_is_dedented() -> None:
    assert extract_region(SOURCE, "impute_median") == (
        "median = df[column].median()\nreturn df.fillna({column: median})\n"
    )


def test_missing_region_raises() -> None:
    with pytest.raises(SnippetError, match="no_such_region"):
        extract_region(SOURCE, "no_such_region")


def test_sync_rewrites_a_stale_block(repo: Path) -> None:
    stale = SYNCED_BLOCK.replace("median()", "mean()")
    result = sync_markdown(chapter(stale), repo)
    assert result.text == chapter(SYNCED_BLOCK)
    assert len(result.stale) == 1
    assert result.errors == []


def test_sync_leaves_an_up_to_date_block_unchanged(repo: Path) -> None:
    result = sync_markdown(chapter(SYNCED_BLOCK), repo)
    assert result.text == chapter(SYNCED_BLOCK)
    assert result.stale == []


def test_illustrative_blocks_are_ignored(repo: Path) -> None:
    body = "<!-- illustrative -->\n```python\nprint('anything')\n```"
    result = sync_markdown(chapter(body), repo)
    assert result.text == chapter(body)
    assert result.stale == []
    assert result.errors == []


def test_missing_region_is_an_error(repo: Path) -> None:
    body = "<!-- snippet: src/pkg/impute.py#missing -->\n```python\nx = 1\n```"
    result = sync_markdown(chapter(body), repo)
    assert len(result.errors) == 1
    assert "missing" in result.errors[0][1]


def test_missing_source_file_is_an_error(repo: Path) -> None:
    body = "<!-- snippet: src/pkg/nope.py#impute_median -->\n```python\nx = 1\n```"
    result = sync_markdown(chapter(body), repo)
    assert len(result.errors) == 1


def test_snippet_comment_without_code_block_is_an_error(repo: Path) -> None:
    body = "<!-- snippet: src/pkg/impute.py#impute_median -->\n\nJust text."
    result = sync_markdown(chapter(body), repo)
    assert len(result.errors) == 1


def test_sync_book_check_mode_reports_without_writing(repo: Path) -> None:
    stale = chapter(SYNCED_BLOCK.replace("median()", "mean()"))
    path = repo / "book" / "03-chapter.md"
    path.write_text(stale)
    issues = sync_book(repo, check=True)
    assert len(issues) == 1
    assert issues[0].path.endswith("03-chapter.md")
    assert path.read_text() == stale


def test_sync_book_write_mode_updates_files(repo: Path) -> None:
    path = repo / "book" / "03-chapter.md"
    path.write_text(chapter(SYNCED_BLOCK.replace("median()", "mean()")))
    assert sync_book(repo, check=False) == []
    assert path.read_text() == chapter(SYNCED_BLOCK)


def test_sql_comment_markers() -> None:
    sql = "-- --8<-- [start:ddl]\nCREATE TABLE t (id INT);\n-- --8<-- [end:ddl]\n"
    assert extract_region(sql, "ddl") == "CREATE TABLE t (id INT);\n"
