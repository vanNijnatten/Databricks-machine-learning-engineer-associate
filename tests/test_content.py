"""The real book passes ``book check`` (non-strict) and its snippets are in sync with the code."""

from pathlib import Path

from db_ml_associate.cli.book_check import run_checks
from db_ml_associate.cli.snippets import sync_book

ROOT = Path(__file__).resolve().parents[1]


def test_book_check_has_no_errors() -> None:
    errors = [str(issue) for issue in run_checks(ROOT) if issue.severity == "error"]
    assert errors == []


def test_snippets_are_in_sync() -> None:
    assert [str(issue) for issue in sync_book(ROOT, check=True)] == []
