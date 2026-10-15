"""Fixtures for CLI tests."""

from datetime import date
from pathlib import Path

import pytest

from helpers.fixture_book import FixtureBook, build_fixture_book


@pytest.fixture
def fixture_book(tmp_path: Path) -> FixtureBook:
    """A complete, valid miniature book whose references were verified today."""
    return build_fixture_book(tmp_path, date.today())
