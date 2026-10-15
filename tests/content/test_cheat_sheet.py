"""Appendix B, the revision cheat sheet: short, complete and linked to the chapters (FR-020, US5)."""

import re
from pathlib import Path

import pytest

from db_ml_associate.content.loaders import parse_markdown

BOOK = Path(__file__).resolve().parents[2] / "book"
CHEAT_SHEET = BOOK / "appendix-b-cheat-sheet.md"
MAX_WORDS = 1600  # about four printed pages
REQUIRED_SECTIONS = ("Key API calls", "Decision tables", "Formulas", "Exam vs current platform")
CHAPTER_LINK = re.compile(r"\]\(((\d{2})-[a-z0-9-]+\.md)#([a-z0-9-]+)\)")


def _text() -> str:
    return CHEAT_SHEET.read_text(encoding="utf-8")


def _table_rows() -> list[str]:
    return [
        line
        for line in _text().splitlines()
        if line.startswith("|") and not re.match(r"^\|[\s:|-]+\|$", line) and not line.startswith("| **")
    ]


def _header_rows() -> set[str]:
    lines = _text().splitlines()
    return {lines[i - 1] for i, line in enumerate(lines) if re.match(r"^\|[\s:|-]+\|$", line)}


def test_fits_in_about_four_pages() -> None:
    words = len(re.sub(r"\]\([^)]*\)", "]", _text()).split())  # link targets do not count as words
    assert words <= MAX_WORDS, f"{words} words"


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_has_required_section(section: str) -> None:
    assert re.search(rf"^## {re.escape(section)}$", _text(), re.MULTILINE)


def test_every_table_row_links_to_a_chapter_anchor_that_exists() -> None:
    headers = _header_rows()
    rows = [row for row in _table_rows() if row not in headers]
    assert len(rows) >= 40
    for row in rows:
        links = CHAPTER_LINK.findall(row)
        assert links, f"no chapter link in row: {row}"
        for file_name, _, anchor in links:
            target = BOOK / file_name
            assert target.is_file(), file_name
            anchors = parse_markdown(target.read_text(encoding="utf-8"), file_name).anchors
            assert anchor in anchors, f"{file_name}#{anchor} not found"


def test_scenario_lookup_for_low_latency_with_fresh_features() -> None:
    text = _text().lower()
    assert "low-latency predictions with fresh features" in text
    row = next(line for line in text.splitlines() if "low-latency predictions with fresh features" in line)
    assert "online" in row
    assert "model serving" in row
