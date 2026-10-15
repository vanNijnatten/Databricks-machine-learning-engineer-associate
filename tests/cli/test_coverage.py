"""``book coverage`` reports objective coverage and writes it into book/README.md."""

from typer.testing import CliRunner

from db_ml_associate.cli import app
from db_ml_associate.cli.coverage import (
    compute_coverage,
    render_coverage_table,
    write_coverage,
)
from db_ml_associate.content.loaders import load_book
from helpers.fixture_book import FixtureBook

README = "# Book\n\nIntro.\n\n<!-- coverage:start -->\nold table\n<!-- coverage:end -->\n\nOutro.\n"


def test_every_objective_is_listed_with_counts(fixture_book: FixtureBook) -> None:
    coverage = compute_coverage(load_book(fixture_book.root))
    assert len(coverage) == 48
    first = coverage[0]
    assert first.objective.id == "S1.01"
    assert first.sections == ("01-platform.md#everything-in-one-section",)
    assert first.tutorial_steps == ("ch01:everything",)
    assert first.questions == 1
    assert first.gaps == ()


def test_gaps_are_flagged(fixture_book: FixtureBook) -> None:
    fixture_book.edit_chapter(", S4.06 -->", " -->")
    coverage = {row.objective.id: row for row in compute_coverage(load_book(fixture_book.root))}
    assert coverage["S4.06"].gaps == ("chapter section",)


def test_practical_objective_needs_a_tutorial_step(fixture_book: FixtureBook) -> None:
    fixture_book.edit_chapter("objectives: [S1.01, S1.04", "objectives: [S1.04")
    coverage = {row.objective.id: row for row in compute_coverage(load_book(fixture_book.root))}
    assert coverage["S1.01"].gaps == ("tutorial step",)
    assert coverage["S1.02"].gaps == ()  # conceptual objectives need no tutorial step


def test_table_has_a_row_per_objective(fixture_book: FixtureBook) -> None:
    table = render_coverage_table(compute_coverage(load_book(fixture_book.root)))
    rows = [line for line in table.splitlines() if line.startswith("| S")]
    assert len(rows) == 48
    assert "[01-platform.md#everything-in-one-section](01-platform.md#everything-in-one-section)" in rows[0]


def test_write_replaces_only_the_marked_block_and_is_idempotent(
    fixture_book: FixtureBook,
) -> None:
    readme = fixture_book.book / "README.md"
    readme.write_text(README)
    table = render_coverage_table(compute_coverage(load_book(fixture_book.root)))
    assert write_coverage(readme, table) is True
    first = readme.read_text()
    assert first.startswith("# Book\n\nIntro.\n\n<!-- coverage:start -->\n")
    assert first.endswith("<!-- coverage:end -->\n\nOutro.\n")
    assert "old table" not in first
    assert write_coverage(readme, table) is False
    assert readme.read_text() == first


def test_cli_write(fixture_book: FixtureBook) -> None:
    readme = fixture_book.book / "README.md"
    readme.write_text(README)
    result = CliRunner().invoke(app, ["book", "coverage", "--root", str(fixture_book.root), "--write"])
    assert result.exit_code == 0, result.output
    assert "| S1.01" in readme.read_text()
