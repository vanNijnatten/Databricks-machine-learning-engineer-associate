"""Appendix A is rendered from book/exam/platform-changes.yaml."""

from typer.testing import CliRunner

from db_ml_associate.cli import app
from db_ml_associate.cli.appendix import render_appendix_a
from db_ml_associate.content.loaders import load_book
from helpers.fixture_book import FixtureBook


def test_table_has_a_row_per_change(fixture_book: FixtureBook) -> None:
    book = load_book(fixture_book.root)
    text = render_appendix_a(book)
    rows = [line for line in text.splitlines() if line.startswith("| ") and not line.startswith("| Exam")]
    assert len(rows) == len(book.platform_changes)
    assert "Hyperopt fmin / SparkTrials" in text
    assert "[ml-overview](https://docs.databricks.com/aws/en/machine-learning/)" in text
    assert "[ch. 1](01-platform.md)" in text


def test_cli_writes_the_appendix(fixture_book: FixtureBook) -> None:
    result = CliRunner().invoke(app, ["book", "render-appendix-a", "--root", str(fixture_book.root)])
    assert result.exit_code == 0, result.output
    written = (fixture_book.book / "appendix-a-exam-vs-platform.md").read_text()
    assert written.startswith("# Appendix A")
    assert "Delta Live Tables" in written
