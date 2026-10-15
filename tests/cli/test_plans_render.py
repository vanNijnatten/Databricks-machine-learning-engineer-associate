"""Appendix C is rendered from book/plans/*.yaml with computed minutes."""

from typer.testing import CliRunner

from db_ml_associate.cli import app
from db_ml_associate.cli.plans import render_plans
from db_ml_associate.content.loaders import load_book
from helpers.fixture_book import FixtureBook

PLAN = {
    "track": "fast",
    "days": [
        {
            "day": 1,
            "goals": ["Start", "Read chapter 1"],
            "items": [{"introduction": True}, {"chapter": 1}],
            "self_check": "Can you explain it?",
        },
        {
            "day": 2,
            "goals": ["Practise"],
            "items": [{"tutorial": "notebooks/ch01_platform.py"}, {"review": "Weak areas"}, {"appendix": "A"}],
            "self_check": "Did it run?",
        },
        {"day": 3, "goals": ["Exam"], "items": [{"exam": "PE1"}], "self_check": "80 %?"},
    ],
}


def _rows(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("| ") and not line.startswith("| Day")]


def test_one_table_per_track_with_a_row_per_day(fixture_book: FixtureBook) -> None:
    fixture_book.write_yaml("plans/fast-14.yaml", PLAN)
    fixture_book.write_yaml("plans/standard-21.yaml", {**PLAN, "track": "standard"})
    text = render_plans(load_book(fixture_book.root))
    assert text.startswith("# Appendix C")
    assert text.count("| Day |") == 2
    assert len(_rows(text)) == 6
    assert text.index("21-day") < text.index("14-day")  # the standard plan comes first


def test_rows_show_goals_linked_items_minutes_and_self_check(fixture_book: FixtureBook) -> None:
    fixture_book.write_yaml("plans/fast-14.yaml", PLAN)
    first, second, third = _rows(render_plans(load_book(fixture_book.root)))
    assert "Start; Read chapter 1" in first
    assert "[Introduction](introduction.md)" in first
    assert "[Chapter 1: The platform](01-platform.md)" in first
    assert "| 176 |" in first  # 20 + 60 + 48 questions × 2
    assert "Can you explain it?" in first
    assert "[Tutorial 1](../notebooks/ch01_platform.py)" in second
    assert "Review: Weak areas" in second
    assert "Appendix A" in second
    assert "Practice exam PE1" in third
    assert "| 135 |" in third


def test_optional_items_are_marked_and_not_counted(fixture_book: FixtureBook) -> None:
    plan = {
        "track": "fast",
        "days": [
            {
                "day": 1,
                "goals": ["g"],
                "items": [{"chapter": 1}, {"tutorial": "notebooks/ch01_platform.py", "optional": True}],
                "self_check": "s",
            }
        ],
    }
    fixture_book.write_yaml("plans/fast-14.yaml", plan)
    (row,) = _rows(render_plans(load_book(fixture_book.root)))
    assert "(optional, 30 min)" in row
    assert "| 156 |" in row


def test_rendering_is_idempotent(fixture_book: FixtureBook) -> None:
    fixture_book.write_yaml("plans/fast-14.yaml", PLAN)
    book = load_book(fixture_book.root)
    assert render_plans(book) == render_plans(book)


def test_cli_writes_appendix_c(fixture_book: FixtureBook) -> None:
    fixture_book.write_yaml("plans/fast-14.yaml", PLAN)
    runner = CliRunner()
    result = runner.invoke(app, ["book", "render-plans", "--root", str(fixture_book.root)])
    assert result.exit_code == 0, result.output
    target = fixture_book.book / "appendix-c-study-plans.md"
    first = target.read_text()
    assert "14-day" in first
    runner.invoke(app, ["book", "render-plans", "--root", str(fixture_book.root)])
    assert target.read_text() == first
