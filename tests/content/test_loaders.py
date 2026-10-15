"""Loaders turn YAML, TOML and Markdown front matter into typed entities with line numbers."""

from datetime import date
from pathlib import Path

import pytest

from db_ml_associate.content.loaders import (
    ContentError,
    load_book,
    load_chapter,
    load_objectives,
    load_questions,
    parse_markdown,
    slugify,
)
from helpers.fixture_book import FixtureBook, build_fixture_book


@pytest.fixture
def fixture_book(tmp_path: Path) -> FixtureBook:
    return build_fixture_book(tmp_path, date(2026, 10, 14))


def test_load_objectives(fixture_book: FixtureBook) -> None:
    guide, objectives = load_objectives(fixture_book.book / "exam" / "objectives.yaml")
    assert guide.version == date(2025, 3, 1)
    assert guide.questions == 48
    assert len(objectives) == 48
    assert objectives[0].id == "S1.01"
    assert objectives[0].line > 0


def test_load_questions_records_file_and_line(fixture_book: FixtureBook) -> None:
    questions = load_questions(fixture_book.book / "questions" / "ch01.yaml")
    assert len(questions) == 48
    assert questions[0].path.endswith("ch01.yaml")
    assert questions[1].line > questions[0].line > 0
    assert questions[0].answer == ("A",)


def test_malformed_yaml_reports_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("- id: Q-01-001\n  options: {A: 1\n")
    with pytest.raises(ContentError) as error:
        load_questions(path)
    assert error.value.issue.path.endswith("broken.yaml")
    assert error.value.issue.line >= 1


def test_load_chapter_front_matter_and_sections(fixture_book: FixtureBook) -> None:
    chapter = load_chapter(fixture_book.chapter)
    assert chapter.number == 1
    assert chapter.title == "The platform"
    assert len(chapter.objectives) == 48
    assert chapter.prerequisites == ()
    assert chapter.tutorial is not None
    assert chapter.tutorial.notebook == "notebooks/ch01_platform.py"
    assert chapter.tutorial.steps[0].id == "everything"
    assert chapter.learning_objectives == (
        "Understand one.",
        "Understand two.",
        "Understand three.",
    )
    assert [section.anchor for section in chapter.sections] == ["everything-in-one-section"]
    assert len(chapter.sections[0].objectives) == 48
    assert chapter.sections[0].line > 0


def test_section_tag_with_unknown_objective_is_reported(
    fixture_book: FixtureBook,
) -> None:
    fixture_book.edit_chapter("<!-- objectives: S1.01,", "<!-- objectives: S9.99, S1.01,")
    book = load_book(fixture_book.root)
    assert any("S9.99" in str(issue) and "01-platform.md" in str(issue) for issue in book.issues)


def test_load_book_collects_everything(fixture_book: FixtureBook) -> None:
    book = load_book(fixture_book.root)
    assert book.guide is not None
    assert len(book.objectives) == 48
    assert [chapter.number for chapter in book.chapters] == [1]
    assert len(book.questions) == 48
    assert [reference.id for reference in book.references] == ["ml-overview"]
    assert len(book.platform_changes) == 7
    assert book.runtime is not None
    assert book.runtime.databricks_runtime == "18 LTS ML"
    assert book.issues == []


def test_introduction_and_appendices_are_front_matter(
    fixture_book: FixtureBook,
) -> None:
    (fixture_book.book / "introduction.md").write_text("# Introduction\n")
    (fixture_book.book / "appendix-b-cheat-sheet.md").write_text("# Cheat sheet\n")
    book = load_book(fixture_book.root)
    assert [chapter.number for chapter in book.chapters] == [1]
    assert sorted(page.name for page in book.front_matter) == [
        "appendix-b-cheat-sheet.md",
        "introduction.md",
    ]


@pytest.mark.parametrize(
    "heading,anchor",
    [
        (
            "Cross-validation vs. train/validation split",
            "cross-validation-vs-trainvalidation-split",
        ),
        ("Hyperopt `fmin`", "hyperopt-fmin"),
        ("What's new?", "whats-new"),
        ("Section 1: Databricks ML", "section-1-databricks-ml"),
    ],
)
def test_slugify_matches_github_anchors(heading: str, anchor: str) -> None:
    assert slugify(heading) == anchor


def test_explicit_html_anchors_are_collected() -> None:
    document = parse_markdown(
        '# Title\n\n<a id="counting-models"></a>\n\n```\n<a id="in-code"></a>\n```\n',
        "x.md",
    )
    assert document.anchors == {"title", "counting-models"}
