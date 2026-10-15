"""Review questions from book/questions/chNN.yaml are rendered into the chapter (contracts/content-formats.md)."""

from pathlib import Path

import yaml

from db_ml_associate.cli.questions import (
    END_MARKER,
    render_review_questions,
    sync_questions,
)
from db_ml_associate.cli.snippets import sync_book
from db_ml_associate.content.models import Question


def make_question(number: int, answers: tuple[str, ...] = ("B",)) -> Question:
    return Question(
        id=f"Q-10-{number:03d}",
        objective="S3.10",
        pool="review",
        type="single" if len(answers) == 1 else "multiple",
        difficulty="easy",
        stem="How many models are trained?",
        options={"A": "12", "B": "36", "C": "7", "D": "None"},
        answer=answers,
        explanation="12 combinations x 3 folds.",
        distractors={"A": "Ignores folds.", "C": "Adds.", "D": "Counts the refit."},
        chapter_ref="10-validation-and-tuning.md#counting-models",
    )


def test_render_shows_options_and_hides_answer_in_details() -> None:
    text = render_review_questions([make_question(1)])
    assert "### Question 1" in text
    assert "- **A.** 12" in text
    assert "*Select one answer.*" in text
    assert "<details>" in text and "**Correct: B**" in text
    assert "- **A**: Ignores folds." in text
    assert "[Revisit](10-validation-and-tuning.md#counting-models)" in text
    assert text.index("<details>") > text.index("- **D.** None")


def test_multiple_answer_instruction() -> None:
    question = make_question(1, answers=("A", "B"))
    text = render_review_questions([question])
    assert "*Select 2 answers.*" in text
    assert "**Correct: A, B**" in text


def test_sync_inserts_and_is_idempotent() -> None:
    chapter = "# Ch\n\n## Review questions\n\n<!-- questions: ch10 -->\n"
    first = sync_questions(chapter, {"ch10": [make_question(1), make_question(2)]})
    assert first.stale
    assert "### Question 2" in first.text
    assert first.text.rstrip().endswith(END_MARKER)
    second = sync_questions(first.text, {"ch10": [make_question(1), make_question(2)]})
    assert not second.stale
    assert second.text == first.text


def test_sync_detects_changed_questions() -> None:
    chapter = sync_questions("<!-- questions: ch10 -->\n", {"ch10": [make_question(1)]}).text
    changed = sync_questions(chapter, {"ch10": [make_question(1), make_question(2)]})
    assert changed.stale


def test_sync_book_renders_questions(tmp_path: Path) -> None:
    (tmp_path / "book" / "questions").mkdir(parents=True)
    question = {
        "id": "Q-10-001",
        "objective": "S3.10",
        "pool": "review",
        "type": "single",
        "difficulty": "easy",
        "stem": "How many?",
        "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "answer": ["B"],
        "explanation": "Two.",
        "distractors": {"A": "x", "C": "y", "D": "z"},
        "chapter_ref": "10-x.md",
    }
    (tmp_path / "book" / "questions" / "ch10.yaml").write_text(yaml.safe_dump([question]))
    chapter = tmp_path / "book" / "10-x.md"
    chapter.write_text("## Review questions\n\n<!-- questions: ch10 -->\n")
    assert len(sync_book(tmp_path, check=True)) == 1  # stale before rendering
    assert sync_book(tmp_path, check=False) == []
    assert "### Question 1" in chapter.read_text()
    assert sync_book(tmp_path, check=True) == []
