"""Practice exams: scoring, readiness, option shuffling, section filter, rendering and the CLI (FR-017-FR-019)."""

import json
import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from db_ml_associate.cli import app
from db_ml_associate.cli.exam import (
    READINESS_THRESHOLD,
    exam_questions,
    render_exam,
    score_attempt,
    shuffle_options,
)
from db_ml_associate.content.loaders import load_book
from db_ml_associate.content.models import Question
from helpers.fixture_book import FixtureBook, objective_ids, question

SECTION_COUNTS = {1: 18, 2: 9, 3: 15, 4: 6}


def _practice_question(number: int, objective: str) -> dict[str, object]:
    raw = question(f"Q-PE1-{number:03d}", objective, pool="practice")
    if number % 4 == 0:  # every fourth question has two answers
        raw.update(type="multiple", answer=["A", "C"], distractors={"B": "No.", "D": "No."})
    return raw


@pytest.fixture
def exam_book(fixture_book: FixtureBook) -> FixtureBook:
    """The fixture book plus a 48-question practice exam PE1 (one question per objective)."""
    ids = objective_ids()
    fixture_book.write_yaml(
        "questions/practice-pe1.yaml",
        [_practice_question(n, objective) for n, objective in enumerate(ids, start=1)],
    )
    fixture_book.write_yaml(
        "exams/PE1.yaml",
        {"id": "PE1", "minutes": 90, "questions": [f"Q-PE1-{n:03d}" for n in range(1, len(ids) + 1)]},
    )
    return fixture_book


def _questions(book_fixture: FixtureBook, section: int | None = None) -> list[Question]:
    return exam_questions(load_book(book_fixture.root), "PE1", section)


def _all_correct(questions: list[Question]) -> dict[str, set[str]]:
    return {q.id: set(q.answer) for q in questions}


class TestScoring:
    def test_all_correct_is_ready_everywhere(self, exam_book: FixtureBook) -> None:
        questions = _questions(exam_book)
        result = score_attempt(questions, _all_correct(questions))
        assert result.correct == result.total == 48
        assert {s.section: (s.correct, s.total) for s in result.sections} == {
            section: (count, count) for section, count in SECTION_COUNTS.items()
        }
        assert all(s.ready for s in result.sections)
        assert result.missed == []

    def test_multiple_choice_is_all_or_nothing(self, exam_book: FixtureBook) -> None:
        questions = _questions(exam_book)
        multiple = next(q for q in questions if q.type == "multiple")
        answers = _all_correct(questions)
        answers[multiple.id] = {"A"}  # one of two correct letters
        result = score_attempt(questions, answers)
        assert result.correct == 47
        assert [q.id for q in result.missed] == [multiple.id]

    def test_extra_wrong_letter_fails_a_multiple_question(self, exam_book: FixtureBook) -> None:
        questions = _questions(exam_book)
        multiple = next(q for q in questions if q.type == "multiple")
        answers = _all_correct(questions)
        answers[multiple.id] = {"A", "B", "C"}
        assert multiple.id in [q.id for q in score_attempt(questions, answers).missed]

    def test_readiness_threshold_is_80_percent_per_section(self, exam_book: FixtureBook) -> None:
        assert READINESS_THRESHOLD == 0.8
        questions = _questions(exam_book)
        answers = _all_correct(questions)
        section_four = [q for q in questions if q.section == 4]
        answers[section_four[0].id] = {"D"}  # 5 / 6 = 83 % → ready
        result = score_attempt(questions, answers)
        assert next(s for s in result.sections if s.section == 4).ready
        answers[section_four[1].id] = {"D"}  # 4 / 6 = 67 % → not ready
        result = score_attempt(questions, answers)
        assert not next(s for s in result.sections if s.section == 4).ready
        assert next(s for s in result.sections if s.section == 1).ready

    def test_unanswered_questions_count_as_missed_with_chapter_ref(self, exam_book: FixtureBook) -> None:
        questions = _questions(exam_book)
        result = score_attempt(questions, {})
        assert result.correct == 0
        assert len(result.missed) == 48
        assert result.missed[0].chapter_ref == "01-platform.md#everything-in-one-section"


class TestShuffling:
    def test_shuffle_is_deterministic_for_a_seed(self, exam_book: FixtureBook) -> None:
        q = _questions(exam_book)[0]
        assert shuffle_options(q, random.Random(7)) == shuffle_options(q, random.Random(7))

    def test_shuffle_preserves_the_answer_mapping(self, exam_book: FixtureBook) -> None:
        for q in _questions(exam_book):
            shown = shuffle_options(q, random.Random(3))
            assert [letter for letter, _, _ in shown] == list("ABCDE"[: len(q.options)])
            assert sorted(original for _, original, _ in shown) == sorted(q.options)
            for _, original, text in shown:
                assert q.options[original] == text

    def test_different_seeds_give_different_orders(self, exam_book: FixtureBook) -> None:
        q = _questions(exam_book)[0]
        orders = {tuple(o for _, o, _ in shuffle_options(q, random.Random(seed))) for seed in range(20)}
        assert len(orders) > 1


def test_section_filter(exam_book: FixtureBook) -> None:
    questions = _questions(exam_book, section=2)
    assert len(questions) == 9
    assert {q.section for q in questions} == {2}


def test_unknown_exam_raises(exam_book: FixtureBook) -> None:
    with pytest.raises(KeyError):
        exam_questions(load_book(exam_book.root), "PE9")


class TestRender:
    def test_questions_only(self, exam_book: FixtureBook) -> None:
        text = render_exam(load_book(exam_book.root), "PE1", answers=False)
        assert text.startswith("## Practice exam PE1")
        assert text.count("### Question") == 48
        assert "First is right." not in text
        assert "Select 2 answers" in text

    def test_answer_key_has_explanations_distractors_and_links(self, exam_book: FixtureBook) -> None:
        text = render_exam(load_book(exam_book.root), "PE1", answers=True)
        assert "Answer key" in text
        assert "First is right." in text
        assert "**B**: No." in text
        assert "(01-platform.md#everything-in-one-section)" in text


def _first_letters(book_fixture: FixtureBook, section: int) -> str:
    """Input that always picks the first option(s): as many letters as each question needs."""
    questions = exam_questions(load_book(book_fixture.root), "PE1", section)
    return "".join(("A" if q.type == "single" else "A,B") + "\n" for q in questions)


class TestCli:
    def test_take_scores_simulated_answers(self, exam_book: FixtureBook) -> None:
        # With the options shuffled, always picking the first options gets some right and some wrong.
        answers = _first_letters(exam_book, 2)
        result = CliRunner().invoke(
            app,
            ["exam", "take", "PE1", "--root", str(exam_book.root), "--no-timer", "--section", "2", "--seed", "1"],
            input=answers,
        )
        assert result.exit_code == 0, result.output
        assert "Question 9 of 9" in result.output
        assert "Section 2" in result.output
        assert "Overall" in result.output

    def test_take_all_correct_with_answers_from_shuffle(self, exam_book: FixtureBook) -> None:
        book = load_book(exam_book.root)
        rng = random.Random(5)
        lines = []
        for q in exam_questions(book, "PE1", 4):
            shown = shuffle_options(q, rng)
            lines.append(",".join(letter for letter, original, _ in shown if original in q.answer))
        result = CliRunner().invoke(
            app,
            ["exam", "take", "PE1", "--root", str(exam_book.root), "--no-timer", "--section", "4", "--seed", "5"],
            input="\n".join(lines) + "\n",
        )
        assert result.exit_code == 0, result.output
        assert "6/6" in result.output
        assert "ready" in result.output.lower()

    def test_invalid_input_is_asked_again(self, exam_book: FixtureBook) -> None:
        result = CliRunner().invoke(
            app,
            ["exam", "take", "PE1", "--root", str(exam_book.root), "--no-timer", "--section", "4", "--seed", "5"],
            input="Z\n" + _first_letters(exam_book, 4),
        )
        assert result.exit_code == 0, result.output
        assert "Answer with" in result.output

    def test_save_writes_attempt_json(
        self, exam_book: FixtureBook, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DB_ML_ASSOCIATE_HOME", str(tmp_path / "home"))
        result = CliRunner().invoke(
            app,
            ["exam", "take", "PE1", "--root", str(exam_book.root), "--no-timer", "--section", "4", "--save"],
            input=_first_letters(exam_book, 4),
        )
        assert result.exit_code == 0, result.output
        (saved,) = (tmp_path / "home" / "attempts").glob("*.json")
        data = json.loads(saved.read_text())
        assert data["exam"] == "PE1"
        assert data["total"] == 6
        assert len(data["answers"]) == 6

    def test_render_cli(self, exam_book: FixtureBook) -> None:
        result = CliRunner().invoke(app, ["exam", "render", "PE1", "--root", str(exam_book.root), "--answers"])
        assert result.exit_code == 0, result.output
        assert "Answer key" in result.output

    def test_unknown_exam_is_a_usage_error(self, exam_book: FixtureBook) -> None:
        result = CliRunner().invoke(app, ["exam", "render", "PE9", "--root", str(exam_book.root)])
        assert result.exit_code == 2


def test_appendix_d_has_worksheet_exams_and_answer_keys(exam_book: FixtureBook) -> None:
    from db_ml_associate.cli.exam import render_appendix_d

    text = render_appendix_d(load_book(exam_book.root))
    assert text.startswith("# Appendix D")
    assert "## Score worksheet" in text
    assert "| 1. Databricks Machine Learning | 18 |" in text
    assert text.index("## Practice exam PE1\n") < text.index("## Practice exam PE1 — Answer key")


def test_render_appendix_d_cli(exam_book: FixtureBook) -> None:
    result = CliRunner().invoke(app, ["book", "render-appendix-d", "--root", str(exam_book.root)])
    assert result.exit_code == 0, result.output
    assert (exam_book.book / "appendix-d-practice-exams.md").read_text().startswith("# Appendix D")
