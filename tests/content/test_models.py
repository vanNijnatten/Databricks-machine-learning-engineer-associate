"""Validation rules of the content entities (data-model.md)."""

from datetime import date

import pytest

from db_ml_associate.content.models import (
    Chapter,
    ExamObjective,
    PracticeExam,
    Question,
    Reference,
    Tutorial,
    objective_set_problems,
    practice_exam_problems,
)


def make_objectives(counts: dict[int, int]) -> list[ExamObjective]:
    return [
        ExamObjective(
            id=f"S{section}.{n:02d}",
            section=section,
            kind="conceptual",
            text=f"objective {section}.{n}",
        )
        for section, total in counts.items()
        for n in range(1, total + 1)
    ]


def make_question(**overrides: object) -> Question:
    fields: dict[str, object] = {
        "id": "Q-10-001",
        "objective": "S3.10",
        "pool": "review",
        "type": "single",
        "difficulty": "medium",
        "stem": "How many models are trained?",
        "options": {"A": "12", "B": "36", "C": "7", "D": "None of the above"},
        "answer": ("B",),
        "explanation": "12 combinations x 3 folds.",
        "distractors": {
            "A": "Ignores folds.",
            "C": "Adds instead of multiplying.",
            "D": "Counts the refit.",
        },
        "chapter_ref": "10-validation-and-tuning.md#counting-models",
    }
    fields.update(overrides)
    return Question(**fields)  # type: ignore[arg-type]


class TestExamObjective:
    @pytest.mark.parametrize("objective_id", ["S1.01", "S4.06", "S3.15"])
    def test_valid_ids(self, objective_id: str) -> None:
        section = int(objective_id[1])
        objective = ExamObjective(id=objective_id, section=section, kind="practical", text="x")
        assert objective.problems() == []

    @pytest.mark.parametrize("objective_id", ["S5.01", "S1.1", "1.01", "S1.001"])
    def test_invalid_ids(self, objective_id: str) -> None:
        objective = ExamObjective(id=objective_id, section=1, kind="practical", text="x")
        assert any("id" in problem for problem in objective.problems())

    def test_section_must_match_id(self) -> None:
        objective = ExamObjective(id="S2.01", section=3, kind="practical", text="x")
        assert any("section" in problem for problem in objective.problems())

    def test_kind_must_be_known(self) -> None:
        objective = ExamObjective(id="S2.01", section=2, kind="other", text="x")
        assert any("kind" in problem for problem in objective.problems())


class TestObjectiveSet:
    def test_exact_exam_counts_pass(self) -> None:
        assert objective_set_problems(make_objectives({1: 18, 2: 9, 3: 15, 4: 6})) == []

    def test_wrong_total_fails(self) -> None:
        problems = objective_set_problems(make_objectives({1: 18, 2: 9, 3: 15, 4: 5}))
        assert any("48" in problem for problem in problems)

    def test_wrong_section_split_fails(self) -> None:
        problems = objective_set_problems(make_objectives({1: 19, 2: 8, 3: 15, 4: 6}))
        assert any("section 1" in problem for problem in problems)

    def test_duplicate_ids_fail(self) -> None:
        objectives = make_objectives({1: 18, 2: 9, 3: 15, 4: 6})
        objectives[1] = objectives[0]
        assert any("duplicate" in problem for problem in objective_set_problems(objectives))


class TestQuestion:
    def test_valid_question(self) -> None:
        assert make_question().problems() == []

    def test_single_needs_exactly_one_answer(self) -> None:
        question = make_question(answer=("A", "B"), distractors={"C": "x", "D": "y"})
        assert any("single" in problem for problem in question.problems())

    def test_multiple_needs_at_least_two_answers(self) -> None:
        question = make_question(type="multiple")
        assert any("multiple" in problem for problem in question.problems())

    @pytest.mark.parametrize(
        "options",
        [
            {"A": "1", "B": "2", "C": "3"},
            {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5", "F": "6"},
        ],
    )
    def test_four_or_five_options(self, options: dict[str, str]) -> None:
        distractors = {letter: "why" for letter in options if letter != "B"}
        question = make_question(options=options, distractors=distractors)
        assert any("options" in problem for problem in question.problems())

    def test_every_wrong_option_needs_a_distractor_explanation(self) -> None:
        question = make_question(distractors={"A": "Ignores folds."})
        assert any("distractor" in problem for problem in question.problems())

    def test_answer_must_be_an_option(self) -> None:
        question = make_question(answer=("E",))
        assert any("answer" in problem for problem in question.problems())

    def test_unknown_objective_is_reported(self) -> None:
        question = make_question(objective="S9.99")
        assert any("objective" in problem for problem in question.problems(known_objectives={"S3.10"}))

    @pytest.mark.parametrize("field,value", [("pool", "quiz"), ("difficulty", "trivial"), ("type", "open")])
    def test_enums(self, field: str, value: str) -> None:
        question = make_question(**{field: value})
        assert any(field in problem for problem in question.problems())


class TestPracticeExam:
    def _bank(self, exam_id: str) -> dict[str, Question]:
        counts = {1: 18, 2: 9, 3: 15, 4: 6}
        bank: dict[str, Question] = {}
        number = 0
        for section, total in counts.items():
            for n in range(1, total + 1):
                number += 1
                question_id = f"Q-{exam_id}-{number:03d}"
                bank[question_id] = make_question(id=question_id, objective=f"S{section}.{n:02d}", pool="practice")
        return bank

    def test_valid_exam(self) -> None:
        bank = self._bank("PE1")
        exam = PracticeExam(id="PE1", minutes=90, questions=tuple(bank))
        assert practice_exam_problems([exam], bank) == []

    def test_exactly_48_questions(self) -> None:
        bank = self._bank("PE1")
        exam = PracticeExam(id="PE1", minutes=90, questions=tuple(bank)[:47])
        assert any("48" in problem for problem in practice_exam_problems([exam], bank))

    def test_only_practice_pool(self) -> None:
        bank = self._bank("PE1")
        first = next(iter(bank))
        bank[first] = make_question(id=first, objective="S1.01", pool="review")
        exam = PracticeExam(id="PE1", minutes=90, questions=tuple(bank))
        assert any("practice" in problem for problem in practice_exam_problems([exam], bank))

    def test_no_reuse_across_exams(self) -> None:
        bank = self._bank("PE1")
        first = PracticeExam(id="PE1", minutes=90, questions=tuple(bank))
        second = PracticeExam(id="PE2", minutes=90, questions=tuple(bank))
        assert any("reused" in problem for problem in practice_exam_problems([first, second], bank))

    def test_section_counts(self) -> None:
        bank = self._bank("PE1")
        ids = list(bank)
        # Replace a Section 4 question by a second Section 1 question.
        bank[ids[-1]] = make_question(id=ids[-1], objective="S1.01", pool="practice")
        exam = PracticeExam(id="PE1", minutes=90, questions=tuple(ids))
        assert any("section" in problem for problem in practice_exam_problems([exam], bank))


class TestTutorial:
    def test_cleanup_required_for_paid_features(self) -> None:
        for feature in ("model_serving", "online_store"):
            tutorial = Tutorial(
                chapter=13,
                notebook="notebooks/ch13_serving.py",
                minutes=60,
                requires=(feature,),
                free_edition="partial",
                cleanup=False,
            )
            assert any("cleanup" in problem for problem in tutorial.problems())

    def test_valid_tutorial(self) -> None:
        tutorial = Tutorial(
            chapter=13,
            notebook="notebooks/ch13_serving.py",
            minutes=60,
            requires=("model_serving",),
            free_edition="partial",
            cleanup=True,
        )
        assert tutorial.problems() == []

    def test_unknown_requirement(self) -> None:
        tutorial = Tutorial(
            chapter=6,
            notebook="notebooks/ch06_exploring.py",
            minutes=30,
            requires=("gpu",),
            free_edition="full",
        )
        assert any("requires" in problem for problem in tutorial.problems())


class TestChapter:
    def test_number_range_and_objectives(self) -> None:
        chapter = Chapter(number=14, title="Extra", objectives=(), path="book/14-extra.md")
        problems = chapter.problems()
        assert any("number" in problem for problem in problems)
        assert any("objective" in problem for problem in problems)


class TestReference:
    def _reference(self, verified: date) -> Reference:
        return Reference(
            id="automl",
            url="https://docs.databricks.com/aws/en/machine-learning/automl/",
            title="AutoML",
            claim="AutoML is not built in from 18.0 ML.",
            objectives=("S1.03",),
            chapters=(2,),
            last_verified=verified,
        )

    def test_recent_reference_is_not_stale(self) -> None:
        assert not self._reference(date(2026, 10, 1)).is_stale(today=date(2026, 10, 14))

    def test_reference_older_than_90_days_is_stale(self) -> None:
        assert self._reference(date(2026, 7, 1)).is_stale(today=date(2026, 10, 14))

    def test_discrepancy_needs_note(self) -> None:
        reference = Reference(
            id="x",
            url="https://docs.databricks.com/aws/en/x",
            title="X",
            claim="c",
            status="discrepancy",
        )
        assert any("note" in problem for problem in reference.problems())
