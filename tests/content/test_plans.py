"""Study-plan time computation and validation rules (data-model.md, StudyPlan / StudyPlanDay)."""

from datetime import date
from pathlib import Path

import pytest

from db_ml_associate.content.loaders import load_book
from db_ml_associate.content.models import PlanItem, StudyPlan, StudyPlanDay
from db_ml_associate.content.plans import (
    APPENDIX_MINUTES,
    EXAM_MINUTES,
    INTRODUCTION_MINUTES,
    QUESTION_MINUTES,
    REVIEW_MINUTES,
    day_minutes,
    item_minutes,
    plan_problems,
)
from helpers.fixture_book import FixtureBook, build_fixture_book

TUTORIAL = "notebooks/ch01_platform.py"


@pytest.fixture
def fixture_book(tmp_path: Path) -> FixtureBook:
    return build_fixture_book(tmp_path, date.today())


def day(number: int, *items: PlanItem) -> StudyPlanDay:
    return StudyPlanDay(day=number, goals=("goal",), items=items, self_check="check")


def chapter(number: int = 1, optional: bool = False) -> PlanItem:
    return PlanItem("chapter", str(number), optional)


def tutorial(optional: bool = False) -> PlanItem:
    return PlanItem("tutorial", TUTORIAL, optional)


def exam(exam_id: str) -> PlanItem:
    return PlanItem("exam", exam_id)


REVIEW = PlanItem("review", "weak areas")


def valid_plan(track: str = "fast") -> StudyPlan:
    """A complete miniature plan for the one-chapter fixture book."""
    return StudyPlan(
        track=track,
        days=(
            day(1, PlanItem("introduction", "True"), chapter()),  # 20 + 60 + 48 × 2 = 176
            day(2, tutorial(), REVIEW, PlanItem("appendix", "A")),  # 30 + 45 + 45 = 120
            day(3, exam("PE1"), REVIEW),  # 135 + 45 = 180
            day(4, exam("PE2"), REVIEW),
        ),
    )


class TestMinutes:
    def test_chapter_counts_reading_and_review_questions(self, fixture_book: FixtureBook) -> None:
        book = load_book(fixture_book.root)
        assert item_minutes(chapter(), book) == 60 + 48 * QUESTION_MINUTES

    def test_tutorial_uses_tutorial_minutes(self, fixture_book: FixtureBook) -> None:
        assert item_minutes(tutorial(), load_book(fixture_book.root)) == 30

    @pytest.mark.parametrize(
        ("item", "minutes"),
        [
            (PlanItem("introduction", "True"), INTRODUCTION_MINUTES),
            (exam("PE1"), EXAM_MINUTES),
            (REVIEW, REVIEW_MINUTES),
            (PlanItem("appendix", "B"), APPENDIX_MINUTES),
        ],
    )
    def test_fixed_estimates(self, fixture_book: FixtureBook, item: PlanItem, minutes: int) -> None:
        assert item_minutes(item, load_book(fixture_book.root)) == minutes

    def test_exam_time_is_exam_plus_answer_review(self) -> None:
        assert EXAM_MINUTES == 90 + 45

    def test_optional_items_are_not_counted(self, fixture_book: FixtureBook) -> None:
        book = load_book(fixture_book.root)
        assert day_minutes(day(1, chapter(), tutorial(optional=True)), book) == 60 + 48 * QUESTION_MINUTES


class TestRules:
    def test_valid_plan_has_no_problems(self, fixture_book: FixtureBook) -> None:
        assert plan_problems(valid_plan(), load_book(fixture_book.root)) == []

    def test_every_chapter_must_be_scheduled(self, fixture_book: FixtureBook) -> None:
        plan = valid_plan()
        days = (day(1, PlanItem("introduction", "True"), REVIEW, REVIEW, REVIEW), *plan.days[1:])
        problems = plan_problems(StudyPlan("fast", days), load_book(fixture_book.root))
        assert any("chapter 1 is not scheduled" in problem for problem in problems)

    def test_every_tutorial_must_be_scheduled(self, fixture_book: FixtureBook) -> None:
        plan = valid_plan()
        days = (plan.days[0], day(2, REVIEW, PlanItem("appendix", "A")), *plan.days[2:])
        problems = plan_problems(StudyPlan("fast", days), load_book(fixture_book.root))
        assert any("tutorial" in problem and "not scheduled" in problem for problem in problems)

    def test_both_practice_exams_must_be_scheduled(self, fixture_book: FixtureBook) -> None:
        plan = valid_plan()
        days = (*plan.days[:3], day(4, REVIEW, REVIEW))
        problems = plan_problems(StudyPlan("fast", days), load_book(fixture_book.root))
        assert any("PE2" in problem for problem in problems)

    def test_at_least_two_days_for_exams_and_review(self, fixture_book: FixtureBook) -> None:
        plan = valid_plan()
        days = (*plan.days[:2], day(3, exam("PE1"), REVIEW), day(4, chapter(), exam("PE2")))
        problems = plan_problems(StudyPlan("fast", days), load_book(fixture_book.root))
        assert any("at least 2 days" in problem for problem in problems)

    @pytest.mark.parametrize("items", [(REVIEW,), (exam("PE1"), exam("PE2"))], ids=["too-short", "too-long"])
    def test_daily_minutes_between_90_and_180(self, fixture_book: FixtureBook, items: tuple[PlanItem, ...]) -> None:
        plan = valid_plan()
        days = (*plan.days, day(5, *items))
        problems = plan_problems(StudyPlan("fast", days), load_book(fixture_book.root))
        assert any("day 5" in problem and "90-180" in problem for problem in problems)

    @pytest.mark.parametrize(("track", "limit"), [("fast", 14), ("standard", 21)])
    def test_track_length_limit(self, fixture_book: FixtureBook, track: str, limit: int) -> None:
        plan = valid_plan(track)
        extra = tuple(day(number, REVIEW, PlanItem("appendix", "A")) for number in range(5, limit + 2))
        problems = plan_problems(StudyPlan(track, (*plan.days, *extra)), load_book(fixture_book.root))
        assert any(f"at most {limit} days" in problem for problem in problems)

    def test_days_must_be_contiguous(self, fixture_book: FixtureBook) -> None:
        plan = valid_plan()
        days = (*plan.days[:3], day(7, exam("PE2"), REVIEW))
        problems = plan_problems(StudyPlan("fast", days), load_book(fixture_book.root))
        assert any("contiguous" in problem for problem in problems)

    def test_optional_tutorials_only_on_fast_track_for_skimmable_chapters(self, fixture_book: FixtureBook) -> None:
        plan = valid_plan("fast")
        days = (plan.days[0], day(2, tutorial(optional=True), REVIEW, PlanItem("appendix", "A")), *plan.days[2:])
        problems = plan_problems(StudyPlan("fast", days), load_book(fixture_book.root))
        assert any("may not be optional" in problem for problem in problems)

    def test_unknown_targets_are_reported(self, fixture_book: FixtureBook) -> None:
        plan = valid_plan()
        days = (*plan.days, day(5, chapter(7), PlanItem("tutorial", "notebooks/nope.py"), exam("PE9")))
        problems = plan_problems(StudyPlan("fast", days), load_book(fixture_book.root))
        assert any("unknown chapter 7" in problem for problem in problems)
        assert any("unknown tutorial notebooks/nope.py" in problem for problem in problems)
        assert any("unknown practice exam PE9" in problem for problem in problems)

    def test_unknown_track(self, fixture_book: FixtureBook) -> None:
        problems = plan_problems(StudyPlan("slow", valid_plan().days), load_book(fixture_book.root))
        assert any("track" in problem for problem in problems)


class TestBookCheckIntegration:
    def test_plan_errors_reported_by_book_check(self, fixture_book: FixtureBook) -> None:
        from db_ml_associate.cli.book_check import run_checks

        fixture_book.write_yaml(
            "plans/fast-14.yaml",
            {"track": "fast", "days": [{"day": 1, "goals": ["g"], "items": [{"review": "x"}], "self_check": "s"}]},
        )
        messages = [str(issue) for issue in run_checks(fixture_book.root) if issue.severity == "error"]
        assert any("plans/fast-14.yaml" in message and "90-180" in message for message in messages)

    def test_missing_plans_are_open_work(self, fixture_book: FixtureBook) -> None:
        from db_ml_associate.cli.book_check import run_checks

        warnings = [str(issue) for issue in run_checks(fixture_book.root) if issue.severity == "warning"]
        assert any("no study plans yet" in message for message in warnings)
