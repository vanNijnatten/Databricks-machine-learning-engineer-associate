"""Study-plan time estimates and validation (data-model.md, StudyPlan / StudyPlanDay; FR-016, SC-002).

Plan files list activities only; the minutes of every day are computed here from the chapters'
``reading_minutes``, their review questions and their tutorials' ``minutes``, plus fixed estimates for
the introduction, practice exams, weak-area reviews and appendix revision.
"""

from collections import Counter

from db_ml_associate.content.loaders import Book
from db_ml_associate.content.models import (
    LAST_CHAPTER,
    PLAN_ITEM_KINDS,
    PLAN_TRACKS,
    Chapter,
    PlanItem,
    StudyPlan,
    StudyPlanDay,
)

QUESTION_MINUTES = 2  # per review question
INTRODUCTION_MINUTES = 20
EXAM_MINUTES = 90 + 45  # the timed exam plus reviewing the answer key
REVIEW_MINUTES = 45  # a weak-area review block
APPENDIX_MINUTES = 45  # revising with an appendix (cheat sheet, exam vs platform)
MIN_DAY_MINUTES = 90
MAX_DAY_MINUTES = 180
PRACTICE_EXAMS = ("PE1", "PE2")
MIN_EXAM_AND_REVIEW_DAYS = 2
EXAM_AND_REVIEW_KINDS = frozenset({"exam", "review", "appendix"})
SKIMMABLE_CHAPTERS = frozenset({8, 9, 11})  # their tutorials may be optional on the fast track


def _review_questions(book: Book, chapter: int) -> int:
    prefix = f"Q-{chapter:02d}-"
    return sum(1 for question in book.questions if question.pool == "review" and question.id.startswith(prefix))


def _tutorial_chapter(book: Book, notebook: str) -> Chapter | None:
    return next(
        (chapter for chapter in book.chapters if chapter.tutorial and chapter.tutorial.notebook == notebook),
        None,
    )


def item_minutes(item: PlanItem, book: Book) -> int:
    """Estimated minutes of one plan activity (0 for an unknown chapter or tutorial)."""
    if item.kind == "chapter":
        chapter = book.chapter(int(item.target)) if item.target.isdigit() else None
        if chapter is None:
            return 0
        return chapter.reading_minutes + QUESTION_MINUTES * _review_questions(book, chapter.number)
    if item.kind == "tutorial":
        chapter = _tutorial_chapter(book, item.target)
        return chapter.tutorial.minutes if chapter is not None and chapter.tutorial is not None else 0
    fixed = {
        "introduction": INTRODUCTION_MINUTES,
        "exam": EXAM_MINUTES,
        "review": REVIEW_MINUTES,
        "appendix": APPENDIX_MINUTES,
    }
    return fixed.get(item.kind, 0)


def day_minutes(day: StudyPlanDay, book: Book) -> int:
    """Minutes of the day's required activities; optional activities are extra."""
    return sum(item_minutes(item, book) for item in day.items if not item.optional)


def _item_problems(plan: StudyPlan, day: StudyPlanDay, item: PlanItem, book: Book) -> list[str]:
    label = f"day {day.day}"
    if item.kind not in PLAN_ITEM_KINDS:
        return [f"{label}: unknown activity {item.kind!r}"]
    if item.kind == "chapter" and (not item.target.isdigit() or book.chapter(int(item.target)) is None):
        return [f"{label}: unknown chapter {item.target}"]
    if item.kind == "tutorial":
        owner = _tutorial_chapter(book, item.target)
        if owner is None:
            return [f"{label}: unknown tutorial {item.target}"]
        if item.optional and (plan.track != "fast" or owner.number not in SKIMMABLE_CHAPTERS):
            return [f"{label}: the tutorial of chapter {owner.number} may not be optional on the {plan.track} track"]
    if item.kind == "exam" and item.target not in PRACTICE_EXAMS:
        return [f"{label}: unknown practice exam {item.target}"]
    return []


def _schedule_problems(plan: StudyPlan, book: Book) -> list[str]:
    items = [item for day in plan.days for item in day.items]
    scheduled = Counter((item.kind, item.target) for item in items)
    found = [
        f"chapter {chapter.number} is not scheduled"
        for chapter in book.chapters
        if 1 <= chapter.number <= LAST_CHAPTER and not scheduled[("chapter", str(chapter.number))]
    ]
    found.extend(
        f"tutorial {chapter.tutorial.notebook} (chapter {chapter.number}) is not scheduled"
        for chapter in book.chapters
        if chapter.tutorial is not None and not scheduled[("tutorial", chapter.tutorial.notebook)]
    )
    found.extend(f"practice exam {exam} is not scheduled" for exam in PRACTICE_EXAMS if not scheduled[("exam", exam)])
    review_days = sum(
        1 for day in plan.days if day.items and {item.kind for item in day.items} <= EXAM_AND_REVIEW_KINDS
    )
    if review_days < MIN_EXAM_AND_REVIEW_DAYS:
        found.append(f"plan needs at least {MIN_EXAM_AND_REVIEW_DAYS} days for practice exams and weak-area review")
    return found


def plan_problems(plan: StudyPlan, book: Book) -> list[str]:
    """Return every rule violation of a study plan."""
    if plan.track not in PLAN_TRACKS:
        return [f"track must be one of {', '.join(PLAN_TRACKS)}, got {plan.track!r}"]
    found: list[str] = []
    limit = PLAN_TRACKS[plan.track]
    if len(plan.days) > limit:
        found.append(f"the {plan.track} track may have at most {limit} days, found {len(plan.days)}")
    if [day.day for day in plan.days] != list(range(1, len(plan.days) + 1)):
        found.append("days must be numbered contiguously from 1")
    for day in plan.days:
        minutes = day_minutes(day, book)
        if not MIN_DAY_MINUTES <= minutes <= MAX_DAY_MINUTES:
            found.append(f"day {day.day}: {minutes} minutes is outside {MIN_DAY_MINUTES}-{MAX_DAY_MINUTES}")
        if not day.goals or not day.self_check.strip():
            found.append(f"day {day.day}: goals and a self_check are required")
        for item in day.items:
            found.extend(_item_problems(plan, day, item, book))
    found.extend(_schedule_problems(plan, book))
    return found
