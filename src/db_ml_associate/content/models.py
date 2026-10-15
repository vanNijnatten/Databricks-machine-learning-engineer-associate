"""Typed content entities of the study book and their validation rules.

The fields and rules follow ``specs/001-exam-study-guide/data-model.md``. Every entity is an
immutable dataclass; ``problems()`` returns human-readable rule violations (an empty list means
the entity is valid). Cross-entity rules (for example "no question reused across exams") live in
module-level functions.
"""

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

Severity = Literal["error", "warning"]

SECTION_OBJECTIVE_COUNTS: Mapping[int, int] = {1: 18, 2: 9, 3: 15, 4: 6}
TOTAL_OBJECTIVES = sum(SECTION_OBJECTIVE_COUNTS.values())
EXAM_QUESTIONS = 48
EXAM_MINUTES = 90
MIN_REVIEW_QUESTIONS = 8
MIN_LEARNING_OBJECTIVES = 3
MIN_QUESTION_BANK = 200
REFERENCE_MAX_AGE_DAYS = 90

OBJECTIVE_ID = re.compile(r"^S([1-4])\.(\d{2})$")
OBJECTIVE_KINDS = ("conceptual", "practical")
QUESTION_TYPES = ("single", "multiple")
QUESTION_POOLS = ("review", "practice")
DIFFICULTIES = ("easy", "medium", "hard")
OPTION_LETTERS = ("A", "B", "C", "D", "E")
MIN_OPTIONS = 4
MAX_OPTIONS = 5
MIN_MULTIPLE_ANSWERS = 2
LAST_CHAPTER = 13
TUTORIAL_REQUIREMENTS = (
    "unity_catalog",
    "serverless",
    "classic_ml_runtime",
    "model_serving",
    "online_store",
    "lakeflow",
)
PAID_REQUIREMENTS = ("model_serving", "online_store")
FREE_EDITION_STATUSES = ("full", "partial", "read_through")
REFERENCE_STATUSES = ("ok", "moved", "discrepancy")
CHANGE_KINDS = ("renamed", "deprecated", "replaced", "repackaged")
PLAN_TRACKS = {"fast": 14, "standard": 21}
PLAN_ITEM_KINDS = ("introduction", "chapter", "tutorial", "exam", "review", "appendix")

# Keywords that must each appear in at least one platform change's exam term (SC-007).
REQUIRED_PLATFORM_CHANGES = (
    "Hyperopt",
    "Delta Live Tables",
    "Online tables",
    "Workspace model registry",
    "FeatureStoreClient",
    "AutoML",
    "MLflow",
)


@dataclass(frozen=True)
class Issue:
    """A validation finding attached to a file and line."""

    path: str
    line: int
    message: str
    severity: Severity = "error"

    def __str__(self) -> str:
        return f"{self.path}:{self.line} — {self.message}"


def section_of(objective_id: str) -> int | None:
    """Return the exam section encoded in an objective id like ``S3.10``, or None if malformed."""
    match = OBJECTIVE_ID.match(objective_id)
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class ExamGuide:
    """The official exam guide version the book targets."""

    version: date
    url: str
    questions: int
    minutes: int
    checked_on: date


@dataclass(frozen=True)
class ExamObjective:
    """One bullet of the official exam outline."""

    id: str
    section: int
    kind: str
    text: str
    line: int = 0

    def problems(self) -> list[str]:
        """Return rule violations for this objective."""
        found: list[str] = []
        encoded_section = section_of(self.id)
        if encoded_section is None:
            found.append(f"objective id {self.id!r} must match S<1-4>.<NN>")
        elif encoded_section != self.section:
            found.append(f"objective {self.id}: section {self.section} does not match its id")
        if self.kind not in OBJECTIVE_KINDS:
            found.append(f"objective {self.id}: kind must be one of {', '.join(OBJECTIVE_KINDS)}")
        if not self.text.strip():
            found.append(f"objective {self.id}: text is empty")
        return found


def objective_set_problems(objectives: Iterable[ExamObjective]) -> list[str]:
    """Check the whole objective registry: 48 unique objectives split 18/9/15/6."""
    objectives = list(objectives)
    found: list[str] = []
    for objective_id, count in Counter(objective.id for objective in objectives).items():
        if count > 1:
            found.append(f"duplicate objective id {objective_id}")
    if len(objectives) != TOTAL_OBJECTIVES:
        found.append(f"expected {TOTAL_OBJECTIVES} objectives, found {len(objectives)}")
    per_section = Counter(objective.section for objective in objectives)
    for section, expected in SECTION_OBJECTIVE_COUNTS.items():
        if per_section[section] != expected:
            found.append(f"section {section} must have {expected} objectives, found {per_section[section]}")
    return found


@dataclass(frozen=True)
class Question:
    """An exam-style question used in chapter reviews or practice exams."""

    id: str
    objective: str
    pool: str
    type: str
    difficulty: str
    stem: str
    options: Mapping[str, str]
    answer: tuple[str, ...]
    explanation: str
    distractors: Mapping[str, str]
    chapter_ref: str
    path: str = ""
    line: int = 0

    @property
    def section(self) -> int | None:
        """The exam section of the question's objective."""
        return section_of(self.objective)

    def problems(self, known_objectives: set[str] | None = None) -> list[str]:
        """Return rule violations; ``known_objectives`` enables the objective-exists check."""
        label = f"question {self.id}"
        found = self._field_problems(label, known_objectives)
        found.extend(self._answer_problems(label))
        return found

    def _field_problems(self, label: str, known_objectives: set[str] | None) -> list[str]:
        found: list[str] = []
        if section_of(self.objective) is None or (
            known_objectives is not None and self.objective not in known_objectives
        ):
            found.append(f"{label}: unknown objective {self.objective!r}")
        for name, value, allowed in (
            ("pool", self.pool, QUESTION_POOLS),
            ("difficulty", self.difficulty, DIFFICULTIES),
            ("type", self.type, QUESTION_TYPES),
        ):
            if value not in allowed:
                found.append(f"{label}: {name} must be one of {', '.join(allowed)}")
        if not self.stem.strip() or not self.explanation.strip():
            found.append(f"{label}: stem and explanation are required")
        if not self.chapter_ref.strip():
            found.append(f"{label}: chapter_ref is required")
        return found

    def _answer_problems(self, label: str) -> list[str]:
        found: list[str] = []
        letters = tuple(self.options)
        if not MIN_OPTIONS <= len(letters) <= MAX_OPTIONS or letters != OPTION_LETTERS[: len(letters)]:
            found.append(f"{label}: options must be 4-5 options labelled A-D or A-E")
        if not set(self.answer) <= set(letters) or len(set(self.answer)) != len(self.answer):
            found.append(f"{label}: answer {list(self.answer)} must be distinct option letters")
        if self.type == "single" and len(self.answer) != 1:
            found.append(f"{label}: type single needs exactly 1 answer")
        if self.type == "multiple" and len(self.answer) < MIN_MULTIPLE_ANSWERS:
            found.append(f"{label}: type multiple needs at least {MIN_MULTIPLE_ANSWERS} answers")
        explained = {letter for letter, text in self.distractors.items() if text.strip()}
        missing = sorted(set(letters) - set(self.answer) - explained)
        if missing:
            found.append(f"{label}: missing distractor explanation for option(s) {', '.join(missing)}")
        return found


@dataclass(frozen=True)
class PracticeExam:
    """A timed, full-length practice exam referencing practice questions by id."""

    id: str
    minutes: int
    questions: tuple[str, ...]
    path: str = ""
    line: int = 0


def _exam_problems(exam: PracticeExam, questions: Mapping[str, Question], seen: dict[str, str]) -> list[str]:
    """Problems of one exam; ``seen`` maps question ids to the exam that used them first."""
    found: list[str] = []
    if len(exam.questions) != EXAM_QUESTIONS:
        found.append(f"exam {exam.id}: must have exactly {EXAM_QUESTIONS} questions, found {len(exam.questions)}")
    if exam.minutes != EXAM_MINUTES:
        found.append(f"exam {exam.id}: minutes must be {EXAM_MINUTES}")
    per_section: Counter[int | None] = Counter()
    for question_id in exam.questions:
        if question_id in seen:
            found.append(f"exam {exam.id}: question {question_id} reused (also in {seen[question_id]})")
        seen.setdefault(question_id, exam.id)
        question = questions.get(question_id)
        if question is None:
            found.append(f"exam {exam.id}: unknown question {question_id}")
            continue
        if question.pool != "practice":
            found.append(f"exam {exam.id}: question {question_id} must be in the practice pool")
        per_section[question.section] += 1
    if len(exam.questions) == EXAM_QUESTIONS:
        found.extend(
            f"exam {exam.id}: section {section} must have {expected} questions, found {per_section[section]}"
            for section, expected in SECTION_OBJECTIVE_COUNTS.items()
            if per_section[section] != expected
        )
    return found


def practice_exam_problems(exams: Iterable[PracticeExam], questions: Mapping[str, Question]) -> list[str]:
    """Check exam size, time, pools, section split and that no question is reused."""
    seen: dict[str, str] = {}
    return [problem for exam in exams for problem in _exam_problems(exam, questions, seen)]


@dataclass(frozen=True)
class TutorialStep:
    """A step of a tutorial notebook and the objectives it exercises."""

    id: str
    objectives: tuple[str, ...]


@dataclass(frozen=True)
class Tutorial:
    """The hands-on notebook paired with a chapter."""

    chapter: int
    notebook: str
    minutes: int
    requires: tuple[str, ...]
    free_edition: str
    cleanup: bool = False
    steps: tuple[TutorialStep, ...] = ()
    extra_files: tuple[str, ...] = ()

    def problems(self) -> list[str]:
        """Return rule violations for this tutorial."""
        found: list[str] = []
        unknown = sorted(set(self.requires) - set(TUTORIAL_REQUIREMENTS))
        if unknown:
            found.append(f"tutorial requires unknown feature(s): {', '.join(unknown)}")
        if self.free_edition not in FREE_EDITION_STATUSES:
            found.append(f"tutorial free_edition must be one of {', '.join(FREE_EDITION_STATUSES)}")
        if self.minutes <= 0:
            found.append("tutorial minutes must be positive")
        if set(self.requires) & set(PAID_REQUIREMENTS) and not self.cleanup:
            found.append("tutorial cleanup must be true when it requires model_serving or online_store")
        return found


@dataclass(frozen=True)
class Heading:
    """A Markdown heading with its GitHub-style anchor."""

    level: int
    text: str
    anchor: str
    line: int


@dataclass(frozen=True)
class ChapterSection:
    """A chapter heading tagged with the exam objectives it teaches."""

    chapter: int
    anchor: str
    title: str
    objectives: tuple[str, ...]
    line: int


@dataclass(frozen=True)
class Chapter:
    """A numbered chapter (01-13) of the book."""

    number: int
    title: str
    objectives: tuple[str, ...]
    path: str
    prerequisites: tuple[int, ...] = ()
    reading_minutes: int = 0
    references: tuple[str, ...] = ()
    tutorial: Tutorial | None = None
    learning_objectives: tuple[str, ...] = ()
    headings: tuple[Heading, ...] = ()
    sections: tuple[ChapterSection, ...] = ()

    @property
    def name(self) -> str:
        """The chapter's file name, e.g. ``10-validation-and-tuning.md``."""
        return self.path.rsplit("/", 1)[-1]

    def problems(self) -> list[str]:
        """Return front-matter rule violations (structure is checked by ``book check``)."""
        found: list[str] = []
        if not 1 <= self.number <= LAST_CHAPTER:
            found.append(f"chapter number {self.number} must be between 1 and 13")
        if not self.objectives:
            found.append("chapter must map to at least one objective")
        if self.reading_minutes <= 0:
            found.append("reading_minutes must be positive")
        if self.number in self.prerequisites:
            found.append("a chapter cannot be its own prerequisite")
        if len(self.learning_objectives) < MIN_LEARNING_OBJECTIVES:
            found.append(f"chapter must list at least {MIN_LEARNING_OBJECTIVES} learning objectives")
        return found


@dataclass(frozen=True)
class Reference:
    """A page of the official Databricks documentation cited by the book."""

    id: str
    url: str
    title: str
    claim: str
    objectives: tuple[str, ...] = ()
    chapters: tuple[int, ...] = ()
    last_verified: date = field(default_factory=date.today)
    status: str = "ok"
    note: str = ""
    path: str = ""
    line: int = 0

    def is_stale(self, today: date, max_age_days: int = REFERENCE_MAX_AGE_DAYS) -> bool:
        """True if the page was last verified more than ``max_age_days`` before ``today``."""
        return today - self.last_verified > timedelta(days=max_age_days)

    def problems(self) -> list[str]:
        """Return rule violations for this reference."""
        found: list[str] = []
        if not self.url.startswith("https://"):
            found.append(f"reference {self.id}: url must be https")
        if self.status not in REFERENCE_STATUSES:
            found.append(f"reference {self.id}: status must be one of {', '.join(REFERENCE_STATUSES)}")
        if self.status == "discrepancy" and not self.note.strip():
            found.append(f"reference {self.id}: a discrepancy needs a note")
        if not self.claim.strip():
            found.append(f"reference {self.id}: claim is required")
        return found


@dataclass(frozen=True)
class PlatformChange:
    """A difference between exam terminology and the current Databricks platform."""

    exam_term: str
    current_term: str
    change: str
    exam_expects: str
    objectives: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    path: str = ""
    line: int = 0

    def problems(self) -> list[str]:
        """Return rule violations for this platform change."""
        found: list[str] = []
        if self.change not in CHANGE_KINDS:
            found.append(f"platform change {self.exam_term!r}: change must be one of {', '.join(CHANGE_KINDS)}")
        if not self.references:
            found.append(f"platform change {self.exam_term!r}: at least one reference is required")
        return found


def missing_platform_changes(changes: Iterable[PlatformChange]) -> list[str]:
    """Return required platform-change keywords not covered by any exam term (SC-007)."""
    terms = [change.exam_term.lower() for change in changes]
    return [keyword for keyword in REQUIRED_PLATFORM_CHANGES if not any(keyword.lower() in term for term in terms)]


@dataclass(frozen=True)
class PlanItem:
    """One activity of a study-plan day."""

    kind: str
    target: str
    optional: bool = False


@dataclass(frozen=True)
class StudyPlanDay:
    """A day of a study plan."""

    day: int
    goals: tuple[str, ...]
    items: tuple[PlanItem, ...]
    self_check: str
    line: int = 0


@dataclass(frozen=True)
class StudyPlan:
    """A 14-day (fast) or 21-day (standard) study plan."""

    track: str
    days: tuple[StudyPlanDay, ...]
    path: str = ""


@dataclass(frozen=True)
class RuntimeTarget:
    """The Databricks runtime and library versions all tutorial code targets."""

    databricks_runtime: str
    docs_snapshot: date
    python: str
    spark: str
    libraries: Mapping[str, str]
