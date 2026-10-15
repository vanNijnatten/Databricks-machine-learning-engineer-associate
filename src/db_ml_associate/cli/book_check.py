"""``book check``: validate every content file against the contracts and data model.

Each check returns :class:`~db_ml_associate.content.models.Issue` objects. Problems that make
content wrong are errors; incomplete work (uncovered objectives, a small question bank, missing
practice exams, stale references) is a warning so that the book can grow chapter by chapter.
``--strict`` turns every warning into an error and is used for the final quality gate.
"""

import re
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from db_ml_associate.cli.coverage import compute_coverage
from db_ml_associate.cli.snippets import sync_book
from db_ml_associate.content.loaders import (
    CHAPTER_FILE,
    FRONT_MATTER_FILE,
    Book,
    ContentError,
    iter_prose,
    load_book,
    parse_markdown,
)
from db_ml_associate.content.models import (
    MIN_QUESTION_BANK,
    MIN_REVIEW_QUESTIONS,
    SECTION_OBJECTIVE_COUNTS,
    TOTAL_OBJECTIVES,
    Chapter,
    Issue,
    missing_platform_changes,
    objective_set_problems,
    practice_exam_problems,
)
from db_ml_associate.content.official_samples import (
    OFFICIAL_SAMPLE_STEMS,
    SIMILARITY_THRESHOLD,
)
from db_ml_associate.content.plans import plan_problems

OBJECTIVES_FILE = "book/exam/objectives.yaml"
REFERENCES_FILE = "book/references.yaml"
PLATFORM_CHANGES_FILE = "book/exam/platform-changes.yaml"
QUESTIONS_DIR = "book/questions/"
REQUIRED_HEADINGS = ("Learning objectives", "Pitfalls", "Summary", "Review questions")
PROPORTION_TOLERANCE = 5.0  # percentage points
LAST_CHAPTER = 13
H2 = 2
DOCS_HOST = "docs.databricks.com"
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
INLINE_CODE = re.compile(r"`[^`]*`")
PIN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([^;\s]+)")


def _error(path: str, line: int, message: str) -> Issue:
    return Issue(path, line, message, "error")


def _warning(path: str, line: int, message: str) -> Issue:
    return Issue(path, line, message, "warning")


# --- Objectives and coverage -----------------------------------------------------------------


def check_objectives(book: Book) -> list[Issue]:
    """The registry holds exactly the 48 exam objectives split 18/9/15/6."""
    if book.guide is None:
        return [_error(OBJECTIVES_FILE, 1, "objectives registry is missing or unreadable")]
    issues = [_error(OBJECTIVES_FILE, 1, problem) for problem in objective_set_problems(book.objectives)]
    for objective in book.objectives:
        issues.extend(_error(OBJECTIVES_FILE, objective.line, problem) for problem in objective.problems())
    return issues


def check_coverage(book: Book) -> list[Issue]:
    """Every objective has a chapter section and a question; practical ones a tutorial step."""
    return [
        _warning(
            OBJECTIVES_FILE,
            row.objective.line,
            f"objective {row.objective.id} has no {gap} (open work)",
        )
        for row in compute_coverage(book)
        for gap in row.gaps
    ]


# --- Chapters ----------------------------------------------------------------------------------


def _heading_issues(chapter_path: str, h2: list[tuple[str, int]]) -> list[Issue]:
    issues = []
    titles = [title for title, _ in h2]
    positions = []
    for required in REQUIRED_HEADINGS:
        if required not in titles:
            issues.append(_error(chapter_path, 1, f"missing required heading '## {required}'"))
        else:
            positions.append(titles.index(required))
    if positions != sorted(positions):
        issues.append(
            _error(
                chapter_path,
                1,
                f"required headings out of order (expected {' → '.join(REQUIRED_HEADINGS)})",
            )
        )
    if titles and titles[0] != REQUIRED_HEADINGS[0]:
        issues.append(
            _error(
                chapter_path,
                h2[0][1],
                "the first section must be '## Learning objectives'",
            )
        )
    return issues


def _front_matter_issues(chapter: Chapter, known_objectives: set[str], known_references: set[str]) -> list[Issue]:
    path = chapter.path
    issues = [_error(path, 1, problem) for problem in chapter.problems()]
    if not chapter.name.startswith(f"{chapter.number:02d}-"):
        issues.append(
            _error(
                path,
                1,
                f"file name must start with {chapter.number:02d}- to match the chapter number",
            )
        )
    unknown_objectives = [o for o in chapter.objectives if known_objectives and o not in known_objectives]
    issues.extend(_error(path, 1, f"unknown objective {o} in front matter") for o in unknown_objectives)
    bad_prerequisites = [p for p in chapter.prerequisites if not 1 <= p <= LAST_CHAPTER]
    issues.extend(_error(path, 1, f"prerequisite {p} is not a chapter number (1-13)") for p in bad_prerequisites)
    unknown_references = [r for r in chapter.references if r not in known_references]
    issues.extend(_error(path, 1, f"unknown reference {r!r} (not in references.yaml)") for r in unknown_references)
    return issues


def _tutorial_issues(book: Book, chapter: Chapter, known_objectives: set[str]) -> list[Issue]:
    tutorial = chapter.tutorial
    if tutorial is None:
        return [_warning(chapter.path, 1, "chapter has no tutorial yet (open work)")]
    path = chapter.path
    issues = [_error(path, 1, problem) for problem in tutorial.problems()]
    missing = [f for f in (tutorial.notebook, *tutorial.extra_files) if not (book.root / f).is_file()]
    issues.extend(_warning(path, 1, f"tutorial file {f} not written yet (open work)") for f in missing)
    issues.extend(
        _error(path, 1, f"tutorial step {step.id}: unknown objective {objective_id}")
        for step in tutorial.steps
        for objective_id in step.objectives
        if known_objectives and objective_id not in known_objectives
    )
    return issues


def _review_question_issues(book: Book, chapter: Chapter) -> list[Issue]:
    prefix = f"Q-{chapter.number:02d}-"
    review = sum(1 for q in book.questions if q.pool == "review" and q.id.startswith(prefix))
    if review >= MIN_REVIEW_QUESTIONS:
        return []
    return [
        _error(
            chapter.path,
            1,
            f"fewer than {MIN_REVIEW_QUESTIONS} review questions ({review}) with ids {prefix}NNN",
        )
    ]


def check_chapters(book: Book) -> list[Issue]:
    """Front matter, required headings, tutorial files, references and review-question counts."""
    issues: list[Issue] = []
    known_objectives = {objective.id for objective in book.objectives}
    known_references = {reference.id for reference in book.references}
    numbers = Counter(chapter.number for chapter in book.chapters)
    for chapter in book.chapters:
        if numbers[chapter.number] > 1:
            issues.append(_error(chapter.path, 1, f"duplicate chapter number {chapter.number}"))
        issues.extend(_front_matter_issues(chapter, known_objectives, known_references))
        h2 = [(heading.text.strip(), heading.line) for heading in chapter.headings if heading.level == H2]
        issues.extend(_heading_issues(chapter.path, h2))
        issues.extend(_tutorial_issues(book, chapter, known_objectives))
        issues.extend(_review_question_issues(book, chapter))
    return issues


# --- Questions, exams and totals ---------------------------------------------------------------


def _similar_to_official(stem: str) -> int | None:
    normalized = " ".join(stem.lower().split())
    for number, official in enumerate(OFFICIAL_SAMPLE_STEMS, start=1):
        if SequenceMatcher(None, normalized, " ".join(official.lower().split())).ratio() >= SIMILARITY_THRESHOLD:
            return number
    return None


class _AnchorIndex:
    """Lazily parsed headings of Markdown files under ``book/``."""

    def __init__(self, book_dir: Path) -> None:
        self.book_dir = book_dir
        self._cache: dict[Path, set[str]] = {}

    def anchors(self, path: Path) -> set[str] | None:
        if not path.is_file():
            return None
        if path not in self._cache:
            try:
                document = parse_markdown(path.read_text(encoding="utf-8"), path.name)
                self._cache[path] = document.anchors
            except ContentError:
                self._cache[path] = set()
        return self._cache[path]


def check_questions(book: Book, anchors: _AnchorIndex) -> list[Issue]:
    """Schema, unique ids, known objectives, resolvable chapter refs and originality."""
    issues: list[Issue] = []
    known = {objective.id for objective in book.objectives} or None
    seen: set[str] = set()
    for question in book.questions:
        if question.id in seen:
            issues.append(_error(question.path, question.line, f"duplicate question id {question.id}"))
        seen.add(question.id)
        issues.extend(_error(question.path, question.line, problem) for problem in question.problems(known))
        official = _similar_to_official(question.stem)
        if official is not None:
            issues.append(
                _error(
                    question.path,
                    question.line,
                    f"question {question.id} is too similar to official sample question {official}",
                )
            )
        target, _, anchor = question.chapter_ref.partition("#")
        found = anchors.anchors(book.book_dir / target)
        if found is None or (anchor and anchor not in found):
            issues.append(
                _error(
                    question.path,
                    question.line,
                    f"question {question.id}: chapter_ref {question.chapter_ref} not found",
                )
            )
    return issues


def check_exams(book: Book) -> list[Issue]:
    """Each practice exam has 48 practice questions split 18/9/15/6, none reused."""
    if not book.exams:
        return [_warning("book/exams/", 1, "no practice exams yet (open work)")]
    by_id = {question.id: question for question in book.questions}
    paths = {exam.id: (exam.path, exam.line) for exam in book.exams}
    issues = []
    for problem in practice_exam_problems(book.exams, by_id):
        exam_id = problem.split(":", 1)[0].removeprefix("exam ").strip()
        path, line = paths.get(exam_id, ("book/exams/", 1))
        issues.append(_error(path, line or 1, problem))
    return issues


def check_totals(book: Book) -> list[Issue]:
    """At least 200 questions, with each section's share close to its share of objectives."""
    issues = []
    total = len(book.questions)
    if total < MIN_QUESTION_BANK:
        issues.append(
            _warning(
                QUESTIONS_DIR,
                1,
                f"question bank has {total} questions; at least {MIN_QUESTION_BANK} are required",
            )
        )
    if total:
        per_section = Counter(question.section for question in book.questions)
        for section, objectives in SECTION_OBJECTIVE_COUNTS.items():
            expected = 100 * objectives / TOTAL_OBJECTIVES
            actual = 100 * per_section[section] / total
            if abs(actual - expected) > PROPORTION_TOLERANCE:
                issues.append(
                    _warning(
                        QUESTIONS_DIR,
                        1,
                        f"section {section} share of questions is {actual:.1f}% "
                        f"(objectives: {expected:.1f}%, tolerance ±{PROPORTION_TOLERANCE:.0f} points)",
                    )
                )
    return issues


# --- Links and references ----------------------------------------------------------------------


def _markdown_files(book: Book) -> list[Path]:
    return sorted(book.book_dir.rglob("*.md"))


def _is_planned_page(name: str) -> bool:
    """A chapter or front-matter page that may not have been written yet."""
    return bool(CHAPTER_FILE.match(name) or FRONT_MATTER_FILE.match(name))


def _normalize_url(url: str) -> str:
    return url.split("#", 1)[0].rstrip("/")


def _link_issue(
    book: Book,
    anchors: _AnchorIndex,
    source: Path,
    line: int,
    target: str,
    registered: set[str],
) -> Issue | None:
    """Check one link target found on ``line`` of ``source``; return an issue or None."""
    shown = source.relative_to(book.root).as_posix()
    if target.startswith(("http://", "https://")):
        if DOCS_HOST in target and _normalize_url(target) not in registered:
            return _error(shown, line, f"docs link {target} is not registered in references.yaml")
        return None
    if target.startswith("mailto:"):
        return None
    file_part, _, anchor = target.partition("#")
    destination = (source.parent / file_part).resolve() if file_part else source
    if file_part and not destination.exists():
        if destination.parent == book.book_dir.resolve() and _is_planned_page(destination.name):
            return _warning(shown, line, f"link to planned page {file_part} (open work)")
        return _error(shown, line, f"broken link {target}")
    if anchor and destination.suffix == ".md":
        found = anchors.anchors(destination)
        if found is not None and anchor not in found:
            return _error(shown, line, f"broken anchor #{anchor} in link {target}")
    return None


def check_links(book: Book, anchors: _AnchorIndex) -> list[Issue]:
    """Relative links and anchors resolve; docs.databricks.com links are registered references."""
    registered = {_normalize_url(reference.url) for reference in book.references}
    issues = []
    for path in _markdown_files(book):
        for line in iter_prose(path.read_text(encoding="utf-8").splitlines()):
            for target in LINK.findall(INLINE_CODE.sub("", line.text)):
                issue = _link_issue(book, anchors, path, line.number, target, registered)
                if issue is not None:
                    issues.append(issue)
    return issues


def check_references(book: Book, today: date) -> list[Issue]:
    """Reference schema, unique ids and the 90-day verification window."""
    issues: list[Issue] = []
    if not (book.root / REFERENCES_FILE).exists():
        return [_warning(REFERENCES_FILE, 1, "references.yaml does not exist yet")]
    ids = Counter(reference.id for reference in book.references)
    for reference in book.references:
        if ids[reference.id] > 1:
            issues.append(
                _error(
                    reference.path,
                    reference.line,
                    f"duplicate reference id {reference.id}",
                )
            )
        issues.extend(_error(reference.path, reference.line, problem) for problem in reference.problems())
        if reference.is_stale(today):
            issues.append(
                _warning(
                    reference.path,
                    reference.line,
                    f"reference {reference.id} last verified {reference.last_verified} (more than 90 days ago)",
                )
            )
    return issues


def check_platform_changes(book: Book) -> list[Issue]:
    """The exam-vs-platform list covers the required set and cites known references."""
    if not (book.root / PLATFORM_CHANGES_FILE).exists():
        return [_error(PLATFORM_CHANGES_FILE, 1, "platform-changes.yaml is missing")]
    known_references = {reference.id for reference in book.references}
    issues = [
        _error(PLATFORM_CHANGES_FILE, 1, f"missing platform change for {keyword}")
        for keyword in missing_platform_changes(book.platform_changes)
    ]
    for change in book.platform_changes:
        issues.extend(_error(change.path, change.line, problem) for problem in change.problems())
        for reference_id in change.references:
            if reference_id not in known_references:
                issues.append(_error(change.path, change.line, f"unknown reference {reference_id!r}"))
    return issues


# --- Plans and runtime ---------------------------------------------------------------------------


def check_plans(book: Book) -> list[Issue]:
    """Each study plan schedules everything within the daily and total time limits."""
    if not book.plans:
        return [_warning("book/plans/", 1, "no study plans yet (open work)")]
    issues = []
    for plan in book.plans:
        lines = {f"day {day.day}:": day.line for day in plan.days}
        for problem in plan_problems(plan, book):
            line = next((line for prefix, line in lines.items() if problem.startswith(prefix)), 1)
            issues.append(_error(plan.path, line or 1, problem))
    return issues


def check_plan_prerequisites(book: Book) -> list[Issue]:
    """Warn when a plan schedules a chapter before one of its prerequisites."""
    issues = []
    for plan in book.plans:
        first_day: dict[int, int] = {}
        for day in plan.days:
            for item in day.items:
                if item.kind == "chapter" and item.target.isdigit():
                    first_day.setdefault(int(item.target), day.day)
        for number, day_number in first_day.items():
            chapter = book.chapter(number)
            if chapter is None:
                continue
            for prerequisite in chapter.prerequisites:
                scheduled = first_day.get(prerequisite)
                if scheduled is None or scheduled > day_number:
                    issues.append(
                        _warning(
                            plan.path,
                            1,
                            f"chapter {number} is scheduled on day {day_number} before its prerequisite chapter {prerequisite}",
                        )
                    )
    return issues


def _pyproject_pins(pyproject: Path) -> dict[str, str]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    requirements = list(data.get("project", {}).get("dependencies", []))
    for group in data.get("dependency-groups", {}).values():
        requirements.extend(item for item in group if isinstance(item, str))
    pins = {}
    for requirement in requirements:
        match = PIN.match(requirement)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def check_runtime(book: Book) -> list[Issue]:
    """pyproject.toml pins and .python-version agree with runtime.toml."""
    if book.runtime is None:
        return [_error("runtime.toml", 1, "runtime.toml is missing")]
    issues = []
    pyproject = book.root / "pyproject.toml"
    if pyproject.exists():
        pins = _pyproject_pins(pyproject)
        for name, version in book.runtime.libraries.items():
            pinned = pins.get(name.lower())
            if pinned is not None and pinned != version:
                issues.append(
                    _error(
                        "pyproject.toml",
                        1,
                        f"{name} is pinned to {pinned} but runtime.toml targets {version}",
                    )
                )
    python_version = book.root / ".python-version"
    if python_version.exists():
        local = python_version.read_text(encoding="utf-8").strip().split(".")[:2]
        if local != book.runtime.python.split(".")[:2]:
            issues.append(
                _error(
                    ".python-version",
                    1,
                    f"local Python {'.'.join(local)} != runtime {book.runtime.python}",
                )
            )
    return issues


# --- Entry point ---------------------------------------------------------------------------------


def run_checks(root: Path, strict: bool = False, today: date | None = None) -> list[Issue]:
    """Run every check on the book under ``root``; with ``strict`` warnings become errors."""
    book = load_book(root)
    anchors = _AnchorIndex(book.book_dir)
    checks: Iterable[Callable[[], list[Issue]]] = (
        lambda: list(book.issues),
        lambda: check_objectives(book),
        lambda: check_chapters(book),
        lambda: check_questions(book, anchors),
        lambda: check_exams(book),
        lambda: check_totals(book),
        lambda: sync_book(root, check=True),
        lambda: check_links(book, anchors),
        lambda: check_references(book, today or date.today()),
        lambda: check_platform_changes(book),
        lambda: check_plans(book),
        lambda: check_plan_prerequisites(book),
        lambda: check_coverage(book),
        lambda: check_runtime(book),
    )
    issues = [issue for check in checks for issue in check()]
    if strict:
        issues = [replace(issue, severity="error") for issue in issues]
    return issues
