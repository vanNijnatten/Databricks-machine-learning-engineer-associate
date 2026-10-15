"""Load the book's structured content (YAML, TOML, Markdown front matter) into typed entities.

Every loader records the source file and line of each entity so that ``book check`` can report
problems as ``file:line — message``.
"""

import re
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from db_ml_associate.content.models import (
    Chapter,
    ChapterSection,
    ExamGuide,
    ExamObjective,
    Heading,
    Issue,
    PlanItem,
    PlatformChange,
    PracticeExam,
    Question,
    Reference,
    RuntimeTarget,
    StudyPlan,
    StudyPlanDay,
    Tutorial,
    TutorialStep,
)

LINE_KEY = "__line__"
CHAPTER_FILE = re.compile(r"^(\d{2})-[a-z0-9-]+\.md$")
FRONT_MATTER_FILE = re.compile(r"^(introduction|appendix-[a-z]-[a-z0-9-]+)\.md$")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
OBJECTIVES_TAG = re.compile(r"<!--\s*objectives:\s*(.*?)\s*-->")
HTML_ANCHOR = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"")
LIST_ITEM = re.compile(r"^\s*[-*]\s+(.+?)\s*$")
LEARNING_OBJECTIVES = "Learning objectives"


class ContentError(Exception):
    """A content file could not be parsed; ``issue`` locates the problem."""

    def __init__(self, issue: Issue) -> None:
        super().__init__(str(issue))
        self.issue = issue


class _LineLoader(yaml.SafeLoader):
    """A YAML loader that records the 1-based line of every mapping under ``__line__``."""


def _construct_mapping(loader: _LineLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    mapping = loader.construct_mapping(node, deep=True)
    mapping[LINE_KEY] = node.start_mark.line + 1
    return mapping


_LineLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _display(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _parse_yaml(text: str, path: str, first_line: int = 1) -> Any:
    try:
        return yaml.load(text, Loader=_LineLoader)  # noqa: S506 - _LineLoader extends SafeLoader  # nosec B506
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        line = first_line + (mark.line if mark is not None else 0)
        raise ContentError(Issue(path, line, f"invalid YAML: {error}")) from error


def read_yaml(path: Path, root: Path | None = None) -> Any:
    """Parse a YAML file, recording mapping line numbers."""
    return _parse_yaml(path.read_text(encoding="utf-8"), _display(path, root))


def _line(mapping: Mapping[str, Any]) -> int:
    return int(mapping.get(LINE_KEY, 0))


def _require(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ContentError(Issue(path, _line(mapping), f"missing required field {key!r}"))
    return mapping[key]


def _strings(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return tuple(part.strip() for part in values.split(",") if part.strip())
    return tuple(str(value) for value in values)


def _ints(values: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in values or ())


def _as_list(data: Any, path: str) -> list[Mapping[str, Any]]:
    if data is None:
        return []
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ContentError(Issue(path, 1, "expected a list of mappings"))
    return data


def _date(value: Any, mapping: Mapping[str, Any], key: str, path: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ContentError(Issue(path, _line(mapping), f"{key} must be a date (YYYY-MM-DD)")) from error


def load_objectives(path: Path, root: Path | None = None) -> tuple[ExamGuide, list[ExamObjective]]:
    """Load ``book/exam/objectives.yaml``."""
    shown = _display(path, root)
    data = read_yaml(path, root)
    if not isinstance(data, dict):
        raise ContentError(Issue(shown, 1, "expected a mapping with exam_guide and objectives"))
    guide_data = _require(data, "exam_guide", shown)
    guide = ExamGuide(
        version=_date(_require(guide_data, "version", shown), guide_data, "version", shown),
        url=str(_require(guide_data, "url", shown)),
        questions=int(_require(guide_data, "questions", shown)),
        minutes=int(_require(guide_data, "minutes", shown)),
        checked_on=_date(_require(guide_data, "checked_on", shown), guide_data, "checked_on", shown),
    )
    objectives = [
        ExamObjective(
            id=str(_require(item, "id", shown)),
            section=int(_require(item, "section", shown)),
            kind=str(_require(item, "kind", shown)),
            text=str(_require(item, "text", shown)),
            line=_line(item),
        )
        for item in _as_list(_require(data, "objectives", shown), shown)
    ]
    return guide, objectives


def load_questions(path: Path, root: Path | None = None) -> list[Question]:
    """Load a question file (``book/questions/*.yaml``)."""
    shown = _display(path, root)
    questions = []
    for item in _as_list(read_yaml(path, root), shown):
        options = _require(item, "options", shown)
        distractors = item.get("distractors") or {}
        questions.append(
            Question(
                id=str(_require(item, "id", shown)),
                objective=str(_require(item, "objective", shown)),
                pool=str(_require(item, "pool", shown)),
                type=str(_require(item, "type", shown)),
                difficulty=str(_require(item, "difficulty", shown)),
                stem=str(_require(item, "stem", shown)).strip(),
                options={str(k): str(v) for k, v in options.items() if k != LINE_KEY},
                answer=_strings(_require(item, "answer", shown)),
                explanation=str(_require(item, "explanation", shown)).strip(),
                distractors={str(k): str(v) for k, v in distractors.items() if k != LINE_KEY},
                chapter_ref=str(_require(item, "chapter_ref", shown)),
                path=shown,
                line=_line(item),
            )
        )
    return questions


def load_exam(path: Path, root: Path | None = None) -> PracticeExam:
    """Load a practice exam definition (``book/exams/*.yaml``)."""
    shown = _display(path, root)
    data = read_yaml(path, root)
    if not isinstance(data, dict):
        raise ContentError(Issue(shown, 1, "expected a mapping with id, minutes and questions"))
    return PracticeExam(
        id=str(_require(data, "id", shown)),
        minutes=int(_require(data, "minutes", shown)),
        questions=_strings(_require(data, "questions", shown)),
        path=shown,
        line=_line(data),
    )


def load_references(path: Path, root: Path | None = None) -> list[Reference]:
    """Load ``book/references.yaml``."""
    shown = _display(path, root)
    return [
        Reference(
            id=str(_require(item, "id", shown)),
            url=str(_require(item, "url", shown)),
            title=str(_require(item, "title", shown)),
            claim=str(_require(item, "claim", shown)).strip(),
            objectives=_strings(item.get("objectives")),
            chapters=_ints(item.get("chapters")),
            last_verified=_date(_require(item, "last_verified", shown), item, "last_verified", shown),
            status=str(item.get("status", "ok")),
            note=str(item.get("note") or ""),
            path=shown,
            line=_line(item),
        )
        for item in _as_list(read_yaml(path, root), shown)
    ]


def load_platform_changes(path: Path, root: Path | None = None) -> list[PlatformChange]:
    """Load ``book/exam/platform-changes.yaml``."""
    shown = _display(path, root)
    return [
        PlatformChange(
            exam_term=str(_require(item, "exam_term", shown)),
            current_term=str(_require(item, "current_term", shown)),
            change=str(_require(item, "change", shown)),
            exam_expects=str(_require(item, "exam_expects", shown)).strip(),
            objectives=_strings(item.get("objectives")),
            references=_strings(item.get("references")),
            path=shown,
            line=_line(item),
        )
        for item in _as_list(read_yaml(path, root), shown)
    ]


def _plan_item(raw: Mapping[str, Any], shown: str) -> PlanItem:
    kinds = [key for key in raw if key not in (LINE_KEY, "optional")]
    if len(kinds) != 1:
        raise ContentError(Issue(shown, _line(raw), "a plan item needs exactly one activity key"))
    kind = kinds[0]
    return PlanItem(kind=kind, target=str(raw[kind]), optional=bool(raw.get("optional", False)))


def load_plan(path: Path, root: Path | None = None) -> StudyPlan:
    """Load a study plan (``book/plans/*.yaml``)."""
    shown = _display(path, root)
    data = read_yaml(path, root)
    if not isinstance(data, dict):
        raise ContentError(Issue(shown, 1, "expected a mapping with track and days"))
    days = tuple(
        StudyPlanDay(
            day=int(_require(day, "day", shown)),
            goals=_strings(day.get("goals")) if isinstance(day.get("goals"), list) else (str(day.get("goals", "")),),
            items=tuple(_plan_item(item, shown) for item in _as_list(_require(day, "items", shown), shown)),
            self_check=str(day.get("self_check") or ""),
            line=_line(day),
        )
        for day in _as_list(_require(data, "days", shown), shown)
    )
    return StudyPlan(track=str(_require(data, "track", shown)), days=days, path=shown)


def load_runtime(path: Path) -> RuntimeTarget:
    """Load ``runtime.toml``."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    snapshot = data.get("docs_snapshot")
    return RuntimeTarget(
        databricks_runtime=str(data["databricks_runtime"]),
        docs_snapshot=snapshot if isinstance(snapshot, date) else date.fromisoformat(str(snapshot)),
        python=str(data["python"]),
        spark=str(data["spark"]),
        libraries={str(name): str(version) for name, version in data.get("libraries", {}).items()},
    )


# --- Markdown -----------------------------------------------------------------------------


def slugify(heading: str) -> str:
    """Return the GitHub anchor for a heading text."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


@dataclass(frozen=True)
class MarkdownLine:
    """A line of Markdown outside fenced code blocks and front matter."""

    number: int
    text: str


@dataclass
class MarkdownDocument:
    """The structural parts of a Markdown file needed by the content checks."""

    front_matter: dict[str, Any] | None
    headings: list[Heading]
    prose: list[MarkdownLine]
    lines: list[str]

    @property
    def anchors(self) -> set[str]:
        """Heading anchors plus explicit ``<a id="...">`` anchors."""
        explicit = {match.group(1) for line in self.prose for match in HTML_ANCHOR.finditer(line.text)}
        return {heading.anchor for heading in self.headings} | explicit


def iter_prose(lines: list[str], start: int = 0) -> Iterator[MarkdownLine]:
    """Yield lines (1-based numbers) that are not inside fenced code blocks."""
    in_code = False
    fence = ""
    for index in range(start, len(lines)):
        text = lines[index]
        match = FENCE.match(text)
        if match and (not in_code or match.group(1) == fence):
            in_code = not in_code
            fence = match.group(1)
            continue
        if not in_code:
            yield MarkdownLine(index + 1, text)


def parse_markdown(text: str, shown: str) -> MarkdownDocument:
    """Split a Markdown file into front matter, headings (with unique anchors) and prose lines."""
    lines = text.splitlines()
    front_matter: dict[str, Any] | None = None
    start = 0
    if lines and lines[0].strip() == "---":
        try:
            end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
        except StopIteration as error:
            raise ContentError(Issue(shown, 1, "front matter is not closed with ---")) from error
        parsed = _parse_yaml("\n".join(lines[1:end]), shown, first_line=2)
        if not isinstance(parsed, dict):
            raise ContentError(Issue(shown, 1, "front matter must be a mapping"))
        front_matter = parsed
        start = end + 1
    prose = list(iter_prose(lines, start))
    headings: list[Heading] = []
    used: dict[str, int] = {}
    for line in prose:
        match = HEADING.match(line.text)
        if not match:
            continue
        title = match.group(2)
        base = slugify(title)
        count = used.get(base, 0)
        used[base] = count + 1
        anchor = base if count == 0 else f"{base}-{count}"
        headings.append(Heading(level=len(match.group(1)), text=title, anchor=anchor, line=line.number))
    return MarkdownDocument(front_matter=front_matter, headings=headings, prose=prose, lines=lines)


def _tutorial(raw: Any, chapter: int, shown: str, line: int) -> Tutorial | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return Tutorial(chapter=chapter, notebook=raw, minutes=1, requires=(), free_edition="full")
    if not isinstance(raw, dict):
        raise ContentError(Issue(shown, line, "tutorial must be a mapping"))
    steps = tuple(
        TutorialStep(
            id=str(_require(step, "id", shown)),
            objectives=_strings(step.get("objectives")),
        )
        for step in _as_list(raw.get("steps"), shown)
    )
    return Tutorial(
        chapter=chapter,
        notebook=str(_require(raw, "notebook", shown)),
        minutes=int(raw.get("minutes", 0)),
        requires=_strings(raw.get("requires")),
        free_edition=str(raw.get("free_edition", "")),
        cleanup=bool(raw.get("cleanup", False)),
        steps=steps,
        extra_files=_strings(raw.get("extra_files")),
    )


def _learning_objectives(document: MarkdownDocument) -> tuple[str, ...]:
    heading = next((h for h in document.headings if h.text.strip() == LEARNING_OBJECTIVES), None)
    if heading is None:
        return ()
    following = [h.line for h in document.headings if h.line > heading.line]
    end = min(following) if following else len(document.lines) + 1
    items = []
    for line in document.prose:
        if heading.line < line.number < end:
            match = LIST_ITEM.match(line.text)
            if match:
                items.append(match.group(1))
    return tuple(items)


def _sections(document: MarkdownDocument, chapter: int) -> tuple[ChapterSection, ...]:
    sections = []
    for line in document.prose:
        match = OBJECTIVES_TAG.search(line.text)
        if not match:
            continue
        heading = max(
            (h for h in document.headings if h.line < line.number),
            key=lambda h: h.line,
            default=None,
        )
        if heading is None:
            continue
        sections.append(
            ChapterSection(
                chapter=chapter,
                anchor=heading.anchor,
                title=heading.text,
                objectives=_strings(match.group(1)),
                line=line.number,
            )
        )
    return tuple(sections)


def load_chapter(path: Path, root: Path | None = None) -> Chapter:
    """Load a numbered chapter ``book/NN-<slug>.md``."""
    shown = _display(path, root)
    document = parse_markdown(path.read_text(encoding="utf-8"), shown)
    meta = document.front_matter
    if meta is None:
        raise ContentError(Issue(shown, 1, "chapter must start with YAML front matter"))
    number = int(_require(meta, "chapter", shown))
    return Chapter(
        number=number,
        title=str(_require(meta, "title", shown)),
        objectives=_strings(meta.get("objectives")),
        path=shown,
        prerequisites=_ints(meta.get("prerequisites")),
        reading_minutes=int(meta.get("reading_minutes", 0)),
        references=_strings(meta.get("references")),
        tutorial=_tutorial(meta.get("tutorial"), number, shown, _line(meta)),
        learning_objectives=_learning_objectives(document),
        headings=tuple(document.headings),
        sections=_sections(document, number),
    )


# --- Whole book ---------------------------------------------------------------------------


@dataclass
class Book:
    """All structured content of the repository's book."""

    root: Path
    guide: ExamGuide | None = None
    objectives: list[ExamObjective] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    front_matter: list[Path] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    exams: list[PracticeExam] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    platform_changes: list[PlatformChange] = field(default_factory=list)
    plans: list[StudyPlan] = field(default_factory=list)
    runtime: RuntimeTarget | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def book_dir(self) -> Path:
        """The ``book/`` directory."""
        return self.root / "book"

    def chapter(self, number: int) -> Chapter | None:
        """Return the chapter with the given number, if it exists."""
        return next((chapter for chapter in self.chapters if chapter.number == number), None)


def _collect(book: Book, loader: Any, path: Path) -> Any:
    try:
        return loader(path, book.root)
    except ContentError as error:
        book.issues.append(error.issue)
        return None


def _load_pages(book: Book) -> None:
    for path in sorted(book.book_dir.glob("*.md")):
        if CHAPTER_FILE.match(path.name):
            chapter = _collect(book, load_chapter, path)
            if chapter is not None:
                book.chapters.append(chapter)
        elif FRONT_MATTER_FILE.match(path.name):
            book.front_matter.append(path)
    book.chapters.sort(key=lambda chapter: chapter.number)


def _load_yaml_content(book: Book) -> None:
    book_dir = book.book_dir
    objectives_file = book_dir / "exam" / "objectives.yaml"
    if objectives_file.exists():
        loaded = _collect(book, load_objectives, objectives_file)
        if loaded is not None:
            book.guide, book.objectives = loaded
    for path in sorted((book_dir / "questions").glob("*.yaml")):
        book.questions.extend(_collect(book, load_questions, path) or [])
    exams = (_collect(book, load_exam, path) for path in sorted((book_dir / "exams").glob("*.yaml")))
    book.exams = [exam for exam in exams if exam is not None]
    if (book_dir / "references.yaml").exists():
        book.references = _collect(book, load_references, book_dir / "references.yaml") or []
    changes_file = book_dir / "exam" / "platform-changes.yaml"
    if changes_file.exists():
        book.platform_changes = _collect(book, load_platform_changes, changes_file) or []
    plans = (_collect(book, load_plan, path) for path in sorted((book_dir / "plans").glob("*.yaml")))
    book.plans = [plan for plan in plans if plan is not None]


def _section_tag_issues(book: Book) -> list[Issue]:
    known = {objective.id for objective in book.objectives}
    if not known:
        return []
    return [
        Issue(
            chapter.path,
            section.line,
            f"section tag references unknown objective {objective_id}",
        )
        for chapter in book.chapters
        for section in chapter.sections
        for objective_id in section.objectives
        if objective_id not in known
    ]


def load_book(root: Path) -> Book:
    """Load every content file under ``root/book`` plus ``root/runtime.toml``.

    Parse problems are collected in ``Book.issues`` rather than raised, so that one broken file
    does not hide problems in the others.
    """
    book = Book(root=root)
    _load_yaml_content(book)
    _load_pages(book)
    if (root / "runtime.toml").exists():
        book.runtime = load_runtime(root / "runtime.toml")
    book.issues.extend(_section_tag_issues(book))
    return book
