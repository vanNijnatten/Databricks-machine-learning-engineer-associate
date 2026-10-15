"""``exam take`` and ``exam render``: timed practice exams with per-section scores (FR-017-FR-019).

Scoring follows the real exam: a multiple-selection question counts only when exactly the correct
options are chosen. A section is "ready" at 80 % or more (SC-003). Options are shuffled per attempt
with a seeded random generator, so a seed reproduces an attempt.
"""

import json
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from db_ml_associate.content.loaders import Book, load_book
from db_ml_associate.content.models import OPTION_LETTERS, SECTION_OBJECTIVE_COUNTS, Question

READINESS_THRESHOLD = 0.8
HOME_VARIABLE = "DB_ML_ASSOCIATE_HOME"
SECTION_NAMES = {
    1: "Databricks Machine Learning",
    2: "Data Processing",
    3: "Model Development",
    4: "Model Deployment",
}

console = Console(highlight=False, soft_wrap=True)
exam_app = typer.Typer(no_args_is_help=True, help="Take or print the practice exams.")


@dataclass(frozen=True)
class SectionScore:
    """Correct answers in one exam section."""

    section: int
    correct: int
    total: int

    @property
    def share(self) -> float:
        """Fraction correct (0 for an empty section)."""
        return self.correct / self.total if self.total else 0.0

    @property
    def ready(self) -> bool:
        """At least 80 % correct."""
        return self.total > 0 and self.share >= READINESS_THRESHOLD


@dataclass(frozen=True)
class ExamResult:
    """Score of one attempt."""

    sections: tuple[SectionScore, ...]
    missed: list[Question]

    @property
    def correct(self) -> int:
        """Correct answers over all sections."""
        return sum(section.correct for section in self.sections)

    @property
    def total(self) -> int:
        """Questions in the attempt."""
        return sum(section.total for section in self.sections)


def exam_questions(book: Book, exam_id: str, section: int | None = None) -> list[Question]:
    """The questions of a practice exam in exam order, optionally only one section's."""
    exam = next((exam for exam in book.exams if exam.id == exam_id), None)
    if exam is None:
        raise KeyError(f"unknown practice exam {exam_id}")
    by_id = {question.id: question for question in book.questions}
    questions = [by_id[question_id] for question_id in exam.questions]
    return [question for question in questions if section is None or question.section == section]


def is_correct(question: Question, chosen: set[str]) -> bool:
    """All-or-nothing: exactly the correct option letters."""
    return chosen == set(question.answer)


def score_attempt(questions: Sequence[Question], answers: Mapping[str, set[str]]) -> ExamResult:
    """Score answers (original option letters per question id); unanswered questions are wrong."""
    sections = []
    for section in SECTION_OBJECTIVE_COUNTS:
        in_section = [question for question in questions if question.section == section]
        if in_section:
            correct = sum(1 for q in in_section if is_correct(q, answers.get(q.id, set())))
            sections.append(SectionScore(section, correct, len(in_section)))
    missed = [question for question in questions if not is_correct(question, answers.get(question.id, set()))]
    return ExamResult(sections=tuple(sections), missed=missed)


def shuffle_options(question: Question, rng: random.Random) -> list[tuple[str, str, str]]:
    """Return ``(shown letter, original letter, text)`` for the options in a shuffled order."""
    originals = list(question.options)
    rng.shuffle(originals)
    return [(OPTION_LETTERS[index], original, question.options[original]) for index, original in enumerate(originals)]


# --- Rendering ------------------------------------------------------------------------------------


def _instruction(question: Question) -> str:
    count = len(question.answer)
    return "*Select one answer.*" if count == 1 else f"*Select {count} answers.*"


def render_exam(book: Book, exam_id: str, answers: bool = False) -> str:
    """Markdown of an exam: the questions, or (``answers=True``) the answer key."""
    questions = exam_questions(book, exam_id)
    if not answers:
        lines = [f"## Practice exam {exam_id}", "", f"{len(questions)} questions · 90 minutes · no notes.", ""]
        for number, question in enumerate(questions, start=1):
            options = "\n".join(f"- **{letter}.** {text}" for letter, text in question.options.items())
            lines += [f"### Question {number}", "", question.stem, "", options, "", _instruction(question), ""]
        return "\n".join(lines).rstrip("\n") + "\n"
    lines = [f"## Practice exam {exam_id} — Answer key", ""]
    for number, question in enumerate(questions, start=1):
        distractors = "\n".join(
            f"- **{letter}**: {question.distractors[letter]}"
            for letter in question.options
            if letter not in question.answer and letter in question.distractors
        )
        lines += [
            f"### {exam_id} question {number} — {', '.join(question.answer)}",
            "",
            question.explanation,
            "",
            distractors,
            "",
            f"Section {question.section} · objective {question.objective} · [Revisit]({question.chapter_ref})",
            "",
        ]
    return "\n".join(lines).rstrip("\n") + "\n"


# --- Interactive attempt --------------------------------------------------------------------------


def _parse_choice(raw: str, shown: Sequence[tuple[str, str, str]], expected: int) -> set[str] | None:
    letters = [part.strip().upper() for part in raw.replace(" ", ",").split(",") if part.strip()]
    mapping = {letter: original for letter, original, _ in shown}
    if len(letters) != expected or len(set(letters)) != expected or not set(letters) <= set(mapping):
        return None
    return {mapping[letter] for letter in letters}


def _ask(
    number: int, total: int, question: Question, shown: Sequence[tuple[str, str, str]], remaining: str
) -> set[str]:
    expected = len(question.answer)
    console.print(f"\n[bold]Question {number} of {total}[/bold]{remaining}", markup=True)
    console.print(question.stem)
    for letter, _, text in shown:
        console.print(f"  {letter}. {text}")
    hint = "one letter" if expected == 1 else f"{expected} letters separated by commas"
    while True:
        chosen = _parse_choice(typer.prompt(f"Your answer ({hint})"), shown, expected)
        if chosen is not None:
            return chosen
        console.print(f"Answer with {hint} from {shown[0][0]}-{shown[-1][0]}.")


def _print_result(result: ExamResult) -> None:
    console.print("\n[bold]Result[/bold]", markup=True)
    for section in result.sections:
        verdict = "ready" if section.ready else "not ready yet (below 80 %)"
        console.print(
            f"Section {section.section} ({SECTION_NAMES[section.section]}): "
            f"{section.correct}/{section.total} = {section.share:.0%} — {verdict}"
        )
    share = result.correct / result.total if result.total else 0.0
    console.print(f"Overall: {result.correct}/{result.total} = {share:.0%}")
    if result.missed:
        console.print("\nRevisit:")
        for question in result.missed:
            console.print(f"- {question.id} (objective {question.objective}): book/{question.chapter_ref}")


def _save(exam_id: str, seed: int, result: ExamResult, answers: Mapping[str, set[str]]) -> Path:
    home = Path(os.environ.get(HOME_VARIABLE) or Path.home() / ".db-ml-associate")
    folder = home / "attempts"
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    path = folder / f"{timestamp}.json"
    data = {
        "exam": exam_id,
        "seed": seed,
        "correct": result.correct,
        "total": result.total,
        "sections": {str(s.section): {"correct": s.correct, "total": s.total} for s in result.sections},
        "answers": {question_id: sorted(letters) for question_id, letters in answers.items()},
        "missed": [question.id for question in result.missed],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _load(root: Path, exam_id: str, section: int | None = None) -> list[Question]:
    try:
        return exam_questions(load_book(root), exam_id.upper(), section)
    except KeyError as error:
        raise typer.BadParameter(f"unknown practice exam {exam_id} (expected PE1 or PE2)") from error


RootOption = Annotated[Path, typer.Option("--root", help="Repository root.", file_okay=False)]


@exam_app.command("take")
def exam_take(
    exam_id: Annotated[str, typer.Argument(help="PE1 or PE2.")],
    root: RootOption = Path("."),
    no_timer: Annotated[bool, typer.Option("--no-timer", help="Do not enforce the time limit.")] = False,
    section: Annotated[int | None, typer.Option("--section", min=1, max=4, help="Only one section.")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Reproduce an option order.")] = None,
    save: Annotated[bool, typer.Option("--save", help="Save the attempt as JSON.")] = False,
) -> None:
    """Take a practice exam in the terminal and get per-section scores."""
    questions = _load(root, exam_id, section)
    seed = seed if seed is not None else random.SystemRandom().randrange(1_000_000)
    rng = random.Random(seed)  # nosec B311 - shuffling options, not security
    minutes = round(90 * len(questions) / 48)
    deadline = None if no_timer else time.monotonic() + minutes * 60
    console.print(f"Practice exam {exam_id.upper()}: {len(questions)} questions, {minutes} minutes (seed {seed}).")
    answers: dict[str, set[str]] = {}
    for number, question in enumerate(questions, start=1):
        shown = shuffle_options(question, rng)
        remaining = ""
        if deadline is not None:
            left = deadline - time.monotonic()
            if left <= 0:
                console.print("\nTime is up — unanswered questions count as wrong.")
                break
            remaining = f" · {int(left // 60)} min left"
        answers[question.id] = _ask(number, len(questions), question, shown, remaining)
    result = score_attempt(questions, answers)
    _print_result(result)
    if save:
        console.print(f"\nSaved to {_save(exam_id.upper(), seed, result, answers)}")


@exam_app.command("render")
def exam_render(
    exam_id: Annotated[str, typer.Argument(help="PE1 or PE2.")],
    root: RootOption = Path("."),
    answers: Annotated[bool, typer.Option("--answers", help="Print the answer key.")] = False,
) -> None:
    """Print a practice exam (or its answer key) as Markdown."""
    _load(root, exam_id)
    typer.echo(render_exam(load_book(root), exam_id.upper(), answers=answers), nl=False)


APPENDIX_D_HEADER = """\
# Appendix D — Practice exams

Two full-length practice exams with 48 questions each, split across the four sections like the real
exam (18 / 9 / 15 / 6). Take each one in **90 minutes without notes**, then score yourself with the
worksheet and read the answer key: every answer explains why the correct options are right and why
each other option is wrong, and links to the section to revisit.

You can also take them in the terminal, timed and scored per section, with shuffled options:

```bash
uv run db-ml-associate exam take PE1          # or PE2; --section 3 to practise one section
```

Book the exam when you score **at least 80 % in every section of both exams** (the official pass
mark is not published).

Generated by `uv run db-ml-associate book render-appendix-d` from [`questions/`](questions/) and
[`exams/`](exams/) — edit those files, not this page.

## Score worksheet

| Section | Questions | PE1 correct | PE1 % | PE2 correct | PE2 % | Ready (≥ 80 %)? |
|---|---|---|---|---|---|---|
{rows}
| **Total** | **{total}** | | | | | |
"""


def render_appendix_d(book: Book) -> str:
    """Markdown of Appendix D: worksheet, both exams, then both answer keys."""
    exams = sorted(exam.id for exam in book.exams)
    rows = "\n".join(
        f"| {section}. {SECTION_NAMES[section]} | {count} | | | | | |"
        for section, count in SECTION_OBJECTIVE_COUNTS.items()
    )
    parts = [APPENDIX_D_HEADER.format(rows=rows, total=sum(SECTION_OBJECTIVE_COUNTS.values())).rstrip("\n")]
    parts += [render_exam(book, exam_id).rstrip("\n") for exam_id in exams]
    parts += [render_exam(book, exam_id, answers=True).rstrip("\n") for exam_id in exams]
    return "\n\n".join(parts) + "\n"
