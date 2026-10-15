"""Objective coverage: which chapter sections, tutorial steps and questions address each objective."""

import re
from dataclasses import dataclass
from pathlib import Path

from db_ml_associate.content.loaders import Book
from db_ml_associate.content.models import ExamObjective

START_MARKER = "<!-- coverage:start -->"
END_MARKER = "<!-- coverage:end -->"


@dataclass(frozen=True)
class ObjectiveCoverage:
    """Where one exam objective is taught, practised and assessed."""

    objective: ExamObjective
    sections: tuple[str, ...]
    tutorial_steps: tuple[str, ...]
    questions: int
    gaps: tuple[str, ...]


def compute_coverage(book: Book) -> list[ObjectiveCoverage]:
    """Return one coverage row per objective, in exam-guide order.

    Gaps are ``chapter section`` (no tagged section), ``question`` (no review or practice
    question) and, for practical objectives only, ``tutorial step``.
    """
    rows = []
    for objective in book.objectives:
        sections = tuple(
            f"{chapter.name}#{section.anchor}"
            for chapter in book.chapters
            for section in chapter.sections
            if objective.id in section.objectives
        )
        steps = tuple(
            f"ch{chapter.number:02d}:{step.id}"
            for chapter in book.chapters
            if chapter.tutorial is not None
            for step in chapter.tutorial.steps
            if objective.id in step.objectives
        )
        questions = sum(1 for question in book.questions if question.objective == objective.id)
        gaps = []
        if not sections:
            gaps.append("chapter section")
        if questions == 0:
            gaps.append("question")
        if objective.kind == "practical" and not steps:
            gaps.append("tutorial step")
        rows.append(
            ObjectiveCoverage(
                objective=objective,
                sections=sections,
                tutorial_steps=steps,
                questions=questions,
                gaps=tuple(gaps),
            )
        )
    return rows


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_coverage_table(coverage: list[ObjectiveCoverage]) -> str:
    """Render the coverage rows as a Markdown table (links are relative to ``book/``)."""
    covered = sum(1 for row in coverage if not row.gaps)
    lines = [
        f"**Fully covered: {covered}/{len(coverage)} objectives.**",
        "",
        "| Objective | Section | Kind | Chapter sections | Tutorial steps | Questions | Gaps |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in coverage:
        sections = "<br>".join(f"[{target}]({target})" for target in row.sections) or "—"
        steps = ", ".join(row.tutorial_steps) or "—"
        gaps = ", ".join(row.gaps) or "—"
        lines.append(
            f"| {row.objective.id} {_cell(row.objective.text)} | {row.objective.section} | {row.objective.kind} "
            f"| {sections} | {steps} | {row.questions} | {gaps} |"
        )
    return "\n".join(lines) + "\n"


def write_coverage(readme: Path, table: str) -> bool:
    """Replace the block between the coverage markers in ``readme``; return True if it changed."""
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START_MARKER) + r"\n.*?" + re.escape(END_MARKER), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"{readme} has no {START_MARKER} … {END_MARKER} block")
    updated = pattern.sub(lambda _: f"{START_MARKER}\n{table}{END_MARKER}", text, count=1)
    if updated == text:
        return False
    readme.write_text(updated, encoding="utf-8")
    return True
