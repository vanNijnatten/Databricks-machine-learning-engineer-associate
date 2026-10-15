"""Render a chapter's review questions from ``book/questions/chNN.yaml`` into the chapter.

A chapter marks the place with ``<!-- questions: chNN -->``. ``book sync-snippets`` writes the
rendered questions after the marker, closed by ``<!-- /questions -->``; answers and explanations
are hidden in a ``<details>`` block so learners can test themselves first.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from db_ml_associate.content.models import Question

START_MARKER = re.compile(r"^<!--\s*questions:\s*(ch\d{2})\s*-->\s*$")
END_MARKER = "<!-- /questions -->"


def _render_question(number: int, question: Question) -> str:
    options = "\n".join(f"- **{letter}.** {text}" for letter, text in question.options.items())
    count = len(question.answer)
    instruction = "*Select one answer.*" if count == 1 else f"*Select {count} answers.*"
    distractors = "\n".join(
        f"- **{letter}**: {question.distractors[letter]}"
        for letter in question.options
        if letter not in question.answer and letter in question.distractors
    )
    return (
        f"### Question {number}\n\n"
        f"{question.stem}\n\n"
        f"{options}\n\n"
        f"{instruction}\n\n"
        "<details>\n<summary>Answer</summary>\n\n"
        f"**Correct: {', '.join(question.answer)}** — {question.explanation}\n\n"
        f"Why the other options are wrong:\n\n{distractors}\n\n"
        f"Objective {question.objective} · [Revisit]({question.chapter_ref})\n\n"
        "</details>\n"
    )


def render_review_questions(questions: Sequence[Question]) -> str:
    """Render questions as numbered Markdown sections with collapsible answers."""
    return "\n".join(_render_question(number, question) for number, question in enumerate(questions, start=1))


@dataclass
class QuestionSync:
    """Result of synchronising question blocks in one Markdown text."""

    text: str
    stale: bool


def sync_questions(text: str, questions: Mapping[str, Sequence[Question]]) -> QuestionSync:
    """Replace every ``<!-- questions: chNN -->`` block in ``text`` with freshly rendered questions."""
    lines = text.split("\n")
    if not any(START_MARKER.match(line) for line in lines):
        return QuestionSync(text=text, stale=False)
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        output.append(line)
        index += 1
        match = START_MARKER.match(line)
        if not match:
            continue
        end = next(
            (i for i in range(index, len(lines)) if lines[i].strip() == END_MARKER),
            None,
        )
        if end is not None:
            index = end + 1
        rendered = render_review_questions(questions.get(match.group(1), ()))
        output.extend(["", *rendered.rstrip("\n").split("\n"), END_MARKER])
    synced = "\n".join(output)
    if not synced.endswith("\n"):
        synced += "\n"
    return QuestionSync(text=synced, stale=synced != (text if text.endswith("\n") else text + "\n"))
