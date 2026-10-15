"""Keep code blocks in the book identical to marked regions of tested source files.

Source files mark a region with ``# --8<-- [start:name]`` and ``# --8<-- [end:name]``. A chapter
embeds it with an HTML comment directly above a fenced code block::

    <!-- snippet: src/db_ml_associate/ch07_cleaning/impute.py#impute_median -->
    ```python
    ...
    ```

``sync`` rewrites such blocks from the source; ``--check`` only reports stale ones.
"""

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from db_ml_associate.cli.questions import sync_questions
from db_ml_associate.content.loaders import ContentError, load_questions
from db_ml_associate.content.models import Issue, Question

SNIPPET_COMMENT = re.compile(r"^\s*<!--\s*snippet:\s*(\S+?)#([\w\-]+)\s*-->\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


class SnippetError(Exception):
    """A snippet region could not be found."""


def _marker(kind: str, name: str) -> re.Pattern[str]:
    return re.compile(rf"^\s*(?:#|--)\s*--8<--\s*\[{kind}:{re.escape(name)}\]\s*$")  # Python or SQL comment


def extract_region(source: str, name: str) -> str:
    """Return the dedented lines between the start and end markers of region ``name``."""
    lines = source.splitlines()
    start = next((i for i, line in enumerate(lines) if _marker("start", name).match(line)), None)
    if start is None:
        raise SnippetError(f"region {name!r} not found")
    end = next(
        (i for i in range(start + 1, len(lines)) if _marker("end", name).match(lines[i])),
        None,
    )
    if end is None:
        raise SnippetError(f"region {name!r} has no end marker")
    return textwrap.dedent("\n".join(lines[start + 1 : end])) + "\n"


@dataclass
class SyncResult:
    """The synchronised Markdown plus the blocks that were stale or broken (1-based lines)."""

    text: str
    stale: list[tuple[int, str]] = field(default_factory=list)
    errors: list[tuple[int, str]] = field(default_factory=list)


def _expected_lines(root: Path, reference: str, name: str) -> list[str]:
    source = root / reference
    if not source.is_file():
        raise SnippetError(f"snippet source {reference} not found")
    return extract_region(source.read_text(encoding="utf-8"), name).rstrip("\n").split("\n")


def sync_markdown(text: str, root: Path) -> SyncResult:
    """Rewrite every snippet block in ``text`` from its source region under ``root``."""
    lines = text.split("\n")
    result = SyncResult(text=text)
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        output.append(line)
        match = SNIPPET_COMMENT.match(line)
        index += 1
        if not match:
            continue
        reference = f"{match.group(1)}#{match.group(2)}"
        fence = FENCE.match(lines[index]) if index < len(lines) else None
        if fence is None:
            result.errors.append((index, f"snippet {reference} must be followed by a fenced code block"))
            continue
        close = next(
            (i for i in range(index + 1, len(lines)) if lines[i].strip() == fence.group(1)),
            None,
        )
        if close is None:
            result.errors.append((index, f"snippet {reference}: code block is not closed"))
            continue
        try:
            expected = _expected_lines(root, match.group(1), match.group(2))
        except SnippetError as error:
            result.errors.append((index, f"snippet {reference}: {error}"))
            output.extend(lines[index : close + 1])
            index = close + 1
            continue
        if lines[index + 1 : close] != expected:
            result.stale.append((index, f"snippet {reference} is out of date"))
        output.append(lines[index])
        output.extend(expected)
        output.append(lines[close])
        index = close + 1
    result.text = "\n".join(output)
    return result


def sync_book(root: Path, check: bool = False) -> list[Issue]:
    """Synchronise snippets and rendered review questions in all Markdown files under ``root/book``.

    With ``check=True`` nothing is written and stale blocks are reported as errors; otherwise
    stale blocks are rewritten. Broken snippet references are always reported.
    """
    issues: list[Issue] = []
    review_questions = _review_questions(root)
    for path in sorted((root / "book").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        result = sync_markdown(text, root)
        rendered = sync_questions(result.text, review_questions)
        shown = path.relative_to(root).as_posix()
        issues.extend(Issue(shown, line, message) for line, message in result.errors)
        if check:
            issues.extend(Issue(shown, line, message) for line, message in result.stale)
            if rendered.stale:
                issues.append(Issue(shown, 1, "rendered review questions are out of date"))
        elif rendered.text != text:
            path.write_text(rendered.text, encoding="utf-8")
    return issues


def _review_questions(root: Path) -> dict[str, list[Question]]:
    """Review-pool questions per ``chNN`` key, read from ``book/questions/chNN.yaml``."""
    questions: dict[str, list[Question]] = {}
    for path in sorted((root / "book" / "questions").glob("ch[0-9][0-9].yaml")):
        try:
            loaded = load_questions(path, root)
        except ContentError:
            continue  # reported by book check
        questions[path.stem] = [question for question in loaded if question.pool == "review"]
    return questions
