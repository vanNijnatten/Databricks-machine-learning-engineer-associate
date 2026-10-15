"""The ``db-ml-associate`` command-line interface (contracts/cli.md).

Exit codes: 0 success, 1 validation failure, 2 usage error.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from db_ml_associate.cli.appendix import render_appendix_a
from db_ml_associate.cli.book_check import run_checks
from db_ml_associate.cli.coverage import (
    compute_coverage,
    render_coverage_table,
    write_coverage,
)
from db_ml_associate.cli.datasets import datasets_app
from db_ml_associate.cli.snippets import sync_book
from db_ml_associate.content.loaders import load_book

console = Console(highlight=False, soft_wrap=True)

app = typer.Typer(
    no_args_is_help=True,
    help="Study guide tooling for the Databricks ML Associate exam.",
)
book_app = typer.Typer(no_args_is_help=True, help="Validate and maintain the book.")
app.add_typer(book_app, name="book")
app.add_typer(datasets_app, name="datasets")

RootOption = Annotated[Path, typer.Option("--root", help="Repository root.", file_okay=False)]


@book_app.command("check")
def book_check(
    root: RootOption = Path("."),
    strict: Annotated[bool, typer.Option("--strict", help="Treat warnings as errors.")] = False,
) -> None:
    """Validate all book content (coverage, structure, questions, links, references, runtime)."""
    issues = run_checks(root, strict=strict)
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    for issue in errors:
        console.print(f"[red]error[/red]   {issue}", markup=True)
    for issue in warnings:
        console.print(f"[yellow]warning[/yellow] {issue}", markup=True)
    console.print(f"{len(errors)} errors, {len(warnings)} warnings")
    if errors:
        raise typer.Exit(code=1)


@book_app.command("sync-snippets")
def book_sync_snippets(
    root: RootOption = Path("."),
    check: Annotated[bool, typer.Option("--check", help="Report stale blocks without writing.")] = False,
) -> None:
    """Rewrite snippet code blocks in the book from their source regions."""
    issues = sync_book(root, check=check)
    for issue in issues:
        console.print(f"[red]error[/red]   {issue}", markup=True)
    console.print(f"{len(issues)} snippet problems")
    if issues:
        raise typer.Exit(code=1)


@book_app.command("coverage")
def book_coverage(
    root: RootOption = Path("."),
    write: Annotated[bool, typer.Option("--write", help="Update the table in book/README.md.")] = False,
) -> None:
    """Show which chapter sections, tutorial steps and questions cover each objective."""
    table = render_coverage_table(compute_coverage(load_book(root)))
    if write:
        changed = write_coverage(root / "book" / "README.md", table)
        console.print("book/README.md updated" if changed else "book/README.md already up to date")
    else:
        console.print(table, markup=False)


@book_app.command("render-appendix-a")
def book_render_appendix_a(root: RootOption = Path(".")) -> None:
    """Write book/appendix-a-exam-vs-platform.md from book/exam/platform-changes.yaml."""
    target = root / "book" / "appendix-a-exam-vs-platform.md"
    target.write_text(render_appendix_a(load_book(root)), encoding="utf-8")
    console.print(f"wrote {target}")
