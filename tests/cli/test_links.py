"""``book check-links``: every reference URL answers 200 at the same path (contracts/cli.md)."""

import urllib.error
from datetime import date
from email.message import Message
from pathlib import Path
from urllib.request import Request

import pytest
import yaml
from typer.testing import CliRunner

from db_ml_associate.cli import app
from db_ml_associate.cli.links import check_url, link_targets, update_verified
from db_ml_associate.content.loaders import load_book
from helpers.fixture_book import DOCS_URL, FixtureBook

ROOT = Path(__file__).resolve().parents[2]


class FakeResponse:
    def __init__(self, url: str, status: int = 200) -> None:
        self.url, self.status = url, status

    def geturl(self) -> str:
        return self.url

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeOpener:
    """Answers from a table: URL → (status, final URL); HEAD may be refused with 405."""

    def __init__(self, answers: dict[str, tuple[int, str]], refuse_head: bool = False) -> None:
        self.answers, self.refuse_head = answers, refuse_head
        self.methods: list[str] = []

    def open(self, request: Request, timeout: float = 0) -> FakeResponse:
        method = request.get_method()
        self.methods.append(method)
        url = request.full_url
        if method == "HEAD" and self.refuse_head:
            raise urllib.error.HTTPError(url, 405, "Method Not Allowed", Message(), None)
        status, final = self.answers.get(url, (404, url))
        if status >= 400:
            raise urllib.error.HTTPError(url, status, "error", Message(), None)
        return FakeResponse(final, status)


URL = "https://docs.databricks.com/aws/en/machine-learning/"


def test_ok_at_same_path() -> None:
    result = check_url(URL, FakeOpener({URL: (200, URL)}))
    assert result.ok and result.status == 200


def test_trailing_slash_and_fragment_are_the_same_path() -> None:
    assert check_url(URL + "#intro", FakeOpener({URL: (200, URL.rstrip("/"))})).ok


def test_redirect_to_other_path_fails() -> None:
    moved = "https://docs.databricks.com/aws/en/machine-learning/new-place"
    result = check_url(URL, FakeOpener({URL: (200, moved)}))
    assert not result.ok
    assert moved in result.message


def test_client_and_server_errors_fail() -> None:
    assert not check_url(URL, FakeOpener({URL: (404, URL)})).ok
    assert not check_url(URL, FakeOpener({URL: (503, URL)})).ok


def test_get_fallback_when_head_is_refused() -> None:
    opener = FakeOpener({URL: (200, URL)}, refuse_head=True)
    assert check_url(URL, opener).ok
    assert opener.methods == ["HEAD", "GET"]


def test_network_errors_fail_with_a_message() -> None:
    class Broken:
        def open(self, request: Request, timeout: float = 0) -> FakeResponse:
            raise urllib.error.URLError("no route to host")

    result = check_url(URL, Broken())
    assert not result.ok
    assert "no route to host" in result.message


def test_targets_include_references_and_exam_guide(fixture_book: FixtureBook) -> None:
    targets = link_targets(load_book(fixture_book.root))
    assert DOCS_URL in targets
    assert "https://example.com/guide.pdf" in targets


def test_update_verified_changes_only_ok_urls(fixture_book: FixtureBook) -> None:
    path = fixture_book.book / "references.yaml"
    update_verified(path, {DOCS_URL}, date(2027, 1, 2))
    (reference,) = yaml.safe_load(path.read_text())
    assert reference["last_verified"] == date(2027, 1, 2)
    update_verified(path, set(), date(2027, 2, 3))
    assert yaml.safe_load(path.read_text())[0]["last_verified"] == date(2027, 1, 2)


def test_cli_reports_failures_with_exit_code_one(fixture_book: FixtureBook, monkeypatch: pytest.MonkeyPatch) -> None:
    import db_ml_associate.cli.links as links

    monkeypatch.setattr(links, "default_opener", lambda: FakeOpener({DOCS_URL: (200, DOCS_URL)}))
    result = CliRunner().invoke(app, ["book", "check-links", "--root", str(fixture_book.root)])
    assert result.exit_code == 1  # the exam guide URL answers 404 in the fake
    assert "example.com/guide.pdf" in result.output


@pytest.mark.network
def test_all_book_references_are_reachable() -> None:  # pragma: no cover - network
    from db_ml_associate.cli.links import check_links

    failures = [str(result) for result in check_links(load_book(ROOT)) if not result.ok]
    assert failures == []
