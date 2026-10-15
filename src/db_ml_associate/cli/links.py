"""``book check-links``: fetch every reference URL and the exam guide (network; contracts/cli.md).

A URL passes when it answers HTTP 200 at the same path, after following redirects. A redirect to a
different path usually means the docs page moved: update ``references.yaml`` (status ``moved``, then
the new URL) and re-read the page. ``--update-verified`` sets ``last_verified`` to today for passing
URLs; re-reading the content of each page remains a manual step.
"""

import re
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from db_ml_associate.content.loaders import Book

TIMEOUT_SECONDS = 30
USER_AGENT = "db-ml-associate-link-check/0.1 (+https://github.com/)"
HTTP_OK = 200
HTTP_ERROR = 400
REFERENCE_URL = re.compile(r"^\s*-?\s*url:\s*(\S+)\s*$")
LAST_VERIFIED = re.compile(r"^(\s*last_verified:\s*)\S+(\s*)$")


class Response(Protocol):
    """The parts of an HTTP response the check needs."""

    status: int

    def geturl(self) -> str: ...

    def __enter__(self) -> "Response": ...

    def __exit__(self, *args: object) -> None: ...


class Opener(Protocol):
    """Something that opens a request, like ``urllib.request.build_opener()``."""

    def open(self, request: urllib.request.Request, timeout: float = ...) -> Response: ...


@dataclass(frozen=True)
class LinkResult:
    """Outcome of checking one URL."""

    url: str
    ok: bool
    status: int | None
    final_url: str | None
    message: str = ""

    def __str__(self) -> str:
        return f"{self.url} — {self.message or self.status}"


def default_opener() -> Opener:
    """A urllib opener that follows redirects (patched in tests)."""
    return urllib.request.build_opener()  # type: ignore[return-value]


def _path(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.netloc}{parts.path}".rstrip("/")


def _request(url: str, method: str) -> urllib.request.Request:
    return urllib.request.Request(url.split("#", 1)[0], method=method, headers={"User-Agent": USER_AGENT})


def _fetch(opener: Opener, url: str) -> tuple[int, str]:
    try:
        with opener.open(_request(url, "HEAD"), timeout=TIMEOUT_SECONDS) as response:
            return response.status, response.geturl()
    except urllib.error.HTTPError as error:
        if error.code < HTTP_ERROR:
            raise
    with opener.open(_request(url, "GET"), timeout=TIMEOUT_SECONDS) as response:  # some servers refuse HEAD
        return response.status, response.geturl()


def check_url(url: str, opener: Opener) -> LinkResult:
    """HEAD the URL (GET if HEAD is refused); pass on 200 at the same path."""
    try:
        status, final_url = _fetch(opener, url)
    except urllib.error.HTTPError as error:
        return LinkResult(url, False, error.code, None, f"HTTP {error.code}")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        return LinkResult(url, False, None, None, f"request failed: {reason}")
    if status != HTTP_OK:
        return LinkResult(url, False, status, final_url, f"HTTP {status}")
    if _path(final_url) != _path(url.split("#", 1)[0]):
        return LinkResult(url, False, status, final_url, f"redirects to a different page: {final_url}")
    return LinkResult(url, True, status, final_url, "ok")


def link_targets(book: Book) -> list[str]:
    """Every reference URL plus the official exam guide, without duplicates, in file order."""
    urls = [reference.url for reference in book.references]
    if book.guide is not None:
        urls.append(book.guide.url)
    return list(dict.fromkeys(urls))


def check_links(book: Book, opener: Opener | None = None) -> list[LinkResult]:
    """Check every link target of the book."""
    opener = opener or default_opener()
    return [check_url(url, opener) for url in link_targets(book)]


def update_verified(references_file: Path, ok_urls: Iterable[str], today: date) -> int:
    """Set ``last_verified: today`` for the references whose URL passed; return how many changed.

    Edits the file line by line so that comments and formatting are kept.
    """
    passing = set(ok_urls)
    lines = references_file.read_text(encoding="utf-8").splitlines(keepends=True)
    current_url: str | None = None
    changed = 0
    for index, line in enumerate(lines):
        url = REFERENCE_URL.match(line)
        if url:
            current_url = url.group(1)
            continue
        verified = LAST_VERIFIED.match(line.rstrip("\n"))
        if verified and current_url in passing:
            lines[index] = f"{verified.group(1)}{today.isoformat()}{verified.group(2)}\n"
            changed += 1
    references_file.write_text("".join(lines), encoding="utf-8")
    return changed
