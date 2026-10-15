"""Every failure condition of ``book check`` (contracts/cli.md) is detected."""

import re
from datetime import date, timedelta

import pytest
from typer.testing import CliRunner

from db_ml_associate.cli import app
from db_ml_associate.cli.book_check import run_checks
from db_ml_associate.content.models import Issue
from helpers.fixture_book import FixtureBook, question

ISSUE_FORMAT = re.compile(r"^.+:\d+ — .+$")


def errors(issues: list[Issue]) -> list[str]:
    return [str(issue) for issue in issues if issue.severity == "error"]


def warnings(issues: list[Issue]) -> list[str]:
    return [str(issue) for issue in issues if issue.severity == "warning"]


def has(messages: list[str], *fragments: str) -> bool:
    return any(all(fragment in message for fragment in fragments) for message in messages)


def test_valid_fixture_has_no_errors(fixture_book: FixtureBook) -> None:
    assert errors(run_checks(fixture_book.root)) == []


def test_issues_are_formatted_as_file_line_message(fixture_book: FixtureBook) -> None:
    fixture_book.edit_chapter("## Pitfalls", "## Traps")
    for message in errors(run_checks(fixture_book.root)):
        assert ISSUE_FORMAT.match(message), message


class TestObjectives:
    def test_wrong_objective_count(self, fixture_book: FixtureBook) -> None:
        data = fixture_book.read_yaml("exam/objectives.yaml")
        assert isinstance(data, dict)
        data["objectives"].pop()
        fixture_book.write_yaml("exam/objectives.yaml", data)
        assert has(errors(run_checks(fixture_book.root)), "objectives.yaml", "48")


class TestCoverage:
    def _drop_objective_s4_06(self, fixture_book: FixtureBook) -> None:
        fixture_book.edit_chapter(", S4.06 -->", " -->")
        fixture_book.edit_chapter(", S4.06]\nprerequisites", "]\nprerequisites")
        bank = fixture_book.read_yaml("questions/ch01.yaml")
        assert isinstance(bank, list)
        fixture_book.write_yaml("questions/ch01.yaml", [q for q in bank if q["objective"] != "S4.06"])

    def test_uncovered_objective_is_a_warning(self, fixture_book: FixtureBook) -> None:
        self._drop_objective_s4_06(fixture_book)
        issues = run_checks(fixture_book.root)
        assert has(warnings(issues), "S4.06", "chapter section")
        assert not has(errors(issues), "S4.06")

    def test_uncovered_objective_is_an_error_when_strict(self, fixture_book: FixtureBook) -> None:
        self._drop_objective_s4_06(fixture_book)
        assert has(errors(run_checks(fixture_book.root, strict=True)), "S4.06")

    def test_practical_objective_without_tutorial_step(self, fixture_book: FixtureBook) -> None:
        fixture_book.edit_chapter("objectives: [S1.01, S1.04", "objectives: [S1.04")
        assert has(warnings(run_checks(fixture_book.root)), "S1.01", "tutorial")


class TestChapters:
    def test_missing_required_heading(self, fixture_book: FixtureBook) -> None:
        fixture_book.edit_chapter("## Pitfalls", "## Traps")
        assert has(errors(run_checks(fixture_book.root)), "01-platform.md", "Pitfalls")

    def test_heading_order(self, fixture_book: FixtureBook) -> None:
        fixture_book.edit_chapter("## Summary\n\nShort summary.\n\n", "")
        fixture_book.edit_chapter("## Pitfalls", "## Summary\n\nShort summary.\n\n## Pitfalls")
        assert has(errors(run_checks(fixture_book.root)), "order")

    def test_empty_objectives(self, fixture_book: FixtureBook) -> None:
        text = fixture_book.chapter.read_text()
        fixture_book.chapter.write_text(re.sub(r"^objectives: \[.*\]$", "objectives: []", text, count=1, flags=re.M))
        assert has(errors(run_checks(fixture_book.root)), "01-platform.md", "objective")

    def test_too_few_learning_objectives(self, fixture_book: FixtureBook) -> None:
        fixture_book.edit_chapter("- Understand two.\n- Understand three.\n", "")
        assert has(errors(run_checks(fixture_book.root)), "learning objectives")

    def test_fewer_than_eight_review_questions(self, fixture_book: FixtureBook) -> None:
        bank = fixture_book.read_yaml("questions/ch01.yaml")
        assert isinstance(bank, list)
        fixture_book.write_yaml("questions/ch01.yaml", bank[:7])
        assert has(errors(run_checks(fixture_book.root)), "review questions")

    def test_introduction_and_appendices_are_exempt(self, fixture_book: FixtureBook) -> None:
        (fixture_book.book / "introduction.md").write_text("# Introduction\n\nWelcome.\n")
        (fixture_book.book / "appendix-a-exam-vs-platform.md").write_text("# Appendix A\n\nTable.\n")
        assert errors(run_checks(fixture_book.root)) == []

    def test_missing_tutorial_notebook(self, fixture_book: FixtureBook) -> None:
        (fixture_book.root / "notebooks" / "ch01_platform.py").unlink()
        assert has(warnings(run_checks(fixture_book.root)), "ch01_platform.py", "open work")
        assert has(errors(run_checks(fixture_book.root, strict=True)), "ch01_platform.py")

    def test_chapter_without_tutorial_is_open_work(self, fixture_book: FixtureBook) -> None:
        text = fixture_book.chapter.read_text()
        start = text.index("tutorial:")
        end = text.index("---", start)
        fixture_book.chapter.write_text(text[:start] + text[end:])
        assert has(warnings(run_checks(fixture_book.root)), "no tutorial yet")
        assert has(errors(run_checks(fixture_book.root, strict=True)), "no tutorial yet")


class TestQuestions:
    def _bank(self, fixture_book: FixtureBook) -> list[dict[str, object]]:
        bank = fixture_book.read_yaml("questions/ch01.yaml")
        assert isinstance(bank, list)
        return bank

    def test_duplicate_id(self, fixture_book: FixtureBook) -> None:
        bank = self._bank(fixture_book)
        bank[1]["id"] = bank[0]["id"]
        fixture_book.write_yaml("questions/ch01.yaml", bank)
        assert has(errors(run_checks(fixture_book.root)), "duplicate", "Q-01-001")

    def test_answer_type_mismatch(self, fixture_book: FixtureBook) -> None:
        bank = self._bank(fixture_book)
        bank[0]["type"] = "multiple"
        fixture_book.write_yaml("questions/ch01.yaml", bank)
        assert has(errors(run_checks(fixture_book.root)), "ch01.yaml", "multiple")

    def test_missing_distractor_explanation(self, fixture_book: FixtureBook) -> None:
        bank = self._bank(fixture_book)
        bank[0]["distractors"] = {"B": "No."}
        fixture_book.write_yaml("questions/ch01.yaml", bank)
        assert has(errors(run_checks(fixture_book.root)), "distractor")

    def test_copy_of_official_sample_question(self, fixture_book: FixtureBook) -> None:
        bank = self._bank(fixture_book)
        bank[0]["stem"] = (
            "A data scientist wants to create a feature table to use in their models. They are working in a "
            "workspace with Unity Catalog enabled and want this feature table to be stored and governed by it. "
            "What is the correct way of creating this feature table?"
        )
        fixture_book.write_yaml("questions/ch01.yaml", bank)
        assert has(errors(run_checks(fixture_book.root)), "official sample")

    def test_broken_chapter_ref(self, fixture_book: FixtureBook) -> None:
        bank = self._bank(fixture_book)
        bank[0]["chapter_ref"] = "01-platform.md#no-such-anchor"
        fixture_book.write_yaml("questions/ch01.yaml", bank)
        assert has(errors(run_checks(fixture_book.root)), "no-such-anchor")


class TestExams:
    def _practice_bank(self, exam_id: str, count: int) -> list[dict[str, object]]:
        ids = [f"S{s}.{n:02d}" for s, total in {1: 18, 2: 9, 3: 15, 4: 6}.items() for n in range(1, total + 1)]
        return [question(f"Q-{exam_id}-{i:03d}", ids[i - 1], pool="practice") for i in range(1, count + 1)]

    def test_exam_with_47_questions(self, fixture_book: FixtureBook) -> None:
        bank = self._practice_bank("PE1", 47)
        fixture_book.write_yaml("questions/practice-pe1.yaml", bank)
        fixture_book.write_yaml(
            "exams/PE1.yaml",
            {"id": "PE1", "minutes": 90, "questions": [q["id"] for q in bank]},
        )
        assert has(errors(run_checks(fixture_book.root)), "PE1", "48")

    def test_valid_exam(self, fixture_book: FixtureBook) -> None:
        bank = self._practice_bank("PE1", 48)
        fixture_book.write_yaml("questions/practice-pe1.yaml", bank)
        fixture_book.write_yaml(
            "exams/PE1.yaml",
            {"id": "PE1", "minutes": 90, "questions": [q["id"] for q in bank]},
        )
        assert errors(run_checks(fixture_book.root)) == []


class TestTotals:
    def test_small_bank_is_a_warning(self, fixture_book: FixtureBook) -> None:
        assert has(warnings(run_checks(fixture_book.root)), "200")

    def test_small_bank_is_an_error_when_strict(self, fixture_book: FixtureBook) -> None:
        assert has(errors(run_checks(fixture_book.root, strict=True)), "200")

    def test_section_proportions(self, fixture_book: FixtureBook) -> None:
        bank = fixture_book.read_yaml("questions/ch01.yaml")
        assert isinstance(bank, list)
        extra = [question(f"Q-01-{n:03d}", "S4.01") for n in range(100, 140)]
        fixture_book.write_yaml("questions/ch01.yaml", bank + extra)
        assert has(warnings(run_checks(fixture_book.root)), "section 4", "share")


class TestSnippetsAndLinks:
    def test_stale_snippet(self, fixture_book: FixtureBook) -> None:
        fixture_book.edit_chapter("value = 1\n```", "value = 2\n```")
        assert has(errors(run_checks(fixture_book.root)), "snippet")

    def test_broken_relative_link(self, fixture_book: FixtureBook) -> None:
        fixture_book.edit_chapter("[summary](#summary)", "[summary](../notes/missing.md)")
        assert has(errors(run_checks(fixture_book.root)), "missing.md")

    def test_link_to_unwritten_chapter_is_planned_work(self, fixture_book: FixtureBook) -> None:
        fixture_book.edit_chapter("[summary](#summary)", "[next chapter](02-next-chapter.md)")
        issues = run_checks(fixture_book.root)
        assert has(warnings(issues), "planned page", "02-next-chapter.md")
        assert not has(errors(issues), "02-next-chapter.md")
        assert has(errors(run_checks(fixture_book.root, strict=True)), "02-next-chapter.md")

    def test_broken_anchor(self, fixture_book: FixtureBook) -> None:
        fixture_book.edit_chapter("[summary](#summary)", "[summary](#nowhere)")
        assert has(errors(run_checks(fixture_book.root)), "#nowhere")

    def test_links_inside_code_blocks_are_ignored(self, fixture_book: FixtureBook) -> None:
        fixture_book.edit_chapter(
            "## Pitfalls",
            "<!-- illustrative -->\n```markdown\n[x](missing.md)\n```\n\n## Pitfalls",
        )
        assert not has(errors(run_checks(fixture_book.root)), "missing.md")


class TestReferences:
    def test_unregistered_docs_url(self, fixture_book: FixtureBook) -> None:
        fixture_book.edit_chapter(
            "See the [ML docs]",
            "Also [AutoML](https://docs.databricks.com/aws/en/machine-learning/automl/). See the [ML docs]",
        )
        assert has(errors(run_checks(fixture_book.root)), "references.yaml", "automl")

    def test_stale_reference(self, fixture_book: FixtureBook) -> None:
        references = fixture_book.read_yaml("references.yaml")
        assert isinstance(references, list)
        references[0]["last_verified"] = date.today() - timedelta(days=120)
        fixture_book.write_yaml("references.yaml", references)
        assert has(warnings(run_checks(fixture_book.root)), "ml-overview", "90")
        assert has(errors(run_checks(fixture_book.root, strict=True)), "ml-overview", "90")

    def test_chapter_cites_unknown_reference(self, fixture_book: FixtureBook) -> None:
        fixture_book.edit_chapter("references: [ml-overview]", "references: [ml-overview, nope]")
        assert has(errors(run_checks(fixture_book.root)), "nope")


class TestPlatformChanges:
    def test_missing_required_change(self, fixture_book: FixtureBook) -> None:
        changes = fixture_book.read_yaml("exam/platform-changes.yaml")
        assert isinstance(changes, list)
        fixture_book.write_yaml(
            "exam/platform-changes.yaml",
            [c for c in changes if "Hyperopt" not in c["exam_term"]],
        )
        assert has(errors(run_checks(fixture_book.root)), "Hyperopt")


class TestPlans:
    def test_prerequisite_order_warning(self, fixture_book: FixtureBook) -> None:
        second = (
            fixture_book.chapter.read_text()
            .replace("chapter: 1", "chapter: 2")
            .replace("prerequisites: []", "prerequisites: [1]")
        )
        (fixture_book.book / "02-second.md").write_text(second)
        fixture_book.write_yaml(
            "plans/standard-21.yaml",
            {
                "track": "standard",
                "days": [
                    {
                        "day": 1,
                        "goals": ["g"],
                        "items": [{"chapter": 2}],
                        "self_check": "s",
                    },
                    {
                        "day": 2,
                        "goals": ["g"],
                        "items": [{"chapter": 1}],
                        "self_check": "s",
                    },
                ],
            },
        )
        assert has(warnings(run_checks(fixture_book.root)), "chapter 2", "prerequisite")


class TestRuntime:
    def test_pin_mismatch(self, fixture_book: FixtureBook) -> None:
        (fixture_book.root / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["pandas==2.3.0"]\n')
        assert has(errors(run_checks(fixture_book.root)), "pandas", "2.2.3")


class TestCli:
    def test_exit_code_zero_on_valid_book(self, fixture_book: FixtureBook) -> None:
        result = CliRunner().invoke(app, ["book", "check", "--root", str(fixture_book.root)])
        assert result.exit_code == 0, result.output
        assert "0 errors" in result.output

    def test_exit_code_one_on_errors(self, fixture_book: FixtureBook) -> None:
        fixture_book.edit_chapter("## Pitfalls", "## Traps")
        result = CliRunner().invoke(app, ["book", "check", "--root", str(fixture_book.root)])
        assert result.exit_code == 1

    def test_strict_turns_warnings_into_errors(self, fixture_book: FixtureBook) -> None:
        result = CliRunner().invoke(app, ["book", "check", "--root", str(fixture_book.root), "--strict"])
        assert result.exit_code == 1

    @pytest.mark.parametrize("args", [["book", "check", "--no-such-option"], ["book", "nonsense"]])
    def test_usage_error_exit_code_two(self, args: list[str]) -> None:
        assert CliRunner().invoke(app, args).exit_code == 2
