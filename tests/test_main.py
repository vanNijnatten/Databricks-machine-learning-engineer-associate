"""The package entry point dispatches to the typer application."""

import pytest
from pytest_mock.plugin import MockerFixture
from typer.testing import CliRunner

from db_ml_associate import __version__, main
from db_ml_associate.cli import app


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_main_runs_the_cli_app(mocker: MockerFixture) -> None:
    run = mocker.patch("db_ml_associate.__main__.app")
    main()
    run.assert_called_once_with()


def test_help_lists_command_groups() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in ("book", "datasets"):
        assert group in result.output


@pytest.mark.parametrize("group", ["book", "datasets"])
def test_group_help(group: str) -> None:
    assert CliRunner().invoke(app, [group, "--help"]).exit_code == 0
