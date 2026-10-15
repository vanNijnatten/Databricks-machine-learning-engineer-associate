"""``datasets generate`` writes the same data as the generator functions."""

from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from db_ml_associate.cli import app
from db_ml_associate.datasets import make_customer_churn, make_house_prices


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_generate_churn(tmp_path: Path, suffix: str) -> None:
    out = tmp_path / f"churn{suffix}"
    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "generate",
            "churn",
            "--rows",
            "100",
            "--seed",
            "1",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    written = pd.read_parquet(out) if suffix == ".parquet" else pd.read_csv(out)
    expected = make_customer_churn(n_rows=100, seed=1)
    if suffix == ".csv":
        expected = pd.read_csv(StringIO(expected.to_csv(index=False)))
    pd.testing.assert_frame_equal(written, expected)


def test_generate_house_prices_defaults(tmp_path: Path) -> None:
    out = tmp_path / "houses.parquet"
    result = CliRunner().invoke(app, ["datasets", "generate", "house_prices", "--out", str(out)])
    assert result.exit_code == 0, result.output
    pd.testing.assert_frame_equal(pd.read_parquet(out), make_house_prices())


def test_unknown_dataset_is_a_usage_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["datasets", "generate", "iris", "--out", str(tmp_path / "x.csv")])
    assert result.exit_code == 2


def test_unsupported_extension_is_a_usage_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["datasets", "generate", "churn", "--out", str(tmp_path / "x.json")])
    assert result.exit_code == 2
