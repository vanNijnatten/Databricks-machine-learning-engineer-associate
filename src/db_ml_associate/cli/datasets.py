"""``datasets generate``: write a synthetic tutorial dataset to CSV or Parquet."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from db_ml_associate.datasets import (
    DEFAULT_SEED,
    make_customer_churn,
    make_house_prices,
)

datasets_app = typer.Typer(no_args_is_help=True, help="Generate the synthetic tutorial datasets.")


class DatasetName(StrEnum):
    """Datasets that can be generated."""

    CHURN = "churn"
    HOUSE_PRICES = "house_prices"


@datasets_app.command("generate")
def generate(
    name: Annotated[DatasetName, typer.Argument(help="Dataset to generate.")],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output file (.csv or .parquet).", dir_okay=False),
    ],
    rows: Annotated[
        int | None,
        typer.Option("--rows", min=1, help="Number of rows (default: dataset default)."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = DEFAULT_SEED,
) -> None:
    """Write a dataset; the defaults reproduce the committed files in ``data/``."""
    if out.suffix not in (".csv", ".parquet"):
        raise typer.BadParameter("output must end in .csv or .parquet", param_hint="--out")
    generator = make_customer_churn if name is DatasetName.CHURN else make_house_prices
    frame: pd.DataFrame = generator(seed=seed) if rows is None else generator(n_rows=rows, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".csv":
        frame.to_csv(out, index=False)
    else:
        frame.to_parquet(out, index=False)
    typer.echo(f"wrote {len(frame)} rows to {out}")
