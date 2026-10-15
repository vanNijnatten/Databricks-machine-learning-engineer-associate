__version__ = "0.1.0"

__all__ = ["__version__", "main"]


def main() -> None:
    """Run the command-line interface.

    Imported lazily so that tutorial notebooks can import the chapter modules without the CLI's
    dependencies (typer, rich).
    """
    from db_ml_associate.__main__ import main as run_cli

    run_cli()
