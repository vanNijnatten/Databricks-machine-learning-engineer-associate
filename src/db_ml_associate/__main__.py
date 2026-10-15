"""Entry point for ``db-ml-associate`` and ``python -m db_ml_associate``."""

from db_ml_associate.cli import app


def main() -> None:
    """Run the command-line interface."""
    app()


if __name__ == "__main__":
    main()
