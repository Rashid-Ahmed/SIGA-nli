#!/usr/bin/ven siga_nli
import typer
from pathlib import Path

import siga_nli
from siga_nli.config import Config

app = typer.Typer()
config = Config()


@app.command()
def train(output_path: Path = typer.Argument(None, help="Directory to save model and tokenier in ")):
    siga_nli.train(output_path, config)


@app.command()
def evaluate():
    siga_nli.evaluate(config)


if __name__ == "__main__":
    app()
