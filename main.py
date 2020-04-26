#!/usr/bin/env python3

import time
import toml
import click
import numpy as np
from tqdm import tqdm
from numba import njit
from pathlib import Path
from random import randrange
from multiprocessing import Pool
from click import clear, echo, style, secho
from typing import Any, List, Dict, Callable, Optional

from utils import (
    get_image_data,
    export_image,
    select_channel,
    accuracy_metric,
    get_neighbors,
    histogram_thresholding,
)

conf: Dict[str, Any] = {}


@click.group()
@click.option(
    "config_location",
    "-c",
    "--config",
    envvar="CMSC630_CONFIG",
    type=click.Path(exists=True),
    default="config.toml",
    show_default=True,
)
@click.pass_context
def main(ctx, config_location: Optional[str]) -> None:
    """
    Setup configurations and context before invoking subcommand
    """

    global conf

    try:
        conf = toml.load(config_location)
    except Exception as e:
        secho(f"[ERROR] problem with configuration file {config_location} : {e}")
        return

    Path(conf["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    ctx.conf = conf

    echo(
        style("[INFO] ", fg="green")
        + f"invoking {ctx.invoked_subcommand}"
    )



@main.command()
@click.pass_context
def create_dataset(ctx):
    """
    - Perform feature extraction on initial dataset
    - Save updated features
    """

    base_path: Path = Path(conf["DATA_DIR"])

    files: List = list(base_path.glob(f"*{conf['FILE_EXTENSION']}"))
    echo(
        style("[INFO] ", fg="green")
        + f"image directory: {str(base_path)}; {len(files)} images found"
    )

        # [!!!] Only for development
    DATA_SUBSET = 10
    files = files[:DATA_SUBSET]


@main.command()
@click.pass_context
def test(ctx):
    """
    Run KNN on dataset to evaluate performance
    """
    pass


@main.command()
@click.pass_context
def predict(ctx):
    """
    Use KNN to perdict the label of a new image
    """
    pass


if __name__ == "__main__":
    main()
