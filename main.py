#!/usr/bin/env python3

import os
import toml
import click
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from click import clear, echo, style, secho
from typing import Any, List, Dict, Callable, Optional

from utils import (
    save_dataset,
    load_dataset,
    evaluate,
    extract_features,
    predict_label,
    deserialize_label,
    normalize,
)

conf: Dict[str, Any] = {}


@click.group(chain=True)
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
    CMSC 630 Image Analysis Project Part 3
    """

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


@main.command()
@click.pass_context
def preprocess(ctx):
    """
    Perform feature extraction on initial dataset
    """

    """
    1. From segmented cell images (choose any segmentation technique you prefer) 
        extract AT LEAST four distinctive features+ 
        assign class label according to cell type  from  documentation  (as  last  column) 
        –there  should  be  seven  distinctive classes. 
    2.  Save  new  dataset  as  a  matrix  with  columns  representing  features  
        (and  last column for class label) and rows representing individual cells. 
        Use .csv format 
    """

    base_path: Path = Path(conf["DATA_DIR"])

    files: List = list(base_path.glob(f"*{conf['FILE_EXTENSION']}"))
    echo(
        style("[INFO] ", fg="green")
        + f"image directory: {str(base_path)}; {len(files)} images found"
    )

    features = []
    extract_img_features = partial(extract_features, conf)
    echo(
        style("[INFO] ", fg="green")
        + f"initilizing process pool (number of processes: {conf['NUM_OF_PROCESSES']})"
    )
    echo(style("[INFO] ", fg="green") + "compiling...")
    with Pool(conf["NUM_OF_PROCESSES"]) as p:
        with tqdm(total=len(files)) as pbar:
            for res in tqdm(p.imap(extract_img_features, files)):
                pbar.write(res["msg"])
                if len(res["features"]) == 5:
                    features.append(res["features"])
                pbar.update()

    output_file = Path(os.path.join(conf["OUTPUT_DIR"], conf["DATASET_OUT_FILE"]))
    echo(
        style("[INFO] ", fg="green")
        + f"saving preprocessed data to {output_file}; {len(features)} rows"
    )
    norm_dataset = normalize(np.array(features))
    save_dataset(norm_dataset, output_file)


@main.command()
@click.pass_context
def test(ctx):
    """
    Run KNN on dataset to evaluate performance
    """

    """
    3.  Implement  (not  use  an  existing implementation)  a  k-NN  classifier  with Euclidean distance. 
    4. Implement 10 fold cross-validation. 
    5. Perform classification of cells using 10 fold cross-validation and k-NN classifier. 
        Report classification accuracy (averaged among all 10 folds of cross validation) 
    6. Evaluate the performance of parameter k on the classification accuracy 
        –run independent experiments with AT LEAST five different values of k and compare the results
    """

    output_file = Path(os.path.join(conf["OUTPUT_DIR"], conf["DATASET_OUT_FILE"]))

    dataset = load_dataset(output_file)

    if len(dataset) == 0:
        echo(
            style("[ERROR] ", fg="red")
            + "preprocessed dataset not found or is empty: exiting..."
        )
        return

    total_avg = 0
    for k in range(1, int(conf["UPPER_BOUND_K"]) + 1):
        scores = evaluate(dataset, conf["N_FOLDS"], k)
        average = sum(scores) / float(len(scores))
        echo("\n")
        echo(style("[INFO] ", fg="green") + f"k={k}")
        echo(f"\tScores: {['{:.3f}%'.format(score) for score in scores]}")
        echo(f"\tMean Accuracy: {average:.3f}%")

        total_avg += average

    total_avg /= int(conf["UPPER_BOUND_K"])
    echo(style("\n[INFO] ", fg="green") + f"Total Average: {total_avg:.3f}%")


@main.command()
@click.argument("path", nargs=1, type=click.Path(exists=True))
@click.option("k", "-k", "--k-value", default=3, show_default=True)
@click.pass_context
def predict(ctx, path, k):
    """
    Use KNN to perdict the label of a new image
    """

    """
    extract features
    open dataset
    KNN predict with extracted features
    """
    features = extract_features(conf, Path(path))["features"]
    if len(features) != 5:
        return

    features = np.reshape(features, (-1, 5))

    output_file = Path(os.path.join(conf["OUTPUT_DIR"], conf["DATASET_OUT_FILE"]))
    dataset = load_dataset(output_file)
    norm_dataset = normalize(dataset)

    label = predict_label(norm_dataset, features, k)

    label_name = deserialize_label(int(label))

    echo(label_name)


if __name__ == "__main__":
    main()
