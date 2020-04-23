#!/usr/bin/env python3

import time
import toml
import click
import numpy as np
from tqdm import tqdm
from pathlib import Path
from random import randrange
from multiprocessing import Pool
from typing import Any, List, Dict
from click import clear, echo, style, secho

from utils import get_image_data, export_image, select_channel, predict_classification

conf: Dict[str, Any] = {}


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = []
    dataset_copy = list(dataset)
    fold_size = len(dataset) // n_folds
    for _ in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def k_nearest_neighbors(train, test, num_neighbors):
    predictions = []
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return predictions


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = []
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def apply_operations(file: Path) -> str:
    """
    1. From segmented cell images (choose any segmentation technique you prefer) 
        extract AT LEAST four distinctive features+ 
        assign class label according to cell type  from  documentation  (as  last  column) 
        –there  should  be  seven  distinctive classes. 
    2.  Save  new  dataset  as  a  matrix  with  columns  representing  features  
        (and  last column for class label) and rows representing individual cells. 
        Use .csv format 
    3.  Implement  (not  use  an  existing implementation)  a  k-NN  classifier  with Euclidean distance. 
    4. Implement 10 fold cross-validation. 
    5. Perform classification of cells using 10 fold cross-validation and k-NN classifier. 
        Report classification accuracy (averaged among all 10 folds of cross validation) 
    6. Evaluate the performance of parameter k on the classification accuracy 
        –run independent experiments with AT LEAST five different values of k and compare the results
    """

    try:
        img = get_image_data(file)
        img = select_channel(img, conf["COLOR_CHANNEL"])

    except Exception as e:
        return style(f"[ERROR] {file.stem} has an issue: {e}", fg="red")

    return style(f"{f'[INFO:{file.stem}]':15}", fg="green")


def parallel_operations(files: List[Path]):
    """
    Batch operates on a set of images in a multiprocess pool
    """

    echo(
        style("[INFO] ", fg="green")
        + f"initilizing process pool (number of processes: {conf['NUM_OF_PROCESSES']})"
    )
    echo(style("[INFO] ", fg="green") + "compiling...")
    with Pool(conf["NUM_OF_PROCESSES"]) as p:
        with tqdm(total=len(files)) as pbar:
            for res in tqdm(p.imap(apply_operations, files)):
                pbar.write(res + f" finished...")
                pbar.update()


@click.command()
@click.option(
    "config_location",
    "-c",
    "--config",
    envvar="CMSC630_CONFIG",
    type=click.Path(exists=True),
    default="config.toml",
    show_default=True,
)
def main(config_location: str):
    global conf
    conf = toml.load(config_location)

    clear()

    base_path: Path = Path(conf["DATA_DIR"])

    files: List = list(base_path.glob(f"*{conf['FILE_EXTENSION']}"))
    echo(
        style("[INFO] ", fg="green")
        + f"image directory: {str(base_path)}; {len(files)} images found"
    )

    Path(conf["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    # [!!!] Only for development
    # DATA_SUBSET = 1
    # files = files[:DATA_SUBSET]

    t0 = time.time()
    parallel_operations(files)
    t_delta = time.time() - t0

    print()
    secho(f"Total time: {t_delta:.2f} s", fg="green")


if __name__ == "__main__":
    main()
