
import numpy as np
import pandas as pd
from PIL import Image
from math import sqrt
from numba import njit
from pathlib import Path
from operator import eq

from typing import List, Callable


labels: List[str] = [
    "cyl",
    "inter",
    "let",
    "mod",
    "para",
    "super",
    "svar",
]


def label_deserialize(value: int) -> str:
    return labels[value]


def label_serialize(label: str) -> int:
    return labels.index(label)


def get_image_data(filename: Path) -> np.array:
    """
    Converts a bmp image to a numpy array
    """

    with Image.open(filename) as img:
        return np.array(img)


def export_image(img_arr: np.array, filename: str, conf: dict) -> None:
    """
    Exports a numpy array as a grey scale bmp image
    """
    img = Image.fromarray(img_arr)
    img = img.convert("L")
    img.save(conf["OUTPUT_DIR"] + filename + conf["FILE_EXTENSION"])


def save_dataset(arr: np.array, filename: str) -> None:
    np.savetxt(filename, arr, header="X1,X2,X3,X4,Y", delimiter=",")


def select_channel(img_array: np.array, color: str = "red") -> np.array:
    """
    select_channel isolates a color channel from a RGB image represented as a numpy array.
    """
    if color == "red":
        return img_array[:, :, 0]

    elif color == "green":
        return img_array[:, :, 1]

    elif color == "blue":
        return img_array[:, :, 2]


@njit(fastmath=True)
def histogram(img_array: np.array) -> np.array:
    """
    >> h=zeros(256,1);              OR    >> h=zeros(256,1);
    >> for l = 0 : 255                    >> for l = 0 : 255
         for i = 1 : N                          h(l +1)=sum(sum(A == l));
            for j = 1 : M                    end
                if (A(i,j) == l)          >> bar(0:255,h);
                    h(l +1) = h(l +1) +1;
                end
            end
        end
    end
    >> bar(0:255,h);
    """

    # Create blank histogram
    hist: np.array = np.zeros(256)

    # Get size of pixel array
    image_size: int = len(img_array)

    for pixel_value in range(256):
        for i in range(image_size):

            # Loop through pixels to calculate histogram
            if img_array.flat[i] == pixel_value:
                hist[pixel_value] += 1

    return hist


def distance(row1: np.array, row2: np.array) -> float:
    """
    calculate the Euclidean distance between two vectors
    """
    return np.linalg.norm(row1 - row2)


@njit
def get_neighbors(train: np.array, test_row: np.array, K: int) -> np.array:
    """
    Locate the most similar neighbors
    """
    distances = [(train_row, distance(test_row, train_row)) for train_row in train]
    distances.sort(key=lambda tup: tup[1])

    neighbors = np.array([distances[i][0] for i in range(K)])

    return neighbors


def accuracy_metric(actual, predicted):
    """
    Calculate accuracy percentage
    """
    correct = map(eq, actual, predicted)

    return (sum(correct) / len(correct)) * 100.0


# @njit
def middle_of(hist: np.array, min_count: int = 5) -> int:

    num_bins = len(hist)
    hist_start = 0
    while hist[hist_start] < min_count:
        hist_start += 1  # ignore small counts at start

    hist_end = num_bins - 1
    while hist[hist_end] < min_count:
        hist_end -= 1  # ignore small counts at end

    hist_center = int(
        round(np.average(np.linspace(0, 2 ** 8 - 1, num_bins), weights=hist))
    )
    left = np.sum(hist[hist_start:hist_center])
    right = np.sum(hist[hist_center : hist_end + 1])

    while hist_start < hist_end:
        if left > right:  # left part became heavier
            left -= hist[hist_start]
            hist_start += 1
        else:  # right part became heavier
            right -= hist[hist_end]
            hist_end -= 1
        new_center = int(
            round((hist_end + hist_start) / 2)
        )  # re-center the weighing scale

        if new_center < hist_center:  # move bin to the other side
            left -= hist[hist_center]
            right += hist[hist_center]
        elif new_center > hist_center:
            left += hist[hist_center]
            right -= hist[hist_center]

        hist_center = new_center

    return hist_center


def histogram_thresholding(img_arr: np.array) -> np.array:

    hist = histogram(img_arr)

    middle = middle_of(hist)

    img_copy = img_arr.copy()
    img_copy[img_copy > middle] = 255
    img_copy[img_copy < middle] = 0

    img_copy = img_copy.astype(np.uint8)

    return img_copy.reshape(img_arr.shape)


def cross_validation_split(dataset: np.array, n_folds: int) -> np.array:
    """
    Split a dataset into k folds
    """
    dataset_split = []
    dataset_copy = dataset.copy()
    fold_size = len(dataset) // n_folds

    for _ in range(n_folds):
        fold = []

        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))

        dataset_split.append(fold)

    return np.array(dataset_split)


@njit
def predict(train: np.array, test_row: np.array, K: int = 3) -> np.array:
    """ 
    Make a classification prediction with neighbors
    """
    neighbors = get_neighbors(train, test_row, K)

    output_values = [row[-1] for row in neighbors]

    prediction = max(set(output_values), key=output_values.count)

    return prediction


def k_nearest_neighbors(train: np.array, test: np.array, K: int) -> np.array:
    """
    Apply K Nearest Neighbor algorithm with a given number of neighbors
    """
    return np.array([predict(train, row, K) for row in test])


def evaluate(dataset: np.array, n_folds: int, *args) -> List:
    """
    Evaluate an algorithm using a cross validation split
    """
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

        predicted = k_nearest_neighbors(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)

        scores.append(accuracy)

    return scores

def parallel_preprocess(files: List[Path], conf):
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
            for res in tqdm(p.imap(preprocess, files)):
                pbar.write(res + f" finished...")
                pbar.update()

    return pd.Dataframe()

def preprocess(file: Path) -> str:
    """
    perform feature extraction on image
    """

    try:
        img = get_image_data(file)
        img = select_channel(img, conf["COLOR_CHANNEL"])

    except Exception as e:
        return style(f"[ERROR] {file.stem} has an issue: {e}", fg="red")

    return []



