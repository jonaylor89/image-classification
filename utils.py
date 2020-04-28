import os
import re
import numpy as np
from PIL import Image
from math import sqrt
from numba import njit
from pathlib import Path
from operator import eq
from click import style
from random import randrange

from typing import List, Callable, Optional


labels: List[str] = ["cyl", "inter", "let", "mod", "para", "super", "svar"]


def deserialize_label(value: int) -> str:
    return labels[value]


def serialize_label(label: str) -> int:
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
    np.savetxt(filename, arr, delimiter=",")


def load_dataset(filename: str) -> np.array:
    try:
        return np.loadtxt(filename, delimiter=",")
    except:
        return []


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


@njit
def entropy(img_arr: np.array) -> int:

    marg = histogram(img_arr) / img_arr.size
    marg = np.array(list(filter(lambda p: p > 0, marg)))

    entropy = -np.sum(np.multiply(marg, np.log2(marg)))

    return entropy


# njit
def accuracy_metric(actual, predicted):
    """
    Calculate accuracy percentage
    """
    correct = list(map(eq, actual, predicted))

    return (sum(correct) / len(correct)) * 100.0


def erode(img_arr: np.array, win: int = 1) -> np.array:
    """
    erodes 2D numpy array holding a binary image
    """

    r = np.zeros_like(img_arr)
    [yy, xx] = np.where(img_arr > 0)

    # prepare neighborhoods
    off = np.tile(range(-win, win + 1), (2 * win + 1, 1))
    x_off = off.flatten()
    y_off = off.T.flatten()

    # duplicate each neighborhood element for each index
    n = len(xx.flatten())
    x_off = np.tile(x_off, (n, 1)).flatten()
    y_off = np.tile(y_off, (n, 1)).flatten()

    # round out offset
    ind = np.sqrt(x_off ** 2 + y_off ** 2) > win
    x_off[ind] = 0
    y_off[ind] = 0

    # duplicate each index for each neighborhood element
    xx = np.tile(xx, ((2 * win + 1) ** 2))
    yy = np.tile(yy, ((2 * win + 1) ** 2))

    nx = xx + x_off
    ny = yy + y_off

    # bounds checking
    ny[ny < 0] = 0
    ny[ny > img_arr.shape[0] - 1] = img_arr.shape[0] - 1
    nx[nx < 0] = 0
    nx[nx > img_arr.shape[1] - 1] = img_arr.shape[1] - 1

    r[ny, nx] = 255

    return r.astype(np.uint8)


def dilate(img_arr: np.array, win: int = 3) -> np.array:

    inverted_img = np.invert(img_arr)
    eroded_inverse = erode(inverted_img, win)
    eroded_img = np.invert(eroded_inverse)

    return eroded_img


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


def histogram_thresholding(
    img_arr: np.array, hist: Optional[np.array] = None
) -> np.array:

    if hist == None:
        hist = histogram(img_arr)

    middle = middle_of(hist)

    img_copy = img_arr.copy()
    img_copy[img_copy > middle] = 255
    img_copy[img_copy < middle] = 0

    img_copy = img_copy.astype(np.uint8)

    return img_copy.reshape(img_arr.shape)


def opening(img_arr: np.array, conf: dict, hist: Optional[np.array] = None) -> np.array:
    segmented_img = histogram_thresholding(img_arr)
    eroded = erode(segmented_img, conf["OPENING_WINDOW"])
    opened = dilate(eroded, conf["OPENING_WINDOW"])

    return opened


def area(img_arr: np.array, conf: dict, hist: Optional[np.array] = None) -> int:

    unique, counts = np.unique(img_arr, return_counts=True)
    counter = dict(zip(unique, counts))

    black_pixel_count = counter[0]

    return black_pixel_count


@njit
def calculate_bound_radius(segmented_img: np.array) -> float:

    center = np.array((0.0, 0.0))  # Fake init radius

    radius = 0.0001  # Fake init radius

    for _ in range(2):
        for pos, x in np.ndenumerate(segmented_img):

            arr_pos = np.array(pos)

            if x != 0:  # Only was pixels a part of the object
                continue

            # dist = distance(arr_pos, center) # Doesn't work because numba is stupid
            diff = arr_pos - center
            dist = np.sqrt(np.sum(diff ** 2))

            if dist < radius:
                continue

            alpha = dist / radius
            alphaSq = alpha ** 2

            radius = 0.5 * (alpha + 1.0 / alpha) * radius

            center = 0.5 * (
                (1.0 + 1.0 / alphaSq) * center + (1.0 - 1.0 / alphaSq) * arr_pos
            )

    for idx, _ in np.ndenumerate(segmented_img):

        arr_pos = np.array(idx)

        # dist = distance(arr_pos, center) # Doesn't work because numba is stupid
        diff = arr_pos - center
        dist = np.sqrt(np.sum(diff ** 2))

        if dist < radius:
            break

        radius = (radius + dist) / 2.0
        center += (dist - radius) / dist * np.subtract(arr_pos, center)

    return radius


@njit
def normalize(dataset: np.array) -> np.array:

    norm_dataset = dataset.copy()

    sans_labels = dataset[:, :-1]
    for idx, column in enumerate(sans_labels.T):
        smallest = np.min(column)
        largest = np.max(column)

        rng = largest - smallest

        if rng == 0:
            continue

        norm_dataset[:, idx] = (norm_dataset[:, idx] - smallest) / rng

    return norm_dataset


def distance(row1: np.array, row2: np.array) -> float:
    """
    calculate the Euclidean distance between two vectors
    """
    return np.linalg.norm(np.subtract(row1, row2))


# @njit
def get_neighbors(train: np.array, test_row: np.array, K: int) -> np.array:
    """
    Locate the most similar neighbors
    """
    distances = [(train_row, distance(test_row, train_row)) for train_row in train]
    distances.sort(key=lambda tup: tup[1])

    neighbors = np.array([distances[i][0] for i in range(K)])

    return neighbors


# @njit
def predict_label(train: np.array, test_row: np.array, K: int = 3) -> np.array:
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
    return np.array([predict_label(train, row, K) for row in test])


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
            fold.append(dataset_copy[index])
            dataset_copy = np.delete(dataset_copy, index, axis=0)

        dataset_split.append(fold)

    return np.array(dataset_split)


def evaluate(dataset: np.array, n_folds: int, K: int) -> List:
    """
    Evaluate an algorithm using a cross validation split
    """

    folds = cross_validation_split(dataset, n_folds)
    scores = []

    for idx, fold in enumerate(folds):
        train_set = np.delete(folds, idx, axis=0)
        train_set = np.concatenate(train_set, axis=0)
        test_set = []

        for row in fold:
            row_copy = row.copy()
            test_set.append(row_copy)
            row_copy[-1] = None

        test_set = np.array(test_set)

        predicted = k_nearest_neighbors(train_set, test_set, K)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)

        scores.append(accuracy)

    return scores


def extract_features(conf: dict, file: Path) -> dict:
    """
    perform feature extraction on image
    """

    try:
        img = get_image_data(file)
        img = select_channel(img, conf["COLOR_CHANNEL"])

        hist = histogram(img)
        opened = opening(img, conf, hist)

        # parse the label name and file number
        search_obj = re.search(r"(\D+)(\d+).*", file.stem, re.M | re.I)
        label = search_obj.group(1)

        try:
            y = serialize_label(label)
        except KeyError:
            y = None

        number = search_obj.group(2)

        # Area of cluster
        x1 = area(opened, conf, hist)

        # Entropy of pixels,
        x2 = entropy(img)

        # histogram mean
        x3 = np.mean(hist)

        # Radius of smallest enclosing sphere
        x4 = calculate_bound_radius(opened)

    except Exception as e:
        raise
        return {
            "features": [],
            "msg": style(f"[ERROR] {file.stem} has an issue: {e}", fg="red"),
        }

    return {
        "features": [x1, x2, x3, x4, y],
        "msg": style(f"{f'[INFO:{file.stem}]':15} finished...", fg="green"),
    }
