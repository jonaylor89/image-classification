import numpy as np
from PIL import Image
from math import sqrt
from numba import njit
from pathlib import Path
from operator import eq


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
def get_neighbors(train, test_row, neighbors):
    """
    Locate the most similar neighbors
    """
    distances = [(train_row, distance(test_row, train_row)) for train_row in train]
    distances.sort(key=lambda tup: tup[1])

    neighbors = [distances[i][0] for i in range(neighbors)]

    return neighbors


@njit
def predict(train, test_row, count):
    """ 
    Make a classification prediction with neighbors
    """
    neighbors = get_neighbors(train, test_row, count)

    output_values = [row[-1] for row in neighbors]

    prediction = max(set(output_values), key=output_values.count)

    return prediction


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
