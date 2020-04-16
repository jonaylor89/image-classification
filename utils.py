import numpy as np
from PIL import Image
from math import sqrt
from numba import njit
from pathlib import Path


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


# @njit(fastmath=True)
def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


# @njit(fastmath=True)
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size : size + 1, -size : size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


@njit(fastmath=True)
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


# @njit(fastmath=True)
def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong


# njit(fastmath=True)
def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                try:
                    if (
                        (img[i + 1, j - 1] == strong)
                        or (img[i + 1, j] == strong)
                        or (img[i + 1, j + 1] == strong)
                        or (img[i, j - 1] == strong)
                        or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong)
                        or (img[i - 1, j] == strong)
                        or (img[i - 1, j + 1] == strong)
                    ):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


@njit(fastmath=True)
def convolve(img_array: np.array, img_filter: np.array) -> np.array:
    """
    Applies a filter to a copy of an image based on filter weights
    """

    rows, cols = img_array.shape
    height, width = img_filter.shape

    output = np.zeros((rows - height + 1, cols - width + 1))

    for rr in range(rows - height + 1):
        for cc in range(cols - width + 1):
            for hh in range(height):
                for ww in range(width):
                    imgval = img_array[rr + hh, cc + ww]
                    filterval = img_filter[hh, ww]
                    output[rr, cc] += imgval * filterval

    return output


@njit(parallel=True)
def kmeans(
    A: np.array, numCenter: int, numIter: int, size: int, features: int
) -> np.array:
    # https://github.com/numba/numba/blob/master/examples/k-means/k-means_numba.py
    centroids = np.random.ranf((numCenter, features))

    for l in range(numIter):
        dist = np.array(
            [
                [
                    sqrt(np.sum((A[i, :] - centroids[j, :]) ** 2))
                    for j in range(numCenter)
                ]
                for i in range(size)
            ]
        )
        labels = np.array([dist[i, :].argmin() for i in range(size)])

        centroids = np.array(
            [
                [
                    np.sum(A[labels == i, j]) / np.sum(labels == i)
                    for j in range(features)
                ]
                for i in range(numCenter)
            ]
        )

    return centroids
