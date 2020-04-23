
import random
import numpy as np
from math import sqrt
from numba import njit


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180.0 / np.pi
    angle[angle < 0] += 180

    Z = fast_suppression(img, angle, N, M, Z)

    return Z


@njit(fastmath=True, cache=True)
def fast_suppression(img, angle, N, M, Z):
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

            except:
                pass

    return Z


# @njit(fastmath=True, cache=True)
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size : size + 1, -size : size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal

    return g


@njit(fastmath=True, cache=True)
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


# @njit(fastmath=True, cache=True)
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


@njit(fastmath=True, cache=True)
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
                except:
                    pass
    return img


@njit(fastmath=True, cache=True)
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


# @njit
def find_middle_hist(hist: np.array, min_count: int = 5) -> int:

    num_bins = len(hist)
    hist_start = 0
    while hist[hist_start] < min_count:
        hist_start += 1  # ignore small counts at start

    hist_end = num_bins - 1
    while hist[hist_end] < min_count:
        hist_end -= 1  # ignore small counts at end

    hist_center = int(round(np.average(np.linspace(0, 2 ** 8 - 1, num_bins), weights=hist)))
    left = np.sum(hist[hist_start:hist_center])
    right = np.sum(hist[hist_center : hist_end + 1])

    while hist_start < hist_end:
        if left > right:  # left part became heavier
            left -= hist[hist_start]
            hist_start += 1
        else:  # right part became heavier
            right -= hist[hist_end]
            hist_end -= 1
        new_center = int(round((hist_end + hist_start) / 2))  # re-center the weighing scale

        if new_center < hist_center:  # move bin to the other side
            left -= hist[hist_center]
            right += hist[hist_center]
        elif new_center > hist_center:
            left += hist[hist_center]
            right -= hist[hist_center]

        hist_center = new_center

    return hist_center


# @njit(parallel=True, cache=True)
def k_means(arr: np.array, k: int, num_iter: int = 5) -> np.array:

    size = len(arr)
    centroids = np.array([random.randint(0, size) for _ in range(k)])

    while centroids[0] == centroids[1]:
        centroids = np.array([random.randint(0, size) for _ in range(k)])

    for _ in range(num_iter):
        dist = np.array(
            [
                [
                    sqrt(
                        np.sum(
                            np.array((arr[i] - centroids[j]) ** 2)
                        )
                    )
                    for j in range(k)
                ]
                for i in range(size)
            ]
        )

        labels = np.array([dist[i, :].argmin() for i in range(size)])

        for i in range(k):
            closer = arr[labels == 1]
            if len(closer) > 0:
                centroids[i] = np.nanmean(closer)

    return centroids
