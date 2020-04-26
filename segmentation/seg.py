
from segmentation.seg_utils import (
    get_image_data,
    export_image,
    select_channel,
    histogram,
    convolve,
    gaussian_kernel,
    sobel_filters,
    non_max_suppression,
    threshold,
    hysteresis,
    find_middle_hist,
    k_means,
)


def erode(img_arr: np.array, win: int = 1) -> np.array:
    """

    erodes 2D numpy array holding a binary image

    """

    r = np.zeros(img_arr.shape)
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

    return r


def dilate(img_arr: np.array, win: int = 1) -> np.array:

    inverted_img = np.invert(img_arr)
    eroded_inverse = erode(inverted_img, win).astype(np.uint8)
    eroded_img = np.invert(eroded_inverse)

    return eroded_img



def histogram_clustering(img_arr: np.array) -> np.array:
    img_hist = histogram(img_arr)
    out = k_means(img_hist, 2)

    diff = abs(out[1] - out[0])

    img_copy = img_arr.copy()
    img_copy[img_copy > diff] = 255
    img_copy[img_copy < diff] = 0

    img_copy = img_copy.astype(np.uint8)

    return img_copy.reshape(img_arr.shape)


def canny_edge_detection(img_arr: np.array) -> np.array:

    guass = gaussian_kernel(5)

    blurred_image = convolve(img_arr, guass)

    sobel, theta = sobel_filters(blurred_image)

    suppresion = non_max_suppression(sobel, theta)

    threshold_image, weak, strong = threshold(suppresion)

    canny_image = hysteresis(threshold_image, weak, strong)

    return canny_image
