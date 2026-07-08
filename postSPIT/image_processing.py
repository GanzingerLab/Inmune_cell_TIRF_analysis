import cv2
import numpy as np

from skimage.draw import polygon
from skimage.filters import threshold_li, threshold_otsu
from skimage.morphology import (
    binary_closing,
    remove_small_holes,
    remove_small_objects,
)


def li_threshold(image, mode="max"):
    """
    Apply Li global threshold to a 2D/3D image stack.

    Parameters
    ----------
    image : np.ndarray
        Input image stack with shape (frames, height, width), or a single frame.
    mode : str, optional
        Mode for threshold calculation:
        - "max": threshold on maximum-intensity projection
        - "full": threshold on the full stack
        - "median": threshold on median projection
        - "last": threshold on the final frame

    Returns
    -------
    np.ndarray or None
        Binary mask after thresholding and closing. Returns None if mode is invalid.
    """
    if mode == "max":
        thresh = threshold_li(np.max(image, axis=0))
    elif mode == "full":
        thresh = threshold_li(np.array(image))
    elif mode == "median":
        thresh = threshold_li(np.median(image, axis=0))
    elif mode == "last":
        thresh = threshold_li(image[-1])
    else:
        print("mode is not valid")
        return None

    return binary_closing(image > thresh)


def otsu_threshold(image, mode="max"):
    """
    Apply Otsu global threshold to a 2D/3D image stack.

    Parameters
    ----------
    image : np.ndarray
        Input image stack with shape (frames, height, width), or a single frame.
    mode : str, optional
        Mode for threshold calculation:
        - "max": threshold on maximum-intensity projection
        - "full": threshold on the full stack
        - "median": threshold on median projection
        - "last": threshold on the final frame

    Returns
    -------
    np.ndarray or None
        Binary mask after thresholding and closing. Returns None if mode is invalid.
    """
    if mode == "max":
        thresh = threshold_otsu(np.max(image, axis=0))
    elif mode == "full":
        thresh = threshold_otsu(np.array(image))
    elif mode == "median":
        thresh = threshold_otsu(np.median(image, axis=0))
    elif mode == "last":
        thresh = threshold_otsu(image[-1])
    else:
        print("mode is not valid")
        return None

    return binary_closing(image > thresh)


def phansalkar_threshold(image_stack, radius=15, k=0.25, p=2.0, q=10.0):
    """
    Apply Phansalkar local thresholding per frame.

    Parameters
    ----------
    image_stack : np.ndarray
        2D or 3D array of images. If 3D, frames are assumed to be on axis 0.
    radius : int, optional
        Local window radius. The actual window size is ``2 * radius + 1``.
    k : float, optional
        Phansalkar parameter k.
    p : float, optional
        Phansalkar parameter p.
    q : float, optional
        Phansalkar parameter q.

    Returns
    -------
    np.ndarray
        Binary thresholded image or stack with the same shape as the input.

    Raises
    ------
    ValueError
        If the input is not a 2D or 3D array.
    """
    window_size = (radius * 2) + 1

    def threshold_single(image):
        # Normalize frame before estimating local mean/std.
        image = image / np.max(image) if np.max(image) > 0 else image

        mean = cv2.blur(image, (window_size, window_size))
        mean_sq = cv2.blur(image**2, (window_size, window_size))
        std = np.sqrt(mean_sq - mean**2)

        threshold = mean * (
            1 + p * np.exp(-q * mean) + k * ((std / 0.5) - 1)
        )

        return image > threshold

    if image_stack.ndim == 2:
        return threshold_single(image_stack)

    if image_stack.ndim == 3:
        binary_stack = np.zeros_like(image_stack, dtype=bool)

        for i in range(image_stack.shape[0]):
            binary_stack[i] = threshold_single(image_stack[i])

        return binary_stack

    raise ValueError("Input must be 2D or 3D numpy array")


def remove_small_objects_per_frame(stack, min_size=100, connectivity=1):
    """
    Remove small connected objects from each frame of a binary stack.

    Parameters
    ----------
    stack : np.ndarray
        Binary image stack with frames on axis 0.
    min_size : int, optional
        Minimum object size in pixels to keep.
    connectivity : int, optional
        Connectivity used by skimage.morphology.remove_small_objects.

    Returns
    -------
    np.ndarray
        Cleaned binary stack.
    """
    cleaned_stack = np.zeros_like(stack, dtype=bool)

    for i in range(stack.shape[0]):
        cleaned_stack[i] = remove_small_objects(
            stack[i],
            min_size=min_size,
            connectivity=connectivity,
        )

    return cleaned_stack


def remove_small_holes_per_frame(stack, min_size=100, connectivity=1):
    """
    Fill small holes in each frame of a binary stack.

    Parameters
    ----------
    stack : np.ndarray
        Binary image stack with frames on axis 0.
    min_size : int, optional
        Maximum hole area to fill, in pixels.
    connectivity : int, optional
        Connectivity used by skimage.morphology.remove_small_holes.

    Returns
    -------
    np.ndarray
        Binary stack with small holes filled.
    """
    cleaned_stack = np.zeros_like(stack, dtype=bool)

    for i in range(stack.shape[0]):
        cleaned_stack[i] = remove_small_holes(
            stack[i],
            area_threshold=min_size,
            connectivity=connectivity,
        )

    return cleaned_stack


def create_mask(image_shape, contour):
    """
    Create a binary mask from a polygon contour.

    Parameters
    ----------
    image_shape : tuple
        Shape of the output mask as (height, width).
    contour : np.ndarray
        Nx2 array of polygon coordinates in (x, y) format.

    Returns
    -------
    np.ndarray
        Boolean mask of the polygon.
    """
    mask = np.zeros(image_shape, dtype=bool)
    rr, cc = polygon(contour[:, 1], contour[:, 0], image_shape)
    mask[rr, cc] = True
    return mask


def paint_square(image, center, size=1, color=None):
    """
    Paint a square marker on an RGB image.

    Parameters
    ----------
    image : np.ndarray
        RGB image modified in place.
    center : tuple
        Marker center as (row, col).
    size : int, optional
        Size of the square marker in pixels.
    color : list or tuple, optional
        RGB color values. Defaults to red, [255, 0, 0].

    Returns
    -------
    None
        The input image is modified in place.
    """
    if color is None:
        color = [255, 0, 0]

    r, c = center
    half = size // 2

    r_start = max(r - half, 0)
    r_end = min(r + half + 1, image.shape[0])
    c_start = max(c - half, 0)
    c_end = min(c + half + 1, image.shape[1])

    image[r_start:r_end, c_start:c_end] = color


def compute_mean_intensity(image, x, y, window=1):
    """
    Compute mean intensity in a local square patch around a spot.

    Parameters
    ----------
    image : np.ndarray
        2D image array.
    x : float
        Spot x-coordinate in pixels.
    y : float
        Spot y-coordinate in pixels.
    window : int, optional
        Half-width of the square patch. The total patch size is
        ``2 * window + 1``.

    Returns
    -------
    float
        Mean intensity in the patch. Returns NaN if the patch is empty.
    """
    xi, yi = int(round(x)), int(round(y))

    x_min = max(0, xi - window)
    x_max = min(image.shape[1], xi + window + 1)
    y_min = max(0, yi - window)
    y_max = min(image.shape[0], yi + window + 1)

    patch = image[y_min:y_max, x_min:x_max]

    return np.mean(patch) if patch.size > 0 else np.nan

def contour_min_distance(cnt1, cnt2):
    dists = np.sqrt(np.sum((cnt1[:, None, :] - cnt2[None, :, :]) ** 2, axis=2))
    return np.min(dists)