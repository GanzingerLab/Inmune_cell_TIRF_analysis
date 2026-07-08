import numpy as np

from postSPIT.image_processing import (
    create_mask,
    compute_mean_intensity,
    contour_min_distance,
    remove_small_objects_per_frame,
    remove_small_holes_per_frame,
)


def test_create_mask_basic_square():
    contour = np.array(
        [
            [1, 1],
            [3, 1],
            [3, 3],
            [1, 3],
        ]
    )

    mask = create_mask((5, 5), contour)

    assert mask.shape == (5, 5)
    assert mask.dtype == bool
    assert mask.any()


def test_compute_mean_intensity_center_patch():
    image = np.arange(25).reshape(5, 5)

    value = compute_mean_intensity(image, x=2, y=2, window=1)

    expected = image[1:4, 1:4].mean()
    assert value == expected


def test_compute_mean_intensity_at_edge():
    image = np.arange(25).reshape(5, 5)

    value = compute_mean_intensity(image, x=0, y=0, window=1)

    expected = image[0:2, 0:2].mean()
    assert value == expected


def test_contour_min_distance():
    cnt1 = np.array([[0, 0], [1, 0]])
    cnt2 = np.array([[3, 0], [4, 0]])

    assert contour_min_distance(cnt1, cnt2) == 2


def test_remove_small_objects_per_frame():
    stack = np.zeros((1, 5, 5), dtype=bool)
    stack[0, 1, 1] = True
    stack[0, 3:5, 3:5] = True

    cleaned = remove_small_objects_per_frame(stack, min_size=2)

    assert cleaned[0, 1, 1] == False
    assert cleaned[0, 3:5, 3:5].all()


def test_remove_small_holes_per_frame():
    stack = np.ones((1, 5, 5), dtype=bool)
    stack[0, 2, 2] = False

    cleaned = remove_small_holes_per_frame(stack, min_size=2)

    assert cleaned[0, 2, 2] == True