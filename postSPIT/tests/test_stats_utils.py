import numpy as np
import pandas as pd

from postSPIT.stats_utils import (
    weighted_mean,
    safe_std,
    sigmoid,
    select_frames,
    summarize_clusters_per_cell_frame,
)


def test_weighted_mean():
    x = np.array([1, 2, 3])
    weights = np.array([1, 1, 2])

    assert weighted_mean(x, weights) == 2.25


def test_weighted_mean_zero_weights_returns_nan():
    x = np.array([1, 2, 3])
    weights = np.array([0, 0, 0])

    assert np.isnan(weighted_mean(x, weights))


def test_safe_std_single_value_returns_zero():
    assert safe_std(pd.Series([5])) == 0


def test_sigmoid_midpoint():
    y = sigmoid(np.array([0]), L=1, x0=0, k=1, b=0)

    assert np.allclose(y, np.array([0.5]))


def test_select_frames_short_video():
    assert select_frames(4) == [0, 1, 2, 3]


def test_select_frames_with_valid_crossing():
    assert select_frames(40, crossing_frame=20) == [0, 10, 20, 29, 39]


def test_select_frames_without_valid_crossing():
    assert select_frames(40, crossing_frame=3) == [0, 10, 20, 30, 39]


def test_summarize_clusters_per_cell_frame_basic():
    df = pd.DataFrame(
        {
            "cell_id": [1, 1],
            "frame": [0, 0],
            "area": [10.0, 20.0],
            "sum_int": [100.0, 200.0],
            "norm_sum_int": [2.0, 4.0],
            "med_int": [50.0, 60.0],
            "norm_med_int": [1.0, 2.0],
            "dist_cent": [5.0, 10.0],
            "solidity": [0.8, 0.9],
            "perimeter": [12.0, 18.0],
            "circularity": [0.7, 0.6],
            "aspect_ratio": [1.2, 1.5],
            "eccentricity": [0.3, 0.4],
            "extent": [0.5, 0.6],
        }
    )

    summary = summarize_clusters_per_cell_frame(df)

    assert len(summary) == 1
    assert summary.loc[0, "cell_id"] == 1
    assert summary.loc[0, "frame"] == 0
    assert summary.loc[0, "num_clusters"] == 2