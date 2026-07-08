import numpy as np
import pandas as pd

from scipy.stats import linregress
from sklearn.metrics import r2_score


def weighted_mean(x, weights):
    """
    Compute a weighted mean if there is any valid value.

    Parameters
    ----------
    x : array-like
        Data values.
    weights : array-like
        Corresponding weights.

    Returns
    -------
    float
        Weighted mean, or NaN if weights sum to zero.
    """
    return np.average(x, weights=weights) if len(x) > 0 and np.sum(weights) > 0 else np.nan


def safe_std(x):
    """
    Compute standard deviation if there is more than one value.

    Parameters
    ----------
    x : array-like
        Data values.

    Returns
    -------
    float
        Standard deviation, or 0 if insufficient data.
    """
    return x.std() if len(x) > 1 else 0


def sigmoid(x, L, x0, k, b):
    """
    Sigmoid function for fitting maturation probabilities.

    Parameters
    ----------
    x : array-like
        Input values, usually frame indices.
    L : float
        Maximum value of the sigmoid.
    x0 : float
        Sigmoid midpoint.
    k : float
        Steepness of the curve.
    b : float
        Baseline offset.

    Returns
    -------
    array-like
        Sigmoid-transformed values corresponding to x.
    """
    z = np.clip(-k * (x - x0), -500, 500)
    return L / (1 + np.exp(z)) + b


def classify_maturation(
    prob,
    fit=None,
    fit_success=False,
    r2_thresh=0.75,
    smooth_window=5,
    low_thresh=0.4,
    high_thresh=0.6,
):
    """
    Classify a cell's maturation probability curve into categories.

    Parameters
    ----------
    prob : np.ndarray
        Probability values per frame, between 0 and 1.
    fit : np.ndarray, optional
        Sigmoid fit of probabilities.
    fit_success : bool, optional
        Whether sigmoid fitting succeeded.
    r2_thresh : float, optional
        Minimum R² for using fit instead of raw probabilities.
    smooth_window : int, optional
        Rolling/smoothing window.
    low_thresh : float, optional
        Lower threshold for classification.
    high_thresh : float, optional
        Upper threshold for classification.

    Returns
    -------
    category : int
        Maturation class:
        0 = never mature,
        1 = matures,
        2 = starts mature,
        3 = not classifiable.
    features : dict
        Extracted metrics such as p_start, p_end, delta, n_crossings, and r2.
    """
    r2 = None
    features = {}

    if fit_success:
        if fit is not None:
            r2 = r2_score(prob, fit)
            residuals = prob - fit

            if r2 > r2_thresh:
                probabilities = fit
                smooth_window = 1
                features["std_residuals"] = np.std(residuals)
            else:
                features["std_curve"] = np.std(prob)
                probabilities = prob
        else:
            print("fit is None, using raw probabilities")
            features["std_curve"] = np.std(prob)
            probabilities = prob
    else:
        print("fit failed, using raw probabilities")
        features["std_curve"] = np.std(prob)
        probabilities = prob

    features["r2"] = r2

    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        prob_smooth = np.convolve(probabilities, kernel, mode="same")
    else:
        prob_smooth = probabilities.copy()

    N_edge = max(3, smooth_window)
    p_start = np.mean(prob_smooth[:N_edge])
    p_end = np.mean(prob_smooth[-N_edge:])
    p_max = np.max(prob_smooth)
    p_min = np.min(prob_smooth)
    delta = p_end - p_start

    crossings = np.where(np.diff((prob_smooth > 0.5).astype(int)) != 0)[0]
    n_crossings = len(crossings)

    if p_max < low_thresh:
        category = 0
    elif p_start < low_thresh and p_end > high_thresh and delta > 0.4:
        category = 1
    elif p_start > high_thresh and p_min > low_thresh:
        category = 2
    else:
        category = 3

    features.update(
        {
            "p_start": p_start,
            "p_end": p_end,
            "p_max": p_max,
            "p_min": p_min,
            "delta": delta,
            "n_crossings": n_crossings,
        }
    )

    return category, features


def select_frames(n_frames, crossing_frame=None, min_gap=15):
    """
    Select representative frames for plotting maturation results.

    Parameters
    ----------
    n_frames : int
        Total number of frames in the video.
    crossing_frame : int or None, optional
        Frame where maturation probability crosses 0.5.
    min_gap : int, optional
        Minimum distance from video edges required to use crossing_frame.

    Returns
    -------
    list
        Selected frame indices, up to 5 frames.
    """
    if n_frames < 5:
        return list(range(n_frames))

    first, last = 0, n_frames - 1

    if crossing_frame is not None and min_gap <= crossing_frame <= n_frames - 1 - min_gap:
        mid = crossing_frame
        second = (first + mid) // 2
        fourth = (mid + last) // 2
        return [first, second, mid, fourth, last]

    return [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]


def summarize_clusters_per_cell_frame(result_df):
    """
    Summarize cluster measurements per cell and frame.

    Parameters
    ----------
    result_df : pd.DataFrame
        Detailed cluster measurements.

    Returns
    -------
    pd.DataFrame
        Summary statistics per cell and frame.
    """
    summary_rows = []

    for (cell_id, frame), group in result_df.groupby(["cell_id", "frame"]):
        weights = group["area"] * group["norm_med_int"]

        summary = {
            "cell_id": cell_id,
            "frame": frame,
            "num_clusters": len(group),
            "area_mean": weighted_mean(group["area"], weights),
            "area_safe_std": safe_std(group["area"]),
            "sum_int_mean": weighted_mean(group["sum_int"], weights),
            "sum_int_safe_std": safe_std(group["sum_int"]),
            "norm_sum_int_mean": weighted_mean(group["norm_sum_int"], weights),
            "norm_sum_int_safe_std": safe_std(group["norm_sum_int"]),
            "med_int_mean": weighted_mean(group["med_int"], weights),
            "med_int_safe_std": safe_std(group["med_int"]),
            "norm_med_int_mean": weighted_mean(group["norm_med_int"], weights),
            "norm_med_int_safe_std": safe_std(group["norm_med_int"]),
            "dist_cent_mean": weighted_mean(group["dist_cent"], weights),
            "dist_cent_safe_std": safe_std(group["dist_cent"]),
            "solidity_mean": weighted_mean(group["solidity"], weights),
            "solidity_safe_std": safe_std(group["solidity"]),
            "perimeter_mean": weighted_mean(group["perimeter"], weights),
            "perimeter_safe_std": safe_std(group["perimeter"]),
            "circularity_mean": weighted_mean(group["circularity"], weights),
            "circularity_safe_std": safe_std(group["circularity"]),
            "aspect_ratio_mean": weighted_mean(group["aspect_ratio"], weights),
            "aspect_ratio_safe_std": safe_std(group["aspect_ratio"]),
            "eccentricity_mean": weighted_mean(group["eccentricity"], weights),
            "eccentricity_safe_std": safe_std(group["eccentricity"]),
            "extent_mean": weighted_mean(group["extent"], weights),
            "extent_safe_std": safe_std(group["extent"]),
        }

        summary_rows.append(summary)

    return pd.DataFrame(summary_rows)


def summarize_per_track(linked_df, nm2px, min_frames=5):
    """
    Summarize linked cluster tracks.

    Parameters
    ----------
    linked_df : pd.DataFrame
        Linked cluster dataframe, usually produced by trackpy.
    nm2px : float
        Nanometers-per-pixel conversion factor.
    min_frames : int, optional
        Minimum number of frames required to summarize a track.

    Returns
    -------
    pd.DataFrame
        Per-track feature summary.
    """
    features = []

    for (cell_id, particle), group in linked_df.groupby(["cell_id", "particle"]):
        group = group.sort_values("frame")

        if len(group) < min_frames:
            continue

        frames = group["frame"].values

        x = group["x"].values * nm2px
        y = group["y"].values * nm2px
        dists_to_center = group["dist_cent"].values

        dx = np.diff(x)
        dy = np.diff(y)
        disp = np.sqrt(dx**2 + dy**2)

        total_distance = np.sum(disp)
        net_displacement = np.linalg.norm([x[-1] - x[0], y[-1] - y[0]])

        directionality_ratio = (
            net_displacement / total_distance if total_distance > 0 else 0
        )
        # TODO: convert frame difference to actual time if needed
        duration = frames[-1] - frames[0]
        speed = total_distance / duration if duration > 0 else 0

        slope, intercept, r_value, p_value, std_err = linregress(
            frames,
            dists_to_center,
        )

        angles = np.arctan2(dy, dx)
        angle_diff = np.diff(angles)
        angle_diff = np.unwrap(angle_diff)
        angle_var = np.var(angle_diff)

        if len(disp) > 1:
            speed_reg = linregress(frames[1:], disp)
            speed_slope = speed_reg.slope
        else:
            speed_slope = np.nan

        if len(angle_diff) > 1:
            turning_reg = linregress(np.arange(len(angle_diff)), np.abs(angle_diff))
            turning_rate_slope = turning_reg.slope
        else:
            turning_rate_slope = np.nan

        bbox_area = (np.max(x) - np.min(x)) * (np.max(y) - np.min(y))

        features.append(
            {
                "cell_id": cell_id,
                "particle": particle,
                "track_duration": duration,
                "mean_speed": speed,
                "net_displacement": net_displacement,
                "directionality_ratio": directionality_ratio,
                "radial_slope": slope,
                "radial_r_squared": r_value**2,
                "radial_change": dists_to_center[-1] - dists_to_center[0],
                "angle_variance": angle_var,
                "motion_bbox_area": bbox_area,
                "speed_slope": speed_slope,
                "turning_rate_slope": turning_rate_slope,
            }
        )

    return pd.DataFrame(features)