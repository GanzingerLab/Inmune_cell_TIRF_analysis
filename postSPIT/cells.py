from .io_utils import get_nm2px
import cv2
import json
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pandas.errors
import re
import tifffile
import warnings

from glob import glob
from natsort import natsorted
from pandas.errors import PerformanceWarning
from picasso.io import load_movie
from scipy.optimize import curve_fit

from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing,
    binary_opening,
)

from .image_processing import (
    li_threshold,
    otsu_threshold,
    phansalkar_threshold,
    remove_small_objects_per_frame,
    create_mask,
    paint_square,
    contour_min_distance,
)

from .stats_utils import (
    sigmoid,
    classify_maturation,
    select_frames,
    summarize_clusters_per_cell_frame,
    summarize_per_track,
)

from spit import tools
from tqdm import tqdm
import trackpy as tp

class Cell_Analyzer:
    """
    A class for analyzing protein clusters and cell maturation from multi-channel time-lapse TIRF microscopy images.

    This class handles:
        - Loading multi-wavelength image sequences from a folder.
        - Cropping images per cell based on ROI files.
        - Thresholding, cluster detection, and cluster feature extraction.
        - Tracking clusters over time and summarizing per-cell statistics.
        - Predicting cell maturation probability using a pre-trained model.
        - Saving cluster and maturation results, including videos and plots.

    Attributes
    ----------
    folder : str
        Path to the experiment folder containing images and ROI files.
    ch0_wl : str
        Wavelength for channel 0 (e.g., '405nm').
    ch1_wl : str
        Wavelength for channel 1 (e.g., '488nm').
    images : dict
        Dictionary mapping wavelength strings to 3D image stacks (frames, height, width).
    channels : list
        List of available wavelength channels.
    sep_cells : dict
        Dictionary mapping cell IDs to cropped images per channel.
    contours : dict
        Dictionary mapping cell IDs to cell contour coordinates.
    centroids : dict
        Dictionary mapping cell IDs to cell centroid coordinates.
    clusters_binary : dict
        Binary cluster masks per cell per channel.
    result_cluster_analysis : dict
        Detailed cluster measurements (DataFrame) per channel.
    summary_cluster_analysis : dict
        Summary statistics per channel.
    linked_clusters : dict
        Cluster tracks (DataFrame) per channel.
    linked_clusters_stats : dict
        Track-level summary statistics per channel.
    maturation : dict
        Predicted maturation per cell per channel.
    """
    def __init__(self, folder, ch0_wl=None, ch1_wl=None, roi_type = None):
        self.folder = folder 
        self.nm2px = get_nm2px(self.folder)

        # load images into dict by wavelength
        paths = glob(self.folder + '/**/*nm.tif', recursive=True)
        self.images = {}
        for image in paths:
            match = re.search(r'(\d{3}nm)', image)
            if match: 
                wl = match.group(1)   # e.g. '405nm'
                self.images[wl] = load_movie(image)[0]

        self.channels = list(self.images.keys())

        # store wavelength choices
        self.ch0_wl = ch0_wl
        self.ch1_wl = ch1_wl

        # validate wavelength choices
        if (ch0_wl and ch0_wl not in self.images) or (ch1_wl and ch1_wl not in self.images):
            raise ValueError(f"Requested wavelengths not found. Available: {self.channels}. Requested: {ch0_wl}, {ch1_wl}")
        # separate the cells:
        # try:
        self.split_cells(roi_type=roi_type)
        # except: 
        #     print('No ROI available')
        # placeholders for downstream analyses
        self.clusters_binary = {wl: None for wl in self.channels}
        self.result_cluster_analysis = {wl: None for wl in self.channels}
        self.summary_cluster_analysis = {wl: None for wl in self.channels}
        self.linked_clusters = {wl: None for wl in self.channels}
        self.linked_clusters_stats = {wl: None for wl in self.channels}
        self.maturation = {wl: None for wl in self.channels}
        
        # # placeholders for downstream analyses
        # self.result_cluster_analysis = {wl: None for wl in ['ch0','ch1']}
        # self.summary_cluster_analysis = {wl: None for wl in ['ch0','ch1']}
        # self.linked_clusters = {wl: None for wl in ['ch0','ch1']}
        # self.linked_clusters_stats = {wl: None for wl in ['ch0','ch1']}
        # self.maturation = {wl: None for wl in ['ch0','ch1']}
        self._auto_load()
    # --- properties (dynamic views into self.images) ---
    @property
    def ch0(self):
        return self.images.get(self.ch0_wl)
    @property
    def ch1(self):
        return self.images.get(self.ch1_wl)
    
    def _auto_load(self):
        """
        Check if cluster/tracking/maturation results exist on disk and load them.
        """
        cluster_dir = os.path.join(self.folder, "cluster_analysis")
        maturation_dir = os.path.join(self.folder, "maturation_analysis")
        
        for ch in self.channels:            
            # --- Cluster results ---
            clusters_npy   = os.path.join(cluster_dir, f"clusters_binary_{ch}.npy")
            clusters_csv   = os.path.join(cluster_dir, f"clusters_{ch}.hdf")
            clusters_stats = os.path.join(cluster_dir, f"clusters_stats_{ch}.csv")
            tracks_csv     = os.path.join(cluster_dir, f"clusters_tracks_{ch}.hdf")
            tracks_stats   = os.path.join(cluster_dir, f"clusters_tracks_stats_{ch}.csv")
            if os.path.exists(clusters_npy): 
                self.clusters_binary[ch] = np.load(clusters_npy, allow_pickle=True).item()
            if os.path.exists(clusters_csv):
                self.result_cluster_analysis[ch] = pd.read_hdf(clusters_csv)
            if os.path.exists(clusters_stats):
                self.summary_cluster_analysis[ch] = pd.read_csv(clusters_stats)
            if os.path.exists(tracks_csv):
                self.linked_clusters[ch] = pd.read_hdf(tracks_csv)
            
            if os.path.exists(tracks_stats):
                try:
                    self.linked_clusters_stats[ch] = pd.read_csv(tracks_stats)
                except pandas.errors.EmptyDataError:
                    print(f"Warning: {tracks_stats} is empty or malformed.")

            
            # --- Maturation results ---
            maturation_json = os.path.join(maturation_dir, f"maturation__{ch}.json")
            if os.path.exists(maturation_json):
                with open(maturation_json, "r") as f:
                    self.maturation[ch] = pd.DataFrame(json.load(f))
            
    def _get_contour(self, cell_id, frame):
        key = (cell_id, frame)
        if key in self.contours.keys():
            return self.contours[key]
        else:
            return self.contours[(cell_id, 0)]
    def _get_centroid(self, cell_id, frame):
        key = (cell_id, frame)
        if key in self.centroids.keys():
            return self.centroids[key]
        else:
            return self.centroids[(cell_id, 0)]
    def split_cells(self, roi_type = None):
        """
       Crop images per cell based on ROI files and store contours and centroids.

       Raises
       ------
       RuntimeError
           If no ROI files are found in the folder.
       """
        roi_contours = {}
        roi_centroids = {}
        bboxes = {}
        rois = natsorted(glob(os.path.join(self.folder, "*.roi")))
        pkl = os.path.join(self.folder,"cell_detection", "linked_rois.pkl")
        if not rois and not os.path.exists(pkl):
            raise RuntimeError(f"No ROI files found in folder: {self.folder}")
        
        if os.path.exists(pkl) and roi_type in (None, 'pkl'):
            linked_rois = pd.read_pickle(pkl)
            if 'label' not in linked_rois.columns:
                linked_rois['label'] = linked_rois['cell_id']
            bbox = linked_rois.groupby('label').agg(
                    bbox_x0=('x','min'),
                    bbox_x1=('x','max'),
                    bbox_y0=('y','min'),
                    bbox_y1=('y','max')
                )
            unique_contours = linked_rois.set_index(['label','frame'])['contour']
            unique_centroids = linked_rois.set_index(['label','frame'])[['x','y']].apply(tuple, axis=1)
            unique_bboxes = bbox.apply(lambda r: [int(r.bbox_x0), int(r.bbox_x1), int(r.bbox_y0), int(r.bbox_y1)], axis=1)

        elif rois and roi_type in (None, 'rois'):
            for roi in rois:
                cell_id = int(re.search(r'roi(\d+)\.roi$', roi).group(1))
                roi_contour = tools.get_roi_contour(roi)
                roi_centroid = tools.get_roi_centroid(roi_contour)
                x0, x1 = int(min(roi_contour[:, 0])), int(max(roi_contour[:, 0]))
                y0, y1 = int(min(roi_contour[:, 1])), int(max(roi_contour[:, 1]))
                roi_contours[(cell_id, 0)] = roi_contour
                roi_centroids[(cell_id, 0)] = roi_centroid[0]
                bboxes[cell_id] = [x0, x1, y0, y1]


            unique_contours = pd.Series(roi_contours)
            unique_contours.index = pd.MultiIndex.from_tuples(unique_contours.index, names=['label','frame'])
            unique_centroids = pd.Series(roi_centroids)
            unique_centroids.index = pd.MultiIndex.from_tuples(unique_centroids.index, names=['label','frame'])
            unique_bboxes = pd.Series(bboxes)
            unique_bboxes.index.name = 'cell_id'
        else:
            raise RuntimeError(f"Requested roi_type={roi_type!r}, but that source is not available.")
        
        self.sep_cells, self.contours, self.centroids = {}, {}, {}
        for (cell_id, frame) in unique_contours.index:
            i = unique_contours[(cell_id, frame)]
            j = unique_centroids[(cell_id, frame)]
            x0, x1, y0, y1 = unique_bboxes[cell_id]

            # crop available channels
            if cell_id not in self.sep_cells:
                self.sep_cells[cell_id] = {wl: im[:, y0:y1, x0:x1] for wl, im in self.images.items()}


            corr = i.copy()
            corr[:, 0] -= x0
            corr[:, 1] -= y0
            cx, cy = j
            corr_cen = (cx - x0, cy - y0)
            self.contours[(cell_id, frame)] = corr
            self.centroids[(cell_id, frame)] = corr_cen
    def analyze_clusters_protein(self, min_size=80, ch='ch0', th_method='li_local', global_th_mode='max',
                             window_size=15, p=2, q=6, save_videos=False, overwrite = False):
        """
        Detect and analyze protein clusters per cell using thresholding and tracking.

        Parameters
        ----------
        min_size : int, optional
            Minimum cluster size in pixels to include (default=80).
        ch : str, optional
            Channel to analyze: 'ch0', 'ch1', or specific wavelength key (default='ch0').
        th_method : str, optional
            Thresholding method: 'li_local', 'otsu_local', etc. (default='li_local').
        global_th_mode : str, optional
            Mode for global threshold: 'max', 'full', 'median', 'last' (default='max').
        window_size : int, optional
            Window size for local thresholding (default=15).
        p : float, optional
            Phansalkar parameter 1 (default=2).
        q : float, optional
            Phansalkar parameter 2 (default=6).
        save_videos : bool, optional
            Whether to save centroid overlay videos (default=False).
        overwrite : bool, optional
            Overwrite existing results if present (default=False).

        Returns
        -------
        clusters_binary : dict
            Binary masks of clusters per cell.
        result : pd.DataFrame
            Detailed measurements per cluster.
        results_stats : pd.DataFrame
            Summary statistics per cell per frame.
        linked_df : pd.DataFrame
            Tracked cluster DataFrame.
        linked_stats : pd.DataFrame
            Summary statistics per track.
        """
        clusters_binary = {}
        all_props = []
        self.cluster_contours = {}  # Store contours by cell/frame/label
    
        # map channel string to actual wavelength key
        if ch == 'ch0':
            wl = self.ch0_wl
        elif ch == 'ch1':
            wl = self.ch1_wl
        elif ch in self.images:
            wl = ch  # ch is already a valid wavelength key
        else:
            raise ValueError(f"Invalid channel {ch}. Must be 'ch0', 'ch1', or one of {list(self.images.keys())}")
        if (self.result_cluster_analysis[wl] is not None and
            self.summary_cluster_analysis[wl] is not None and
            self.linked_clusters[wl] is not None and
            self.linked_clusters_stats[wl] is not None and
            not overwrite):
            print(f"Skipping {wl}: analysis results already loaded in memory.")
            return (None, 
                    self.result_cluster_analysis[wl],
                    self.summary_cluster_analysis[wl],
                    self.linked_clusters[wl],
                    self.linked_clusters_stats[wl])
    
        for cell_id in self.sep_cells.keys():
            to_work = self.sep_cells[cell_id][wl]
    
            if 'li' in th_method:
                global_mask = li_threshold(to_work, mode=global_th_mode)
            elif 'otsu' in th_method:
                global_mask = otsu_threshold(to_work, mode=global_th_mode)
            else:
                raise ValueError(f'{th_method} is not a valid thresholding method')
    
            if 'local' in th_method:
                local_mask = phansalkar_threshold(to_work, radius=window_size, p=p, q=q)
                binary_stack = remove_small_objects_per_frame(
                    binary_opening(binary_closing(local_mask & global_mask)), min_size=min_size
                )
            else:
                binary_stack = global_mask
    
            clusters_binary[cell_id] = binary_stack
            # mask = create_mask(to_work[0].shape, self.contour[cell_id])
    
            if cell_id not in self.cluster_contours:
                self.cluster_contours[cell_id] = {}
            
            for frame_num, binary_frame in enumerate(binary_stack):
                if frame_num not in self.cluster_contours[cell_id]:
                    self.cluster_contours[cell_id][frame_num] = {}
    
                mask = create_mask(to_work[0].shape, self._get_contour(cell_id, frame_num))
                labeled = label(binary_frame)

                for region in regionprops(labeled, intensity_image=to_work[frame_num]):
                    if region.area <= min_size:
                        continue
    
                    region_mask = (labeled == region.label).astype(np.uint8)
                    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
                    if len(contours) == 0:
                        contour_list = np.array([]).reshape(0, 2)
                        perimeter = np.nan
                        solidity = np.nan
                        area = 0
                    else:
                        cnt = max(contours, key=cv2.contourArea)
                        perimeter = cv2.arcLength(cnt, True)
                        contour_list = cnt.reshape(-1, 2)
                        area = cv2.contourArea(cnt)
                        hull_area = cv2.contourArea(cv2.convexHull(cnt))
                        solidity = area / hull_area if hull_area > 0 else np.nan
    
                    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter and perimeter > 1e-6 else np.nan
    
                    M = cv2.moments(region_mask)
                    if M['m00'] != 0:
                        centroid_row = M['m01'] / M['m00']
                        centroid_col = M['m10'] / M['m00']
                    else:
                        centroid_row, centroid_col = region.centroid
    
                    bbox = region.bbox
                    region_mask_bool = (labeled == region.label)
                    median_int_out = np.median(to_work[frame_num][~mask])
                    median_int = np.median(to_work[frame_num][region_mask_bool])
                    sum_int = np.sum(to_work[frame_num][region_mask_bool])
    
                    minr, minc, maxr, maxc = bbox
                    bbox_height = maxr - minr
                    bbox_width = maxc - minc
                    aspect_ratio = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else np.nan
                    extent = region.area / (bbox_height * bbox_width) if bbox_height > 0 and bbox_width > 0 else np.nan
                    eccentricity = region.eccentricity if hasattr(region, 'eccentricity') else np.nan
    
                    props = {
                        'cell_id': cell_id,
                        'frame': frame_num,
                        'area': area * self.nm2px ** 2,
                        'sum_int': sum_int,
                        'norm_sum_int': sum_int / median_int_out,
                        'med_int': median_int,
                        'norm_med_int': median_int / median_int_out,
                        'dist_cent': np.linalg.norm(np.array([centroid_row, centroid_col]) - self._get_centroid(cell_id, frame_num)) * self.nm2px,
                        'solidity': solidity,
                        'perimeter': perimeter * self.nm2px,
                        'circularity': circularity,
                        'centroid_row': centroid_row,
                        'centroid_col': centroid_col,
                        'bbox_min_row': bbox[0],
                        'bbox_min_col': bbox[1],
                        'bbox_max_row': bbox[2],
                        'bbox_max_col': bbox[3],
                        'contour': contour_list,
                        'label': region.label,
                        'aspect_ratio': aspect_ratio,
                        'extent': extent,
                        'eccentricity': eccentricity
                    }
                    all_props.append(props)
                    self.cluster_contours[cell_id][frame_num][region.label] = contour_list
    
        result = pd.DataFrame(all_props)
        if result.empty: 
            results_stats = pd.DataFrame()
            linked_df = pd.DataFrame()
            linked_stats = pd.DataFrame()
        else:
            results_stats = summarize_clusters_per_cell_frame(result)
            df_tp = result.rename(columns={"centroid_col": "x", "centroid_row": "y", "norm_sum_int": "mass", "area": "size"})
            df_tp['label'] = result['label']
    
            linked_all = []
            for cell_id, df_cell in df_tp.groupby('cell_id'):
                linked = tp.link_df(df_cell, search_range=25, memory=0, adaptive_step=0.95,
                                    adaptive_stop=2, link_strategy='hybrid')
                linked['cell_id'] = cell_id
                linked_all.append(linked)
    
            linked_df = pd.concat(linked_all, ignore_index=True)
            linked_stats = summarize_per_track(linked_df)
        self.result_cluster_analysis[wl] = result
        self.summary_cluster_analysis[wl] = results_stats
        self.linked_clusters[wl] = linked_df
        self.linked_clusters_stats[wl] = linked_stats
        
        if save_videos:
            self._save_centroid_videos_per_cell(clusters_binary, result, self.sep_cells, ch=ch, square_size=2)
        if self.folder is not None:
            output_dir = os.path.join(self.folder, 'cluster_analysis')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        
        np.save(os.path.join(output_dir, f'clusters_binary_{wl}'), clusters_binary, allow_pickle=True)
        if not result.empty:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PerformanceWarning)
                result.to_hdf(os.path.join(output_dir, f'clusters_{wl}.hdf'), key='df')   
        if not results_stats.empty:
            results_stats.to_csv(os.path.join(output_dir, f'clusters_stats_{wl}.csv'), index = False) 
        if not linked_df.empty:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PerformanceWarning)
                linked_df.to_hdf(os.path.join(output_dir, f'clusters_tracks_{wl}.hdf'), key = 'df')   
        if not linked_stats.empty:
            linked_stats.to_csv(os.path.join(output_dir, f'clusters_tracks_stats_{wl}.csv'), index = False) 
        return clusters_binary, result, results_stats, linked_df, linked_stats
    def predict_maturation(self, model, preprocess_frame_func = None, save_plot = False, 
                           N_rolling = 5, ch = 'ch0', r2_thresh = 0.75, low_thresh=0.4, high_thresh=0.6, overwrite = False):
        """
        Predict cell maturation using a deep learning model and optionally plot results.

        Parameters
        ----------
        model : tf.keras.Model
            Pre-trained model to predict maturation probability.
        preprocess_frame_func : callable, optional
            Function to preprocess a single frame for model input. If none is given a default one is used. 
        save_plot : bool, optional
            Save probability and sigmoid fit plots for each cell (default=False).
        N_rolling : int, optional
            Rolling window for probability smoothing (default=5).
        ch : str, optional
            Channel to analyze: 'ch0', 'ch1', or specific wavelength key (default='ch0').
        r2_thresh : float, optional
            Minimum R² value for sigmoid fit to be considered valid (default=0.75).
        low_thresh : float, optional
            Lower threshold for maturation probability classification (default=0.4).
        high_thresh : float, optional
            Upper threshold for maturation probability classification (default=0.6).
        overwrite : bool, optional
            Overwrite existing maturation results if present (default=False).

        Returns
        -------
        pd.DataFrame
            DataFrame containing predicted probabilities, sigmoid fit parameters,
            classification category, crossing frame, and features per cell.
        """
        # map channel string to actual wavelength key
        if ch == 'ch0':
            wl = self.ch0_wl
        elif ch == 'ch1':
            wl = self.ch1_wl
        elif ch in self.images:
            wl = ch  # ch is already a valid wavelength key
        else:
            raise ValueError(f"Invalid channel {ch}. Must be 'ch0', 'ch1', or one of {list(self.images.keys())}")
        if self.maturation[wl] is not None and not overwrite:
            print(f"Skipping {wl}: maturation results already loaded in memory.")
            return self.maturation[wl]
        results_list = []
        for cell_id in tqdm(self.sep_cells.keys()):
            video = self.sep_cells[cell_id][wl]
            if preprocess_frame_func is not None:
                video_processed = np.array([preprocess_frame_func(f)[0] for f in video])
            else:
                video_processed = np.array([self._preprocess_frame(f)[0] for f in video])
            predictions = model.predict(video_processed, verbose=0)  # single progress bar
            predictions = predictions[:, 0]  # flatten
            x_data = np.array(range(video.shape[0]))
            fit_success = True
            try:
                popt, pcov = curve_fit(sigmoid, x_data, predictions, p0=[1, 0, 1, 0], maxfev=10000)
                y_fit = sigmoid(x_data, *popt)
            except RuntimeError:
                print(f"Fit failed for: {self.folder}, cell: {cell_id}")
                fit_success = False
                y_fit = None  # or skip plotting
            category, features = classify_maturation(predictions, fit = y_fit, 
                                                    fit_success = fit_success, r2_thresh = r2_thresh, smooth_window=N_rolling, 
                                                                            low_thresh=low_thresh, high_thresh=high_thresh) 
            r2 = features['r2']
            # Create figure
            fig = plt.figure(figsize=(20, 10))  # slightly shorter height
            gs = gridspec.GridSpec(2, 1, height_ratios=[2, 0.5], hspace=0.02)  # smaller bottom row
            
            # Figure title
            if 'output' in self.folder:
                title = self.folder.split(r'output')[-1]
            else:
                title = self.folder
            fig.suptitle(f"{title}, cell: {cell_id}", fontsize=16, y=0.95)
            x_cross = None
            # --- Main probability curve ---
            ax_curve = fig.add_subplot(gs[0])
            if fit_success: 
                L, x0, k, b = popt
                ax_curve.plot(x_data, y_fit, 
                label=f'Sigmoid Fit (R²={r2:.2f})\nL={L:.2f}, x0={x0:.2f}, k={k:.2f}, b={b:.2f}', color="#DD8452")

                crossing_indices = np.where(y_fit > 0.5)[0]
                if len(crossing_indices) > 0:
                        x_cross = x_data[crossing_indices[0]]
                        ax_curve.axvline(x=x_cross, color="#55A868", linestyle='--', label=f'Crosses 0.5 at x={x_cross:.0f}')

            crossing_frame = int(x_cross) if x_cross is not None else None
            selected_indices = select_frames(len(video), crossing_frame)

            rolling_avg = np.convolve(predictions, np.ones(N_rolling)/N_rolling, mode='valid')
            ax_curve.plot(range(N_rolling-1, len(predictions)), rolling_avg,
                          label=f'Rolling Avg (N={N_rolling})', color="#4C72B0", linewidth=2)
            
            exclude_keys = {"r2"}
            features_str = ", ".join([f"{k}: {v:.2f}" for k, v in features.items() if k not in exclude_keys])
            
            ax_curve.set_ylabel("Probability", fontsize=12)
            ax_curve.set_xlabel("Frame", fontsize=12)
            ax_curve.set_ylim(0, 1)
            ax_curve.set_title(f"DL - {category} - {features_str}", fontsize=14)
            ax_curve.legend()
            
            # --- Row of images ---
            # n_images = len(selected_indices)
            bottom = 0.05  # bottom of image row
            height = 0.15
            for i, idx in enumerate(selected_indices):
                ax_img = fig.add_axes([0.05 + i*0.18, bottom, 0.16, height])
                ax_img.imshow(video[idx], cmap='gray')
                ax_img.set_title(f"Frame {idx}", fontsize=10)
                ax_img.axis('off')
                
                
            # Create save folder
            save_folder = os.path.join(self.folder, 'maturation_analysis')
            os.makedirs(save_folder, exist_ok=True)
            
            # Prepare data
            sig_params = dict(zip(["L", "x0", "k", "b"], [float(v) for v in popt])) if fit_success else None
            
            cell_result = {
                "run": self.folder,
                "cell": int(cell_id),
                "n_frames": int(len(video)),
                "probabilities": [float(p) for p in predictions],
                "sigmoid_params": sig_params,
                "sigmoid_r2": float(r2) if r2 is not None else None,
                "features": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in features.items()},
                "category": category,
                "crossing_frame": int(crossing_frame) if crossing_frame is not None else None,
            }
            
            # Flatten for DataFrame
            flat_result = {
                "run": cell_result["run"],
                "cell": cell_result["cell"],
                "n_frames": cell_result["n_frames"],
                "category": cell_result["category"],
                "crossing_frame": cell_result["crossing_frame"],
                **cell_result["features"],
                **(cell_result["sigmoid_params"] or {}), 
                "sigmoid_r2": cell_result["sigmoid_r2"], 
                "probabilities": cell_result["probabilities"]
            }

            results_list.append(flat_result)

            
            # Define base name
            base_name = f"maturation_roi{cell_id}"
            
            # Save plot
            if save_plot:
                plot_path = os.path.join(save_folder, f"{base_name}_{wl}_plot.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            plt.show()
        results_df = pd.DataFrame(results_list)
        json_path = os.path.join(save_folder, f"maturation__{wl}.json")
        with open(json_path, "w") as f:
            json.dump(results_list, f, indent=2)  # indent=2 for readability
        CSV = results_df.drop(columns=["probabilities"])        
        CSV.to_csv(os.path.join(save_folder, f"maturation_{wl}.csv"), index = False)
        self.maturation[wl] = results_df
        return results_df
    
    def _preprocess_frame(self, frame, target_size=(224, 224)):
        """
            Preprocess a single grayscale frame for model input.
        
            Parameters
            ----------
            frame : np.ndarray
                2D grayscale image.
            target_size : tuple, optional
                Desired output size (height, width), by default (224, 224).
        
            Returns
            -------
            np.ndarray
                Preprocessed frame ready for model prediction (shape: 1xHxWx3).
    """
        # frame: 2D grayscale
        frame = frame.astype(np.float32) / 4095.0   # normalize like during training
        frame = np.expand_dims(frame, axis=-1)      # HxWx1
        frame_rgb = np.repeat(frame, 3, axis=-1)    # HxWx3
        frame_rgb = cv2.resize(frame_rgb, target_size)  # resize to model input
        frame_rgb = np.expand_dims(frame_rgb, axis=0)   # add batch dimension
        return frame_rgb
    
    def _save_centroid_videos_per_cell(self, clusters_binary, all_props, sep_cells, ch='ch0', square_size=3, output_dir='cluster_analysis', filtered_spots=None):
        
        """
            Save videos overlaying cluster centroids on original and binary images.
        
            Parameters
            ----------
            clusters_binary : dict
                Binary cluster masks per cell.
            all_props : pd.DataFrame
                Cluster properties per cell/frame.
            sep_cells : dict
                Cropped images per cell and channel.
            ch : str, optional
                Channel to save, by default 'ch0'.
            square_size : int, optional
                Size of centroid marker squares, by default 3.
            output_dir : str, optional
                Directory to save videos, by default 'cluster_analysis'.
            filtered_spots : pd.DataFrame, optional
                Spots to overlay additionally, by default None.
    """
        
        if ch == 'ch0':
            wl = self.ch0_wl
        elif ch == 'ch1':
            wl = self.ch1_wl
        elif ch in self.images:
            wl = ch
        else:
            raise ValueError(f"Invalid channel {ch}. Must be 'ch0', 'ch1', or one of {list(self.images.keys())}")
    
        if self.folder is not None and not os.path.isabs(output_dir):
            output_dir = os.path.join(self.folder, output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        for cell_id, cell in sep_cells.items():
            if wl not in cell:
                print(f"Warning: Cell {cell_id} does not have channel {wl}, skipping")
                continue
    
            num_frames = cell[wl].shape[0]
    
            orig_stack = []
            bin_stack = []
    
            cell_props = all_props[all_props['cell_id'] == cell_id]
            if filtered_spots is not None:
                cell_spots = filtered_spots[filtered_spots['cell_id'] == cell_id]
    
            for frame_num in range(num_frames):
                orig_frame = cell[wl][frame_num]  # use actual wavelength key
                binary_frame = clusters_binary[cell_id][frame_num]
    
                norm = (orig_frame - orig_frame.min()) / (orig_frame.ptp() + 1e-9)
                orig_rgb = (np.dstack([norm] * 3) * 255).astype(np.uint8)
                bin_rgb = np.dstack([binary_frame.astype(np.uint8) * 255] * 3)
    
                binary_uint8 = (binary_frame * 255).astype(np.uint8)
                contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(orig_rgb, contours, -1, (128, 0, 128), 1)
    
                frame_props = cell_props[cell_props['frame'] == frame_num]
                for _, row in frame_props.iterrows():
                    r, c = int(round(row['centroid_row'])), int(round(row['centroid_col']))
                    paint_square(orig_rgb, (r, c), size=square_size)
                    paint_square(bin_rgb, (r, c), size=square_size)
    
                if filtered_spots is not None:
                    frame_spots = cell_spots[cell_spots['t'] == frame_num]
                    for _, spot in frame_spots.iterrows():
                        r, c = int(round(spot['y_per_cell'])), int(round(spot['x_per_cell']))
                        paint_square(orig_rgb, (r, c), size=square_size, color=[255, 255, 0])
                        paint_square(bin_rgb, (r, c), size=square_size, color=[255, 255, 0])
    
                orig_stack.append(orig_rgb)
                bin_stack.append(bin_rgb)
    
            orig_path = os.path.join(output_dir, f"{wl}_cell{cell_id}_centroids.tif")
            bin_path = os.path.join(output_dir, f"{wl}_cell{cell_id}_binary_centroids.tif")
    
            tifffile.imwrite(orig_path, np.array(orig_stack), photometric='rgb')
            tifffile.imwrite(bin_path, np.array(bin_stack), photometric='rgb')
            # print(f"Saved Cell {cell_id} to:\n- {orig_path}\n- {bin_path}")      
    
    def _detect_splits_and_merges(self, linked_df, distance_threshold=20, frame_gap=1):
        linked_df = linked_df.copy()
        linked_df['split_event'] = False
        linked_df['merge_event'] = False

        frames = sorted(linked_df['frame'].unique())

        # Build contours dict: {frame: {particle: contour_points}}
        polys_by_frame = {}

        for frame in frames:
            polys_by_frame[frame] = {}
            frame_data = linked_df[linked_df['frame'] == frame]

            for _, row in frame_data.iterrows():
                particle = row['particle']
                contour_points = self.contours_by_frame_and_particle.get(frame, {}).get(particle, None)
                if contour_points is None or len(contour_points) < 3:
                    continue
                polys_by_frame[frame][particle] = contour_points

        for t in frames[:-frame_gap]:
            current_contours = polys_by_frame.get(t, {})
            next_contours = polys_by_frame.get(t + frame_gap, {})

            if not current_contours or not next_contours:
                continue

            # Detect merges: many → one
            for next_part, next_cnt in next_contours.items():
                close_parents = []
                for cur_part, cur_cnt in current_contours.items():
                    dist = contour_min_distance(cur_cnt, next_cnt)
                    if dist < distance_threshold:
                        close_parents.append(cur_part)
                if len(close_parents) > 1:
                    linked_df.loc[
                        (linked_df['frame'] == t + frame_gap) & (linked_df['particle'] == next_part),
                        'merge_event'
                    ] = True

            # Detect splits: one → many
            for cur_part, cur_cnt in current_contours.items():
                close_children = []
                for next_part, next_cnt in next_contours.items():
                    dist = contour_min_distance(cur_cnt, next_cnt)
                    if dist < distance_threshold:
                        close_children.append(next_part)
                if len(close_children) > 1:
                    linked_df.loc[
                        (linked_df['frame'] == t) & (linked_df['particle'] == cur_part),
                        'split_event'
                    ] = True

        return linked_df
    
    def _get_contour_for_particle(self, cell_id, frame, particle):
        """
        Retrieve stored contour points for a given cell/frame/particle.
        Returns None if not found.
        """
        return self.cluster_contours.get(cell_id, {}).get(frame, {}).get(particle, None)
    def _detect_merges_proximity_based(self, linked_df, proximity_threshold=15, frame_gap=1):
        """
        Detect merges based on proximity of clusters at time t and disappearance at t+1.
    
        Parameters:
        -----------
        linked_df : pd.DataFrame
            DataFrame from trackpy with particle tracks and frames.
        proximity_threshold : float
            Distance (in pixels) under which two clusters are considered "close".
        frame_gap : int
            How many frames ahead to check for merging.
        
        Returns:
        --------
        pd.DataFrame : updated linked_df with a new 'merge_event_prox' column
        """
        import numpy as np
    
        linked_df = linked_df.copy()
        linked_df['merge_event_prox'] = False
    
        frames = sorted(linked_df['frame'].unique())
    
        # Build contour lookup
        contours_by_frame = self.contours_by_frame_and_particle
    
        for t in frames[:-frame_gap]:
            frame_df = linked_df[linked_df['frame'] == t]
            next_df = linked_df[linked_df['frame'] == t + frame_gap]
    
            current_particles = frame_df['particle'].values
            next_particles = next_df['particle'].values
    
            for i, row1 in frame_df.iterrows():
                p1 = row1['particle']
                cnt1 = contours_by_frame.get(t, {}).get(p1, None)
                if cnt1 is None or len(cnt1) < 3:
                    continue
    
                for j, row2 in frame_df.iterrows():
                    p2 = row2['particle']
                    if p1 >= p2:
                        continue  # avoid duplicate pairs
    
                    cnt2 = contours_by_frame.get(t, {}).get(p2, None)
                    if cnt2 is None or len(cnt2) < 3:
                        continue
    
                    # Check if these two are close enough to consider for merging
                    dist = contour_min_distance(np.array(cnt1), np.array(cnt2))
                    if dist < proximity_threshold:
                        # Now check if only one object exists nearby in next frame
                        possible_merge_target = []
                        for _, row_next in next_df.iterrows():
                            p_next = row_next['particle']
                            cnt_next = contours_by_frame.get(t + frame_gap, {}).get(p_next, None)
                            if cnt_next is None or len(cnt_next) < 3:
                                continue
    
                            d1 = contour_min_distance(np.array(cnt1), np.array(cnt_next))
                            d2 = contour_min_distance(np.array(cnt2), np.array(cnt_next))
    
                            if d1 < proximity_threshold and d2 < proximity_threshold:
                                possible_merge_target.append(p_next)
    
                        # If exactly one such object exists in the next frame, mark both as merging
                        if len(possible_merge_target) == 1:
                            linked_df.loc[(linked_df['frame'] == t) & (linked_df['particle'].isin([p1, p2])), 'merge_event_prox'] = True
    
        return linked_df