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
from scipy.stats import linregress
from sklearn.metrics import r2_score
from skimage.draw import polygon
from skimage.filters import threshold_li, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing,
    binary_opening,
    remove_small_holes,
    remove_small_objects,
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
        self.nm2px = self._get_nm2px()

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

    def _get_nm2px(self):
        """
        Get nanometers-per-pixel scaling factor from result.txt.

        Returns
        -------
        float
            Nanometers per pixel.

        Raises
        ------
        ValueError
            If the microscope source is unknown.
        """
        resultPath  = glob(self.folder + '/**/*result.txt', recursive=True)[0]
        result_txt  = tools.read_result_file(resultPath)
        if result_txt['Computer'] == 'ANNAPURNA': 
            return 90.16
        elif result_txt['Computer'] == 'K2-BIVOUAC':
            return 108
        else:
            raise ValueError(f"Unknown microscope source: {result_txt['Computer']}")
    def get_time_interval(self):
        """
        Retrieve the time interval (dt) between frames from result.txt.

        Returns
        -------
        float
            Time interval in seconds.

        Raises
        ------
        FileNotFoundError
            If no result.txt file is found in the folder.
        """
        # Extract dt (frame interval) from result.txt
        result_files = glob(os.path.join(self.folder, "*result.txt"))
        if not result_files:
            raise FileNotFoundError("No result.txt file found in the folder.")
        with open(result_files[0], 'r') as f:
            resultLines = f.readlines()
    
        if tools.find_string(resultLines, 'Interval'): 
            interval = tools.find_string(resultLines, 'Interval').split(":")[-1].strip()
            if interval.split(" ")[-1] == 'sec':
                dt = 1.0 * float(interval.split(" ")[0])
            elif interval.split(" ")[-1] == 'ms':
                dt = 0.001 * float(interval.split(" ")[0])
        else:
            dtStr = tools.find_string(resultLines, 'Camera Exposure')[17:-1]
            dt = 0.001 * float((''.join(c for c in dtStr if (c.isdigit() or c == '.'))))
        return dt
            
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
                global_mask = self._li_threshold(to_work, mode=global_th_mode)
            elif 'otsu' in th_method:
                global_mask = self._otsu_threshold(to_work, mode=global_th_mode)
            else:
                raise ValueError(f'{th_method} is not a valid thresholding method')
    
            if 'local' in th_method:
                local_mask = self._phansalkar_threshold(to_work, radius=window_size, p=p, q=q)
                binary_stack = self._remove_small_objects_per_frame(
                    binary_opening(binary_closing(local_mask & global_mask)), min_size=min_size
                )
            else:
                binary_stack = global_mask
    
            clusters_binary[cell_id] = binary_stack
            # mask = self._create_mask(to_work[0].shape, self.contour[cell_id])
    
            if cell_id not in self.cluster_contours:
                self.cluster_contours[cell_id] = {}
            
            for frame_num, binary_frame in enumerate(binary_stack):
                if frame_num not in self.cluster_contours[cell_id]:
                    self.cluster_contours[cell_id][frame_num] = {}
    
                mask = self._create_mask(to_work[0].shape, self._get_contour(cell_id, frame_num))
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
            results_stats = self._summarize_clusters_per_cell_frame(result)
            df_tp = result.rename(columns={"centroid_col": "x", "centroid_row": "y", "norm_sum_int": "mass", "area": "size"})
            df_tp['label'] = result['label']
    
            linked_all = []
            for cell_id, df_cell in df_tp.groupby('cell_id'):
                linked = tp.link_df(df_cell, search_range=25, memory=0, adaptive_step=0.95,
                                    adaptive_stop=2, link_strategy='hybrid')
                linked['cell_id'] = cell_id
                linked_all.append(linked)
    
            linked_df = pd.concat(linked_all, ignore_index=True)
            linked_stats = self._summarize_per_track(linked_df)
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
                popt, pcov = curve_fit(self._sigmoid, x_data, predictions, p0=[1, 0, 1, 0], maxfev=10000)
                y_fit = self._sigmoid(x_data, *popt)
            except RuntimeError:
                print(f"Fit failed for: {self.folder}, cell: {cell_id}")
                fit_success = False
                y_fit = None  # or skip plotting
            category, features = self._classify_maturation(predictions, fit = y_fit, 
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
            selected_indices = self._select_frames(len(video), crossing_frame)

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
    def _sigmoid(self, x, L, x0, k, b):
        """
            Sigmoid function for fitting maturation probabilities.
        
            Parameters
            ----------
            x : array-like
                Input values (frame indices).
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
                Sigmoid-transformed values corresponding to `x`.
    """
        z = np.clip(-k * (x - x0), -500, 500)  # prevent overflow
        return L / (1 + np.exp(z)) + b
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
    def _classify_maturation(self, prob, fit = None, fit_success = False, r2_thresh = 0.75, smooth_window=5, 
                            low_thresh=0.4, high_thresh=0.6):
        """
    Classify a cell's maturation probability curve into categories.

    Parameters
    ----------
    prob : np.ndarray
        Probability values per frame (0-1).
    fit : np.ndarray, optional
        Sigmoid fit of probabilities, by default None.
    fit_success : bool, optional
        Whether sigmoid fitting succeeded, by default False.
    r2_thresh : float, optional
        Minimum R² for using fit instead of raw probabilities, by default 0.75.
    smooth_window : int, optional
        Rolling average window for smoothing, by default 5.
    low_thresh : float, optional
        Lower threshold for classification, by default 0.4.
    high_thresh : float, optional
        Upper threshold for classification, by default 0.6.

    Returns
    -------
    category : int
        Maturation class (0: never, 1: matures, 2: starts mature, 3: not classifiable).
    features : dict
        Extracted metrics such as p_start, p_end, delta, n_crossings, r2, etc.
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
                print('fit is None, using raw probabilities')
                features["std_curve"] = np.std(prob)
                probabilities = prob
        else:
            print('fit failed, using raw probabilities')
            features["std_curve"] = np.std(prob)
            probabilities = prob

        features['r2'] = r2
        # Smooth
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            prob_smooth = np.convolve(probabilities, kernel, mode='same')
        else:
            prob_smooth = probabilities.copy()

        # Features
        N_edge = max(3, smooth_window)  # use first/last few points
        p_start = np.mean(prob_smooth[:N_edge])
        p_end   = np.mean(prob_smooth[-N_edge:])
        p_max   = np.max(prob_smooth)
        p_min   = np.min(prob_smooth)
        delta   = p_end - p_start

        # Count threshold crossings at 0.5
        crossings = np.where(np.diff((prob_smooth > 0.5).astype(int)) != 0)[0]
        n_crossings = len(crossings)

        # Classification rules
        if p_max < low_thresh:
            category = 0
        elif p_start < low_thresh and p_end > high_thresh and delta > 0.4:
            category = 1
        elif p_start > high_thresh and p_min > low_thresh:
            category = 2
        else:
            category = 3

        features.update({
        "p_start": p_start,
        "p_end": p_end,
        "p_max": p_max,
        "p_min": p_min,
        "delta": delta,
        "n_crossings": n_crossings
    })

        return category, features
    def _select_frames(self, n_frames, crossing_frame=None, min_gap=15):
        """
            Select representative frames for plotting or analysis.
        
            Parameters
            ----------
            n_frames : int
                Total number of frames in the video.
            crossing_frame : int or None, optional
                Frame where maturation probability crosses 0.5, by default None.
            min_gap : int, optional
                Minimum gap from edges to consider crossing_frame, by default 15.
        
            Returns
            -------
            list
                Selected frame indices (up to 5 frames).
    """
        if n_frames < 5:
            # Not enough frames, just return all
            return list(range(n_frames))

        first, last = 0, n_frames - 1

        if crossing_frame is not None and min_gap <= crossing_frame <= n_frames - 1 - min_gap:
            mid = crossing_frame
            # 2 is midway between 1 and 3
            second = (first + mid) // 2
            # 4 is midway between 3 and 5
            fourth = (mid + last) // 2
            return [first, second, mid, fourth, last]
        else:
            # No valid crossing, just pick 5 equally spaced
            return [0, n_frames//4, n_frames//2, 3*n_frames//4, n_frames-1]
    def _li_threshold(self, image, mode = "max"):
        """
            Apply Li global threshold to a 2D/3D image stack.
        
            Parameters
            ----------
            image : np.ndarray
                Input image stack (frames, height, width) or single frame.
            mask : np.ndarray
                Cell mask (not currently applied in global thresholding).
            mode : str, optional
                Mode for threshold calculation: 'max', 'full', 'median', 'last', by default "max".
        
            Returns
            -------
            np.ndarray
                Binary mask after thresholding and closing.
    """
        if mode == 'max':
            thresh = threshold_li(np.max(image, axis = 0))
        elif mode == 'full':
            thresh = threshold_li(np.array(image))
        elif mode == 'median':
            thresh = threshold_li(np.median(image, axis = 0))
        elif mode == 'last':
            thresh = threshold_li(image[-1])
        else:
            print("mode is not valid")
            return None
        global_mask = binary_closing((image > thresh)) #& self._create_mask(image[0].shape, mask))
        return global_mask
    def _otsu_threshold(self, original, image, mode = "max"): ##TODO: remove original? 
        """
            Apply Otsu global threshold to a 2D/3D image stack.
        
            Parameters
            ----------
            original : np.ndarray
                Original image stack.
            image : np.ndarray
                Image stack to threshold.
            mask : np.ndarray
                Cell mask (not currently applied in global thresholding).
            mode : str, optional
                Mode for threshold calculation: 'max', 'full', 'median', 'last', by default "max".
        
            Returns
            -------
            np.ndarray
                Binary mask after thresholding and closing.
    """
        if mode == 'max':
            thresh = threshold_otsu(np.max(original, axis = 0))
        elif mode == 'full':
            thresh = threshold_otsu(np.array(original))
        elif mode == 'median':
            thresh = threshold_otsu(np.median(original, axis = 0))
        elif mode == 'last':
            thresh = threshold_otsu(original[-1])
        else:
            print("mode is not valid")
            return None
        global_mask = binary_closing((image > thresh))# & self._create_mask(image[0].shape, mask))
        return global_mask
    def _phansalkar_threshold(self, image_stack, radius=15, k=0.25, p=2.0, q=10.0):
        """
            Apply Phansalkar local thresholding per frame.
        
            Parameters
            ----------
            image_stack : np.ndarray
                2D or 3D array of images (frames, height, width).
            radius : int, optional
                Local window radius, by default 15.
            k : float, optional
                Phansalkar parameter k, by default 0.25.
            p : float, optional
                Phansalkar parameter p, by default 2.0.
            q : float, optional
                Phansalkar parameter q, by default 10.0.
        
            Returns
            -------
            np.ndarray
                Binary thresholded image or stack of same shape.
    """
        window_size = (radius * 2) + 1

        def threshold_single(image):
            image = image / np.max(image) if np.max(image) > 0 else image
            mean = cv2.blur(image, (window_size, window_size))
            mean_sq = cv2.blur(image**2, (window_size, window_size))
            std = np.sqrt(mean_sq - mean**2)
            threshold = mean * (1 + p * np.exp(-q * mean) + k * ((std / 0.5) - 1))
            return image > threshold

        if image_stack.ndim == 2:
            return threshold_single(image_stack)
        elif image_stack.ndim == 3:
            # Process each frame independently and stack results
            binary_stack = np.zeros_like(image_stack, dtype=bool)
            for i in range(image_stack.shape[0]):
                binary_stack[i] = threshold_single(image_stack[i])
            return binary_stack
        else:
            raise ValueError("Input must be 2D or 3D numpy array")
    def _remove_small_objects_per_frame(self, stack, min_size=100, connectivity=1):
        """
            Remove small objects from each frame of a binary stack.
        
            Parameters
            ----------
            stack : np.ndarray
                Binary image stack (frames, height, width).
            min_size : int, optional
                Minimum object size in pixels to keep, by default 100.
            connectivity : int, optional
                Connectivity for object removal, by default 1.
        
            Returns
            -------
            np.ndarray
                Cleaned binary stack.
    """
        cleaned_stack = np.zeros_like(stack, dtype=bool)
        for i in range(stack.shape[0]):  # assuming frames on axis 0
            cleaned_stack[i] = remove_small_objects(stack[i], min_size=min_size, connectivity=connectivity)
        return cleaned_stack
    def _remove_small_holes_per_frame(self, stack, min_size=100, connectivity=1):
        """
            Fill small holes in each frame of a binary stack.
        
            Parameters
            ----------
            stack : np.ndarray
                Binary image stack (frames, height, width).
            min_size : int, optional
                Maximum hole size in pixels to fill, by default 100.
            connectivity : int, optional
                Connectivity for hole filling, by default 1.
        
            Returns
            -------
            np.ndarray
                Binary stack with small holes removed.
    """
        cleaned_stack = np.zeros_like(stack, dtype=bool)
        for i in range(stack.shape[0]):  # assuming frames on axis 0
            cleaned_stack[i] = remove_small_holes(stack[i], area_threshold=min_size, connectivity=connectivity)
        return cleaned_stack
    def _summarize_clusters_per_cell_frame(self, result_df):
        """
            Summarize cluster measurements per cell and per frame by averaging and 
            calculating the std of the different parameters.
        
            Parameters
            ----------
            result_df : pd.DataFrame
                Detailed cluster measurements.
        
            Returns
            -------
            pd.DataFrame
                Summary statistics per cell per frame.
        """
        summary_rows = []

        for (cell_id, frame), group in result_df.groupby(['cell_id', 'frame']):
            weights = group['area'] * group['norm_med_int']
            summary = {
                'cell_id': cell_id,
                'frame': frame,
                'num_clusters': len(group),
                'area_mean': self._weighted_mean(group['area'], weights),
                'area_safe_std': self._safe_std(group['area']),
                'sum_int_mean': self._weighted_mean(group['sum_int'], weights),
                'sum_int_safe_std': self._safe_std(group['sum_int']),
                'norm_sum_int_mean': self._weighted_mean(group['norm_sum_int'], weights),
                'norm_sum_int_safe_std': self._safe_std(group['norm_sum_int']),
                'med_int_mean': self._weighted_mean(group['med_int'], weights),
                'med_int_safe_std': self._safe_std(group['med_int']),
                'norm_med_int_mean': self._weighted_mean(group['norm_med_int'], weights),
                'norm_med_int_safe_std': self._safe_std(group['norm_med_int']),
                'dist_cent_mean': self._weighted_mean(group['dist_cent'], weights),
                'dist_cent_safe_std': self._safe_std(group['dist_cent']),
                'solidity_mean': self._weighted_mean(group['solidity'], weights),
                'solidity_safe_std': self._safe_std(group['solidity']),
                'perimeter_mean': self._weighted_mean(group['perimeter'], weights),
                'perimeter_safe_std': self._safe_std(group['perimeter']),
                'circularity_mean': self._weighted_mean(group['circularity'], weights),
                'circularity_safe_std': self._safe_std(group['circularity']),
                'aspect_ratio_mean': self._weighted_mean(group['aspect_ratio'], weights),
                'aspect_ratio_safe_std': self._safe_std(group['aspect_ratio']),
                'eccentricity_mean': self._weighted_mean(group['eccentricity'], weights),
                'eccentricity_safe_std': self._safe_std(group['eccentricity']),
                'extent_mean': self._weighted_mean(group['extent'], weights),
                'extent_safe_std': self._safe_std(group['extent']),
            }
            summary_rows.append(summary)

        return pd.DataFrame(summary_rows) 
    def _weighted_mean(self, x, weights):
        """
            Compute a weighted mean if there is any valid value, if not it returns NaN.
        
            Parameters
            ----------
            x : array-like
                Data values.
            weights : array-like
                Corresponding weights.
        
            Returns
            -------
            float
                Weighted mean, or NaN if weights sum to 0.
            """
        return np.average(x, weights=weights) if len(x) > 0 and np.sum(weights) > 0 else np.nan
    def _safe_std(self, x):
        """
            Compute standard deviation if there is any valid value, if not it returns 0.
        
            Parameters
            ----------
            x : array-like
                Data values.
        
            Returns
            -------
            float
                Standard deviation or 0 if insufficient data.
    """
        return x.std() if len(x) > 1 else 0
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
                    self._paint_red_square(orig_rgb, (r, c), size=square_size)
                    self._paint_red_square(bin_rgb, (r, c), size=square_size)
    
                if filtered_spots is not None:
                    frame_spots = cell_spots[cell_spots['t'] == frame_num]
                    for _, spot in frame_spots.iterrows():
                        r, c = int(round(spot['y_per_cell'])), int(round(spot['x_per_cell']))
                        self._paint_red_square(orig_rgb, (r, c), size=square_size, color=[255, 255, 0])
                        self._paint_red_square(bin_rgb, (r, c), size=square_size, color=[255, 255, 0])
    
                orig_stack.append(orig_rgb)
                bin_stack.append(bin_rgb)
    
            orig_path = os.path.join(output_dir, f"{wl}_cell{cell_id}_centroids.tif")
            bin_path = os.path.join(output_dir, f"{wl}_cell{cell_id}_binary_centroids.tif")
    
            tifffile.imwrite(orig_path, np.array(orig_stack), photometric='rgb')
            tifffile.imwrite(bin_path, np.array(bin_stack), photometric='rgb')
            # print(f"Saved Cell {cell_id} to:\n- {orig_path}\n- {bin_path}")
    def _paint_red_square(self, image, center, size=1, color=None):
        """
    Paint a square on an RGB image at a specified center.

    Parameters
    ----------
    image : np.ndarray
        RGB image.
    center : tuple
        (row, col) coordinates of the square center.
    size : int, optional
        Size of the square (pixels), by default 1.
    color : list, optional
        RGB color values, by default red [255,0,0].
    """
        if color is None:
            color = [255, 0, 0]  # red

        r, c = center
        half = size // 2
        r_start = max(r - half, 0)
        r_end = min(r + half + 1, image.shape[0])
        c_start = max(c - half, 0)
        c_end = min(c + half + 1, image.shape[1])
        image[r_start:r_end, c_start:c_end] = color        
    def _create_mask(self, image_shape, contour):
        """
            Create a binary mask from a polygon contour.
        
            Parameters
            ----------
            image_shape : tuple
                Shape of the output mask (height, width).
            contour : np.ndarray
                Nx2 array of polygon coordinates.
        
            Returns
            -------
            np.ndarray
                Boolean mask of the polygon.
    """
        mask = np.zeros(image_shape, dtype=bool)
        rr, cc = polygon(contour[:, 1], contour[:, 0], image_shape)
        mask[rr, cc] = True
        return mask
    def _summarize_per_track(self, linked_df, min_frames=5):
        features = []
        
        for (cell_id, particle), group in linked_df.groupby(['cell_id', 'particle']):
            group = group.sort_values('frame')
            if len(group) < min_frames:
                continue
    
            frames = group['frame'].values
            # Convert coordinates to nanometers
            x = group['x'].values * self.nm2px
            y = group['y'].values * self.nm2px
            dists_to_center = group['dist_cent'].values  # already in nm if set that way
            
            # Compute frame-to-frame displacements and instantaneous speeds
            dx = np.diff(x)
            dy = np.diff(y)
            disp = np.sqrt(dx**2 + dy**2)  # instantaneous displacement (nm)
            
            total_distance = np.sum(disp)
            net_displacement = np.linalg.norm([x[-1] - x[0], y[-1] - y[0]])
            directionality_ratio = net_displacement / total_distance if total_distance > 0 else 0
            # TODO: convert frame difference to actual time if needed
            duration = frames[-1] - frames[0]
            speed = total_distance / duration if duration > 0 else 0  # mean speed (nm per frame)
    
            # Radial regression from dists_to_center vs frames (slope in nm per frame)
            slope, intercept, r_value, p_value, std_err = linregress(frames, dists_to_center)
    
            # Compute instantaneous angles for movement
            angles = np.arctan2(dy, dx)
            # Differences between successive angles (turning angles)
            angle_diff = np.diff(angles)  
            # Unwrap to reduce artefactual jumps at ±pi
            angle_diff = np.unwrap(angle_diff)
            angle_var = np.var(angle_diff)  # overall variance
    
            # --- New Temporal Features ---
            # Instantaneous speed trend: regression on disp versus frame (for frames[1:])
            if len(disp) > 1:
                speed_reg = linregress(frames[1:], disp)
                speed_slope = speed_reg.slope  # change in instantaneous speed (nm per frame^2)
            else:
                speed_slope = np.nan
    
            # Turning rate trend: regression on absolute turning angles versus index (0, 1, 2,...)
            if len(angle_diff) > 1:
                turning_reg = linregress(np.arange(len(angle_diff)), np.abs(angle_diff))
                turning_rate_slope = turning_reg.slope  # change in turning (radians per frame)
            else:
                turning_rate_slope = np.nan
    
            # Bounding box area (in nm^2)
            bbox_area = (np.max(x) - np.min(x)) * (np.max(y) - np.min(y))
            
            features.append({
                'cell_id': cell_id,
                'particle': particle,
                'track_duration': duration,
                'mean_speed': speed,
                'net_displacement': net_displacement,
                'directionality_ratio': directionality_ratio,
                'radial_slope': slope,
                'radial_r_squared': r_value**2,
                'radial_change': dists_to_center[-1] - dists_to_center[0],
                'angle_variance': angle_var,
                'motion_bbox_area': bbox_area,
                # New features capturing temporal trends:
                'speed_slope': speed_slope,
                'turning_rate_slope': turning_rate_slope,
            })
        
        return pd.DataFrame(features)
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
                    dist = self._contour_min_distance(cur_cnt, next_cnt)
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
                    dist = self._contour_min_distance(cur_cnt, next_cnt)
                    if dist < distance_threshold:
                        close_children.append(next_part)
                if len(close_children) > 1:
                    linked_df.loc[
                        (linked_df['frame'] == t) & (linked_df['particle'] == cur_part),
                        'split_event'
                    ] = True

        return linked_df
    def _contour_min_distance(self, cnt1, cnt2):
        """
        Compute the minimum Euclidean distance between two contours (Nx2 numpy arrays).
        """
        # cnt1 and cnt2 are arrays of shape (N_points, 2)
        # Compute all pairwise distances and find the minimum
        dists = np.sqrt(np.sum((cnt1[:, None, :] - cnt2[None, :, :])**2, axis=2))
        return np.min(dists)
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
                    dist = self._contour_min_distance(np.array(cnt1), np.array(cnt2))
                    if dist < proximity_threshold:
                        # Now check if only one object exists nearby in next frame
                        possible_merge_target = []
                        for _, row_next in next_df.iterrows():
                            p_next = row_next['particle']
                            cnt_next = contours_by_frame.get(t + frame_gap, {}).get(p_next, None)
                            if cnt_next is None or len(cnt_next) < 3:
                                continue
    
                            d1 = self._contour_min_distance(np.array(cnt1), np.array(cnt_next))
                            d2 = self._contour_min_distance(np.array(cnt2), np.array(cnt_next))
    
                            if d1 < proximity_threshold and d2 < proximity_threshold:
                                possible_merge_target.append(p_next)
    
                        # If exactly one such object exists in the next frame, mark both as merging
                        if len(possible_merge_target) == 1:
                            linked_df.loc[(linked_df['frame'] == t) & (linked_df['particle'].isin([p1, p2])), 'merge_event_prox'] = True
    
        return linked_df