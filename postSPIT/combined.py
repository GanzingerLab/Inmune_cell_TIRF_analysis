import cv2
import numpy as np
import os
import pandas as pd
import yaml

from glob import glob
from spit import colocalize as coloc
from spit import linking as link
from spit import tools
from tqdm import tqdm

from .cells import Cell_Analyzer
from .tracked import Single_tracked_folder

from .image_processing import compute_mean_intensity

class Combined_analysis:
    """
        Class for combined analysis of tracked spots and clusters in microscopy images.
        
        This class integrates tracked particle data with cluster analysis results,
        allowing users to:
        - Combine single-molecule spots with cluster ROIs.
        - Remove spots within clusters.
        - Compute mean intensities of spots outside clusters.
        - Retrack spots after excluding the ones inside clusters using Trackpy.
        - Re-colocalize tracks for dual-channel analysis.
        - Extract filtered diffusion coefficients (D) and dwell times for specific cell maturation behaviour.
        
        Attributes
        ----------
        folder : str
            Path to the folder containing the tracked images and cluster analysis data.
        tracked_folder : Single_tracked_folder
            Instance managing the tracked files in the folder.
        tracked : Tracked_image or None
            Loaded tracked image data for both channels.
        clusters : Cell_Analyzer or None
            Instance containing cluster analysis results.
        spots_outside_clusters : dict
            Filtered spots outside clusters, keyed by channel wavelength.
        cluster_and_spots_stats : dict
            Statistics merging cluster and spot data, keyed by channel wavelength.
        tracks_outside_clusters : dict
            Re-tracked particle data outside clusters, keyed by channel wavelength.
        tracks_outside_clusters_stats : dict
            Statistics for re-tracked particles outside clusters.
        cotracks_outside_clusters : pd.DataFrame or None
            Co-localized tracks outside clusters (dual-channel only).
        cotracks_outside_clusters_stats : pd.DataFrame or None
            Statistics for co-localized tracks outside clusters.
        nm2px : float
            Conversion factor from nanometers to pixels.
    """
    def __init__(self, folder, ch0_hint = None, ch1_hint = None, verbose = True):
        self.folder = folder
        self.tracked_folder = Single_tracked_folder(folder, ch0_hint = ch0_hint, ch1_hint = ch1_hint) # your tracked spots
        yaml_file = self.tracked_folder._openyaml()
        ch0_laser = None
        if yaml_file:
            ch0_laser = yaml_file['ch0']
            ch1_laser = yaml_file['ch1']
        
        try:
            self.tracked = self.tracked_folder.open_files()
        except Exception as e:
            if verbose:
                print(f"Warning: could not open tracked files: {e}")
            self.tracked = None
        
        try:
            if ch0_laser:
                self.clusters = Cell_Analyzer(folder, ch0_laser + 'nm', ch1_laser + 'nm')
            elif ch0_hint:
                self.clusters = Cell_Analyzer(folder, ch0_hint, ch1_hint)
            else:
                self.clusters = Cell_Analyzer(folder)
            
            self.spots_outside_clusters = {wl: None for wl in self.clusters.channels}
            self.cluster_and_spots_stats = {wl: None for wl in self.clusters.channels}
            self.tracks_outside_clusters = {wl: None for wl in [self.clusters.ch0_wl, self.clusters.ch1_wl]}
            self.tracks_outside_clusters_stats = {wl: None for wl in [self.clusters.ch0_wl, self.clusters.ch1_wl]}
            self.cotracks_outside_clusters = None
            self.cotracks_outside_clusters_stats = None
            
            self._load_previous_results()
            self.nm2px = self.clusters.nm2px
        
        except RuntimeError as e:
            # Catch missing ROI or other initialization errors
            if verbose:
                print(f"Warning: could not open images or ROIs files: {e}")
            self.clusters = None
          
    def _load_previous_results(self):
        """
            Load previously saved results from disk if they exist.
        
            This includes:
            - Filtered spots outside clusters (CSV)
            - Cluster and spots statistics (CSV)
            - Tracks outside clusters (CSV + HDF)
            - Co-localized tracks outside clusters (CSV + HDF)
        
            Notes
            -----
            Results are loaded into the corresponding class attributes:
            `spots_outside_clusters`, `cluster_and_spots_stats`,
            `tracks_outside_clusters`, `tracks_outside_clusters_stats`,
            `cotracks_outside_clusters`, and `cotracks_outside_clusters_stats`.
    """
        cluster_dir = os.path.join(self.folder, "cluster_analysis_spots_filtered")
        
        # Load filtered spots
        for ch in self.clusters.channels:
            spots_file = os.path.join(cluster_dir, f'{ch}_roi_locs_nm.csv')
            if os.path.exists(spots_file):
                self.spots_outside_clusters[ch] = pd.read_csv(spots_file)
        # load cluster and spots stats        
        for ch in self.clusters.channels:
            stats_file = os.path.join(cluster_dir, f'{ch}_clusters_and_spots_stats.csv')
            if os.path.exists(stats_file):
                self.cluster_and_spots_stats[ch] = pd.read_csv(stats_file)
        
        # Load tracks_outside_clusters and tracks_outside_clusters_stats
        for ch in [self.clusters.ch0_wl, self.clusters.ch1_wl]:
            tracks_file = os.path.join(cluster_dir, f"{ch}_roi_locs_nm_trackpy.csv")
            stats_file  = os.path.join(cluster_dir, f"{ch}_roi_locs_nm_trackpy_stats.hdf")
            
            if os.path.exists(tracks_file) and os.path.exists(stats_file):
                self.tracks_outside_clusters[ch] = pd.read_csv(tracks_file)
                self.tracks_outside_clusters_stats[ch] = pd.read_hdf(stats_file, key='df')
        
        # Load co-tracking results
        coloc_csv  = os.path.join(cluster_dir, f"{self.clusters.ch0_wl}_roi_locs_nm_trackpy_ColocsTracks.csv")
        coloc_hdf  = os.path.join(cluster_dir, f"{self.clusters.ch0_wl}_roi_locs_nm_trackpy_ColocsTracks_stats.hdf")
    
        if os.path.exists(coloc_csv) and os.path.exists(coloc_hdf):
            self.cotracks_outside_clusters = pd.read_csv(coloc_csv)
            self.cotracks_outside_clusters_stats = pd.read_hdf(coloc_hdf, key='df')

    def combine_spots_clusters(self, min_size=80, ch='ch0', th_method='li_local',
                           global_th_mode='max', window_size=15, p=2, q=6, save_videos=False, overwrite = False):
        """
        Combine tracked spots with cluster analysis and compute spot statistics, it includes:
                 - checks if clusters have been found by checking if the results have been reloaded. If they are not, 
                 runs analysze clusters from Cell_Analyzer 
                 - From the spots found by SPIT, it removes the ones inside a cluster. 
                 - Calculate the mean intensities of the spots and saves them together in the file with the filtered spots. 
                 - calculates the number of spots, their mean intensity and the standard deviation and couples it to the 
                 clusters stats. 
    
        Parameters
        ----------
        min_size : int, optional
            Minimum size of clusters to consider, by default 80
        ch : str, optional
            Channel to process ('ch0' or 'ch1'), by default 'ch0'
        th_method : str, optional
            Local thresholding method, by default 'li_local'
        global_th_mode : str, optional
            Global threshold mode for clusters, by default 'max'
        window_size : int, optional
            Window size for local thresholding, by default 15
        p : int, optional
            Local threshold parameter, by default 2
        q : int, optional
            Local threshold parameter, by default 6
        save_videos : bool, optional
            Save centroid videos for clusters, by default False
        overwrite : bool, optional
            Overwrite existing results, by default False
    
        Returns
        -------
        tuple
            clusters_binary : np.ndarray
                Binary mask of clusters per frame.
            clusters_frame : pd.DataFrame
                Cluster labels per frame.
            results_stats : pd.DataFrame
                Cluster and spot statistics merged.
            linked_df : pd.DataFrame
                Linked clusters per cell.
            linked_stats : pd.DataFrame
                Statistics of linked clusters.
            spots_filtered : pd.DataFrame
                Spots filtered outside clusters.
        """
        if ch == 'ch0':
            wl = self.clusters.ch0_wl
        elif ch == 'ch1':
            wl = self.clusters.ch1_wl
        results_in_memory = (
                self.spots_outside_clusters.get(wl) is not None and
                self.cluster_and_spots_stats.get(wl) is not None
                         )
        if not overwrite and results_in_memory:
            print('Returning results already in memory')
            return None, self.clusters.result_cluster_analysis[wl], \
                self.cluster_and_spots_stats[wl], self.clusters.linked_clusters[wl], \
                self.clusters.linked_clusters_stats[wl], self.spots_outside_clusters[wl]
                
        steps = ["Cluster analysis", "Remove spots within clusters", 
                 "Compute mean intensity", "Merge stats & save videos"]
    
        # Use tqdm over the list of steps
        for step in tqdm(steps, desc="combine_spots_clusters progress", ncols=100):
            if step == "Cluster analysis":
                if (self.clusters.result_cluster_analysis[wl] is not None and
                    self.clusters.summary_cluster_analysis[wl] is not None and
                    not overwrite):
                    # Already loaded, skip analysis
                    clusters_binary = self.clusters.clusters_binary[wl]
                    clusters_frame = self.clusters.result_cluster_analysis[wl]
                    results_stats = self.clusters.summary_cluster_analysis[wl]
                    linked_df = self.clusters.linked_clusters[wl]
                    linked_stats = self.clusters.linked_clusters_stats[wl]
                else:
                    clusters_binary, clusters_frame, results_stats, linked_df, linked_stats = \
                        self.clusters.analyze_clusters_protein(min_size=min_size, ch=ch, th_method=th_method,
                                                           global_th_mode=global_th_mode, window_size=window_size,
                                                           p=p, q=q, save_videos=False, overwrite=overwrite)
            elif step == "Remove spots within clusters":
                try:
                    spots_filtered = self.remove_spots_within_clusters(ch)
                except:
                    spots_filtered = pd.DataFrame(columns=['cell_id', 't', 'x_per_cell', 'y_per_cell'])
    
            elif step == "Compute mean intensity":
                mean_intensities = []
                norm_mean_intensities = []
            
                for cell_id, group in spots_filtered.groupby('cell_id'):
                    img_stack = self.clusters.sep_cells[cell_id][wl]
                    # mask = self.clusters._create_mask(img_stack[0].shape, self.clusters._contours[cell_id])
                    median_int_out = np.median([
                                np.median(
                                    img[~self.clusters._create_mask(
                                        img.shape,
                                        self.clusters._get_contour(cell_id, frame_num)
                                    )]
                                )
                                for frame_num, img in enumerate(img_stack)
                            ])
            
                    for idx, spot in group.iterrows():
                        t = int(spot["t"])
                        x = spot["x_per_cell"]
                        y = spot["y_per_cell"]
                        img = img_stack[t]
                        mean_intensity = self._compute_mean_intensity(img, x, y, window=1)
                        mean_intensities.append(mean_intensity)
                        norm_mean_intensities.append(mean_intensity / median_int_out)
            
                spots_filtered["intensity_im"] = mean_intensities
                spots_filtered["norm_intensity_im"] = norm_mean_intensities

    
            elif step == "Merge stats & save videos":
                spot_stats = (
                    spots_filtered.groupby(['cell_id', 't'])
                    .agg(
                        num_spots=('intensity_im', 'count'),
                        spot_mean_intensity=('intensity_im', 'mean'),
                        spot_std_intensity=('intensity_im', 'std'),
                        spot_norm_mean_intensity=('norm_intensity_im', 'mean'),
                        spot_norm_std_intensity=('norm_intensity_im', 'std')
                    )
                    .reset_index()
                    .rename(columns={'t': 'frame'})
                )
                
                expected_columns = ['num_spots', 'spot_mean_intensity','spot_std_intensity','spot_norm_mean_intensity','spot_norm_std_intensity']

                try:
                    results_stats = results_stats.merge(
                        spot_stats,
                        on=['cell_id', 'frame'],
                        how='left'
                    )
                
                except Exception as e:
                    print(f"Merge failed likely due to the lack of clusters: {e}")
                    for col in expected_columns:
                        results_stats[col] = np.nan

                    
                if save_videos:
                    try:
                        self.clusters._save_centroid_videos_per_cell(
                            clusters_binary, clusters_frame, self.clusters.sep_cells, ch=ch,
                            square_size=2, filtered_spots=spots_filtered, output_dir="cluster_analysis_spots_filtered"
                        )
                    except:
                        pass
        if self.folder is not None:
            output_dir = os.path.join(self.folder, 'cluster_analysis_spots_filtered')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # clusters_frame.to_csv(os.path.join(output_dir, f'clusters_{wl}.csv'))   
        results_stats.to_csv(os.path.join(output_dir, f'{wl}_clusters_and_spots_stats.csv'), index = False) 
        spots_filtered.to_csv(os.path.join(output_dir, f'{wl}_roi_locs_nm.csv'), index = False) 
        self.cluster_and_spots_stats[wl] = results_stats
        
        return clusters_binary, clusters_frame, results_stats, linked_df, linked_stats, spots_filtered
    def remove_spots_within_clusters(self, ch='ch0'):
        """
            Remove spots that fall inside clusters for a given channel.
        
            Parameters
            ----------
            ch : str, optional
                Channel to process ('ch0' or 'ch1'), by default 'ch0'
        
            Returns
            -------
            pd.DataFrame
                DataFrame of spots outside clusters with x/y coordinates converted to pixels.
        
            Raises
            ------
            RuntimeError
                If tracked data is missing.
            ValueError
                If the channel is invalid.
    """
        if self.tracked is None:
            raise RuntimeError("Tracked files are missing or tracking failed for this folder. Cannot proceed.")
        if ch == 'ch0':
            wl = self.clusters.ch0_wl
        elif ch == 'ch1':
            wl = self.clusters.ch1_wl
        
        clusters = self.clusters.result_cluster_analysis[wl].copy()
        if ch == 'ch0':
            spots = self.tracked.tracks0.copy()
        elif ch == 'ch1':
            spots = self.tracked.tracks1.copy()
        else:
            raise ValueError(f'Channel {ch} is not valid. Use ch0 or ch1.')

        cells = list(spots['cell_id'].drop_duplicates())
        unique_contour = self.tracked.stats0.loc[self.tracked.stats0['contour'].apply(lambda 
                                                x: str(x)).drop_duplicates().index, 'contour']
        
        corrections = {}
        for cell, contour in zip(cells, unique_contour):
            x0 = int(min(contour[:, 0]))
            y0 = int(min(contour[:, 1]))
            corrections[cell] = (x0, y0)
        spots[['x_per_cell', 'y_per_cell']] = spots.apply(
            lambda row: pd.Series([
                (row['x']/self.nm2px) - corrections[row['cell_id']][0],
                (row['y']/self.nm2px) - corrections[row['cell_id']][1]
            ]),
            axis=1
        )
        # print(clusters)
        filtered_spots = self._filter_spots_outside_clusters(spots, clusters)
        cols_to_drop = ['locID', 'track.id', 'loc_count', 'seg.id']
        spots_to_save = filtered_spots.drop(columns=[c for c in cols_to_drop if c in filtered_spots.columns])
        spots_to_save = tools.df_convert2px(spots_to_save, self.nm2px)
        self.spots_outside_clusters[wl] = spots_to_save

        return spots_to_save
    def retrack(self, overwrite = False):
        """
            Re-link spots outside clusters using Trackpy, and the same settings and method used for SPIT
        
            Parameters
            ----------
            overwrite : bool, optional
                If True, recompute tracks even if they exist, by default False
        
            Notes
            -----
            - Requires `combine_spots_clusters()` to have been run first.
            - Updates `tracks_outside_clusters` and `tracks_outside_clusters_stats`.
            - Tracks are only kept if they have more than one localization.
        """
        try:
            if not any(v is not None for v in self.spots_outside_clusters.values()):
                raise RuntimeError(
                    "You must run combine_spots_clusters() on at least one channel before retrack()."
                )
    
            
            # Load Trackpy linking settings
            yaml_files = glob(os.path.join(self.folder, "*trackpy.yaml"))
            if not yaml_files:
                raise FileNotFoundError("No trackpy YAML file found in the folder.")
            with open(yaml_files[0], 'r') as f:
                link_settings = yaml.safe_load(f)
        
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
        
            # Loop through channels and process only those with data
            for ch, spots in self.spots_outside_clusters.items():
                if spots is None:
                    print(f"Skipping {ch}: combine_spots_clusters() has not been run for this channel or there are no spots.")
                    continue
                
                if (self.tracks_outside_clusters.get(ch) is not None and
                self.tracks_outside_clusters_stats.get(ch) is not None and
                not overwrite):
                    print(f"Skipping {ch}: retracking results already loaded in memory.")
                    continue
                
                print(f"Ret­racking {ch} ...")
                # Drop old Trackpy-specific columns if present
                df_locs_clean = spots.rename_axis('locID').reset_index()
                # Convert nm → px
                # df_locs_clean = tools.df_convert2px(df_locs_clean, self.nm2px)
        
                # Re-link with Trackpy
                # df_tracksTP = link.link_locs_trackpy(
                #     df_locs_clean,
                #     search=link_settings['search'],
                #     memory=link_settings['memory']
                # )
                
                tracks_list = []
                next_track_id = 0  # global track counter
                
                for cid, group in df_locs_clean.groupby("cell_id"):
                    group_sorted = group.sort_values("t")  # ensure t is increasing
                    linked = link.link_locs_trackpy(
                        group_sorted,
                        search=link_settings['search'],
                        memory=link_settings['memory']
                    )
                
                    # offset track IDs to make them globally unique
                    if not linked.empty:
                        linked["track.id"] += next_track_id
                        next_track_id = linked["track.id"].max() + 1
                
                    linked["cell_id"] = cid
                    tracks_list.append(linked)
                
                df_tracksTP = pd.concat(tracks_list, ignore_index=True)
                
                df_tracksTP = tools.df_convert2nm(df_tracksTP, self.nm2px)
                df_tracksTP['seg.id'] = df_tracksTP['track.id']
               # Keep only tracks with more than 1 localization
                track_counts = df_tracksTP['track.id'].value_counts()
                tracks_to_keep = track_counts[track_counts > 1].index
                df_filtered = df_tracksTP[df_tracksTP['track.id'].isin(tracks_to_keep)].copy()
                
                # Remove rows with NaNs in critical columns
                df_filtered = df_filtered.dropna(subset=['x', 'y', 't'])
                
                # Ensure numeric types
                df_filtered[['x','y','t']] = df_filtered[['x','y','t']].apply(pd.to_numeric, errors='coerce')
                df_filtered = df_filtered.dropna(subset=['x','y','t'])
                
                
                
                # Now compute stats safely
                df_stats = link.get_particle_stats(df_filtered, dt=dt, particle='track.id', t='t')
    
                # Add ROI info by cell_id
                roi_info = (
                    self.tracked.stats0[['path', 'contour', 'area', 'centroid', 'cell_id']]
                    .drop_duplicates(subset=['cell_id'])
                )
                
                df_stats = df_stats.merge(roi_info, on='cell_id', how='left')
        
                # Convert back px → nm
                
                output_dir = os.path.join(self.folder, 'cluster_analysis_spots_filtered')
                
                df_tracksTP.to_csv(os.path.join(output_dir, f"{ch}_roi_locs_nm_trackpy.csv"), index = False)
                df_stats.to_hdf(os.path.join(output_dir, f"{ch}_roi_locs_nm_trackpy_stats.hdf"), key = 'df')
                # Save results
                self.tracks_outside_clusters[ch] = df_tracksTP
                self.tracks_outside_clusters_stats[ch] = df_stats
        
                print(f"Finished retracking {ch} ({len(df_tracksTP)} tracks).")
        except:
            self.tracks_outside_clusters[ch] = None
            self.tracks_outside_clusters_stats[ch] = None
    def recoloc_tracks(self, overwrite = False):
            """
            Perform co-localization analysis on re-tracked spots outside clusters using the same method and settings as SPIT.
        
            Parameters
            ----------
            overwrite : bool, optional
                If True, recompute co-localized tracks even if they exist, by default False
        
            Notes
            -----
            - Requires `retrack()` to have been run first.
            - Loads co-localization parameters from `_colocsTracks.yaml`.
            """
            try:
                results_in_memory = (
                        self.cotracks_outside_clusters is not None and
                            self.cotracks_outside_clusters_stats is not None
                            )
                if results_in_memory and not overwrite:
                    print('Skipping: recoloc results already loaded in memory.')
                    return
                
                
                if not any(v is not None for v in self.tracks_outside_clusters.values()):
                    raise RuntimeError(
                        "You must run retrack() before recoloc_tracks()."
                    )
                
                # 1. load coloc settings from YAML
                yaml_files = glob(os.path.join(self.folder, "*_colocsTracks.yaml"))
                if not yaml_files:
                    raise FileNotFoundError("No *_colocsTracks.yaml found in the folder.")
                yaml_file = yaml_files[0]
                with open(yaml_file, "r") as f:
                    coloc_settings_all = list(yaml.safe_load_all(f))
                coloc_settings = coloc_settings_all[-1]
                # Create local aliases for channel-specific filetered tracks made from the filteres spots stored in the object
                df_locs_ch0 = self.tracks_outside_clusters[self.clusters.ch0_wl]
                df_locs_ch1 = self.tracks_outside_clusters[self.clusters.ch1_wl]
                # 2. run coloc analysis
                df_colocs, coloc_stats = coloc.coloc_tracks(
                    df_locs_ch0,
                    df_locs_ch1,
                    leng=coloc_settings['min_len_track'] ,
                    max_distance=coloc_settings['th'],
                    n=coloc_settings['min_overlapped_frames']
                )
                # 3. Attach ROI info (if available in stats0)
                if hasattr(self.tracked, "stats0"):
                    roi_info = (
                        self.tracked.stats0[["path", "contour", "area", "centroid", "cell_id"]]
                        .drop_duplicates(subset=["cell_id"])
                    )
                    coloc_stats = coloc_stats.merge(roi_info, on="cell_id", how="left")
            
                # 4. Save results to self and to folder
                output_dir = os.path.join(self.folder, 'cluster_analysis_spots_filtered')
                df_colocs.to_csv(os.path.join(output_dir, f"{self.clusters.ch0_wl}_roi_locs_nm_trackpy_ColocsTracks.csv"), index = False)
                coloc_stats.to_hdf(os.path.join(output_dir, f"{self.clusters.ch0_wl}_roi_locs_nm_trackpy_ColocsTracks_stats.hdf"), key = 'df')
                self.cotracks_outside_clusters = df_colocs
                self.cotracks_outside_clusters_stats = coloc_stats
            except: 
                self.cotracks_outside_clusters = None
                self.cotracks_outside_clusters_stats = None
    def extract_Ds_filtered(self, mature_class = 1, min_len = 10,  ch = 'ch0'):
        """
        Extract diffusion coefficients (D_msd) for tracks outside clusters filtered by cell maturation.
    
        Parameters
        ----------
        mature_class : int, optional
            Maturation category to filter cells, by default 1
        min_len : int, optional
            Minimum track length to include, by default 10
        ch : str, optional
            Channel to process ('ch0' or 'ch1'), by default 'ch0'
    
        Returns
        -------
        pd.DataFrame
            DataFrame containing columns ['track.id', 'cell_id', 'D_msd'].
    
        Raises
        ------
        RuntimeError
            If `retrack()` has not been run or maturation data is missing.
        """
        if not any(v is not None for v in self.tracks_outside_clusters.values()):
            raise RuntimeError(
                "You must run retrack() before extract_Ds_filtered()."
            )
        if not any(v is not None for v in self.clusters.maturation.values()):
            raise RuntimeError(
                "You must run clusters.predict_maturation() before get_Ds_filtered()."
            )
        if ch == 'ch0':
            wl = self.clusters.ch0_wl
        elif ch == 'ch1':
            wl = self.clusters.ch1_wl  
        maturation = self.clusters.maturation[wl]
        
        if not any(v is not None for v in self.clusters.maturation.values()):
            print("Warning: maturation is empty. Returning empty DataFrame.")
            return pd.DataFrame(columns=['track.id', 'cell_id', 'D_msd'])
        else:
            if mature_class:
                mature_cells = maturation.loc[maturation['category'] == mature_class, 'cell']
            else:
                mature_cells = maturation.cell
            stats = self.tracks_outside_clusters_stats[wl] 
            columns_to_extract = ['track.id', 'cell_id', 'D_msd']
            ds = stats[
            (stats['length'] >= min_len) & (stats['cell_id'].isin(mature_cells))][columns_to_extract]
            ds = ds.dropna(subset=['D_msd'])
            return ds
    def extract_dwell_filtered(self,mature_class = 1, frame_rate = None, min_len=10, 
                               max_dist = 250, ref='ch0', ch_maturation_selection = 'ch1'):
        """
            Extract dwell times of co-localized tracks filtered by cell maturation.
        
            Parameters
            ----------
            mature_class : int, optional
                Maturation category to filter cells, by default 1
            frame_rate : float, optional
                Frame interval in seconds. If None, read from result.txt, by default None
            min_len : int, optional
                Minimum number of frames to include, by default 10
            max_dist : float, optional
                Maximum distance in nm to consider co-localized, by default 250
            ref : str, optional
                Reference channel for dwell time calculation ('ch0' or 'ch1'), by default 'ch0'
            ch_maturation_selection : str, optional
                Channel used to select maturation classes, by default 'ch1'
        
            Returns
            -------
            pd.DataFrame
                DataFrame with columns ['colocID', 'track.id_ref', 'track.id_binds', 'cell_id', 'dwell_time'].
        
            Raises
            ------
            RuntimeError
                If `recoloc_tracks()` has not been run or maturation data is missing.
        """
        if self.cotracks_outside_clusters is None or self.cotracks_outside_clusters_stats is None:
            raise RuntimeError("You must run recoloc_tracks() before extract_dwell_filtered().")
        if not any(v is not None for v in self.clusters.maturation.values()):
            raise RuntimeError("You must run clusters.predict_maturation() before extract_dwell_filtered().")

        stats = self.cotracks_outside_clusters_stats
        tracks = self.cotracks_outside_clusters
        # Extract dt (frame interval) from result.txt
        
        if not frame_rate: 
            result_files = glob(os.path.join(self.folder, "*result.txt"))
            if not result_files:
                raise FileNotFoundError("No result.txt file found in the folder.")
            else: 
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
                frame_rate = dt
        
        if ch_maturation_selection == 'ch0':
            wl_mat = self.clusters.ch0_wl
        elif ch_maturation_selection == 'ch1':
            wl_mat = self.clusters.ch1_wl  
        if ref == 'ch0':
            wl = self.clusters.ch0_wl
        elif ref== 'ch1':
            wl = self.clusters.ch1_wl 
        
        maturation = self.clusters.maturation[wl_mat]
        mature_cells = maturation.loc[maturation['category'] == mature_class, 'cell'].tolist()
                
        if all(isinstance(obj, pd.DataFrame) for obj in [stats, tracks]):
            unique_colocIDs = stats[
            (stats['num_frames_coloc'] > min_len) &
            (stats['cell_id'].isin(mature_cells))
        ]['colocID'].unique()
            data = []
            for i in unique_colocIDs:#[7:8]:
                colocalized_df = tracks[(tracks['colocID'] == i) & (tracks['distance'] <= max_dist) & pd.notna(tracks['x']) & pd.notna(tracks['y'])]
                if not colocalized_df.empty:
                    start_time = colocalized_df['t'].min()*frame_rate
                locs_ch0 = tracks[tracks.colocID == i][['x_0', 'y_0', 't', 'intensity_0']]
                locs_ch0 = locs_ch0.dropna(axis = 0)
                locs_ch1 = tracks[tracks.colocID == i][['x_1', 'y_1', 't', 'intensity_1']]
                locs_ch1 = locs_ch1.dropna(axis = 0)
                times_0 = {'ch0': locs_ch0.t, 'ch1': locs_ch1.t}.get(ref)
                if min(times_0) * frame_rate < start_time:
                    dt = start_time - min(times_0) * frame_rate
                    # Create a Series with the same columns
                    other_ref = 'ch1' if ref == 'ch0' else 'ch0'
                    track_ids = {'ch0': stats[stats.colocID == i]['track.id0'].values[0], 'ch1': stats[stats.colocID == i]['track.id1'].values[0]}
                    data.append([i, track_ids.get(ref), track_ids.get(other_ref),
                                          
                                          stats[stats.colocID == i]['cell_id'].values[0],
                                          dt])
                    
            dwell_times = pd.DataFrame(data, columns=['colocID', 'track.id_ref', 'track.id_binds', 'cell_id', 'dwell_time'])
            
            return dwell_times
        else:
            return None
    def _filter_spots_outside_clusters(self, spots_df, clusters_df):
        """
            Filter spots that fall outside clusters for a given DataFrame of spots.
        
            Parameters
            ----------
            spots_df : pd.DataFrame
                DataFrame containing spot coordinates (x_per_cell, y_per_cell) and frame index 't'.
            clusters_df : pd.DataFrame
                DataFrame containing cluster contours per frame.
        
            Returns
            -------
            pd.DataFrame
                Filtered spots outside clusters.
        """
        keep_mask = []

        for idx, spot in spots_df.iterrows():
            frame = spot['t']
            x, y = spot['x_per_cell'], spot['y_per_cell']
            contours = clusters_df[clusters_df['frame'] == int(frame)]['contour']

            inside_any = False
            for contour in contours:
                if contour.size == 0:
                    continue
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    inside_any = True
                    break
            keep_mask.append(not inside_any)

        filtered_spots = spots_df[keep_mask].reset_index(drop=True)
        return filtered_spots
    def _compute_mean_intensity(self, image, x, y, window=1):
        return compute_mean_intensity(image, x, y, window)