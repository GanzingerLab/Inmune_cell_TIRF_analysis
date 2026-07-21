import json
import os
import pandas as pd
import re

from glob import glob
from tqdm import tqdm

from .io_utils import get_time_interval, openyaml
from .combined import Combined_analysis
from .tracked import Single_tracked_folder
from .plotting import BoxPlotter, HistogramPlotter


class Dataset_combined_analysis:
    """
    High-level manager for batch analysis of multiple experimental conditions and runs.

    The `Dataset_combined_analysis` class automates dataset-level operations
    on TIRF tracking/imaging experiments. It scans all experimental conditions and their
    corresponding ``Run#`` subfolders, initializes per-run analyses, and performs
    processing such as cluster analysis, retracking, colocalization, diffusion coefficient
    extraction, and dwell time analysis.

    This class serves as a wrapper around lower-level components like
    `Combined_analysis` and `Tracked_image`, ensuring that
    all runs within a dataset are processed in a consistent and reproducible manner.

    Parameters
    ----------
    folder : str
        Path to the root dataset directory. The directory must contain one or more
        subfolders representing experimental conditions. Each condition folder
        should contain subfolders named ``Run#`` (e.g., ``Run1``, ``Run2``),
        which store the analysis data for that run.

    Attributes
    ----------
    folder : str
        Path to the root dataset directory.
    conditions : list of str
        All condition folder names detected in the dataset.
    conditions_to_use : list of str
        Subset of conditions selected for analysis. Defaults to all detected conditions.
    _conditions_paths : list of str
        Full paths to each selected condition directory.
    run_paths : list of str
        List of full paths to all detected ``Run#`` folders across all selected conditions.
    failed_folders : list
        List of tuples ``(path, error, stage)`` describing any failed analyses.
    result_count : pandas.DataFrame or None
        Table storing the most recent results from track-counting analyses.
    ch0_hint : str
        Wavelength identifier (e.g., ``'488nm'``) for channel 0, parsed from YAML metadata.
    ch1_hint : str
        Wavelength identifier (e.g., ``'561nm'``) for channel 1, parsed from YAML metadata.
    """
    def __init__(self, folder):
        self.folder = folder
        self.conditions = self._get_conditions()
        self.conditions_to_use = self.conditions
        self._conditions_paths = [os.path.join(self.folder, subfolder) for subfolder in self.conditions_to_use]
        self.failed_folders = []
        self.result_count = None
        self.run_paths = []
        self._collect_run_paths()
        path_yaml = glob(self.folder + r'/**/*_colocsTracks.yaml', recursive=True)
        general_yaml_file = openyaml(path_yaml)
        self.ch0_hint = general_yaml_file['ch0']+'nm'
        self.ch1_hint = general_yaml_file['ch1']+'nm'
        
    def _get_conditions(self):
        return [f.name for f in os.scandir(self.folder) if f.is_dir()]

    def _collect_run_paths(self):
        run_pattern = re.compile(r'^Run\d+$')
        for cond_path in self._conditions_paths:
            for root, dirs, files in os.walk(cond_path):
                matching_dirs = [d for d in dirs if run_pattern.match(d)]
                for d in matching_dirs:
                    full_path = os.path.join(root, d)
                    self.run_paths.append(full_path)
                dirs[:] = [d for d in dirs if not run_pattern.match(d)]
    def select_conditions(self, indices):
        self.conditions_to_use = [self.conditions[i] for i in indices]
        self._conditions_paths = [os.path.join(self.folder, subfolder) for subfolder in self.conditions_to_use]

        # Filter run_paths to only include selected conditions
        self.run_paths = [p for p in self.run_paths if any(cond in p for cond in self.conditions_to_use)]
    def analyze_clusters_protein(self, min_size=80, ch='ch0', th_method='li_local', global_th_mode='max',
                             window_size=15, p=2, q=6, save_videos=False, overwrite=False, verbose=True):
        """
        Run Cell_analyzer.clusters.analyze_clusters_protein on all runs in this dataset.
        """
        for run_path in self.run_paths:
            try:
                if verbose:
                    print(f"Analyzing clusters: {run_path}")
                ca = Combined_analysis(run_path,ch0_hint=self.ch0_hint, ch1_hint=self.ch1_hint, verbose=verbose)
                ca.clusters.analyze_clusters_protein(
                    min_size=min_size,
                    ch=ch,
                    th_method=th_method,
                    global_th_mode=global_th_mode,
                    window_size=window_size,
                    p=p,
                    q=q,
                    save_videos=save_videos,
                    overwrite=overwrite
                )
            except Exception as e:
                if verbose:
                    print(f"Failed cluster analysis on {run_path}: {e}")
                self.failed_folders.append([run_path, e, 'analyze_clusters_protein'])
    def predict_maturation(self, model, preprocess_frame_func=None, save_plot=False, 
                       N_rolling=5, ch='ch0', r2_thresh=0.75, low_thresh=0.4, high_thresh=0.6,
                       overwrite=False, verbose=True):
        """
        Run Cell_analyzer.predict_maturation on all runs in this dataset.
        """
        for run_path in self.run_paths:
            try:
                if verbose:
                    print(f"Predicting maturation: {run_path}")
                ca = Combined_analysis(run_path,ch0_hint=self.ch0_hint, ch1_hint=self.ch1_hint, verbose=verbose)
                ca.clusters.predict_maturation(
                    model=model,
                    preprocess_frame_func=preprocess_frame_func,
                    save_plot=save_plot,
                    N_rolling=N_rolling,
                    ch=ch,
                    r2_thresh=r2_thresh,
                    low_thresh=low_thresh,
                    high_thresh=high_thresh,
                    overwrite=overwrite
                )
            except Exception as e:
                if verbose:
                    print(f"Failed maturation prediction on {run_path}: {e}")
                self.failed_folders.append([run_path, e, 'predict_maturation'])
    def combine_spots_clusters(self, min_size=80, ch='ch0', th_method='li_local',
                    global_th_mode='max', window_size=15, p=2, q=6, save_videos=True, verbose=True, overwrite = False):
        """
        Run Combined_analysis.combine_spots_clusters on all runs in this dataset.
        Saves outputs in each run folder automatically.
        """

        for run_path in self.run_paths:
            try:
                print(f'Analyzing: {run_path}')
                ca = Combined_analysis(run_path,ch0_hint=self.ch0_hint, ch1_hint=self.ch1_hint, verbose=verbose)
                if ca.clusters is None:
                    if verbose:
                        print(f"Skipping {run_path}: cluster initialization failed.")
                    self.failed_folders.append(run_path)
                    continue

                _ = ca.combine_spots_clusters(
                    min_size=min_size, ch=ch, th_method=th_method,
                    global_th_mode=global_th_mode, window_size=window_size,
                    p=p, q=q, save_videos=save_videos, overwrite=overwrite
                )
                # if verbose:
                    # print(f"Finished combine_spots_clusters for {run_path}")

            except Exception as e:
                if verbose:
                    print(f"Failed on {run_path}: {e}")
                self.failed_folders.append([run_path, e, 'combine_spots_clusters'])
                
    def retrack(self, verbose=True, overwrite = False):
        """
        Run Combined_analysis.retrack on all runs in this dataset.
        Saves outputs in each run folder automatically.
        """
        for run_path in self.run_paths:
            try:
                print(f'Analyzing: {run_path}')
                ca = Combined_analysis(run_path, verbose=verbose)
                if not any(v is not None for v in ca.spots_outside_clusters.values()):
                    if verbose:
                        print(f"Skipping {run_path}: combine_spots_clusters() not run yet.")
                    self.failed_folders.append(run_path)
                    continue

                ca.retrack(overwrite=overwrite)

                # if verbose:
                #     print(f"Finished retrack for {run_path}")

            except Exception as e:
                if verbose:
                    print(f"Failed retrack on {run_path}: {e}")
                self.failed_folders.append([run_path, e, 'retrack'])

    def recoloc_tracks(self, verbose=True, overwrite= False):
        """
        Run Combined_analysis.recoloc_tracks on all runs in this dataset.
        Saves outputs in each run folder automatically.
        """
        for run_path in self.run_paths:
            try:
                ca = Combined_analysis(run_path, verbose=verbose)
                if not any(v is not None for v in ca.tracks_outside_clusters.values()):
                    if verbose:
                        print(f"Skipping {run_path}: retrack() not run yet.")
                    self.failed_folders.append(run_path)
                    continue

                ca.recoloc_tracks(overwrite=overwrite)

                if verbose:
                    print(f"Finished recoloc_tracks for {run_path}")

            except Exception as e:
                if verbose:
                    print(f"Failed recoloc_tracks on {run_path}: {e}")
                self.failed_folders.append([run_path, e, 'recoloc_tracks'])
    #predict maturity and analyze clusters
    def count_number_cotracks(self,mature_class=1, min_len = 5,
                              ch_maturation_selection = 'ch1', source = 'tracked'):
        """
        Count the number of total and co-localized tracks across all run folders.
    
        This method iterates through all experimental conditions and run folders,
        and depending on the ``source`` parameter, extracts track statistics
        either from:
        
        - Raw tracked data (`source='tracked'`), loaded using :class:`Single_tracked_folder`
        - Filtered data outside clusters (`source='filtered'`)
        - Filtered and maturation-classified data (`source='filtered_mature'`)
    
        It counts:
        - The number of valid tracks in each fluorescence channel (``ch0`` and ``ch1``)
        - The number of co-localized tracks across both channels
    
        The results are compiled into a DataFrame stored in
        ``self.result_count`` and also returned.
    
        Parameters
        ----------
        mature_class : int, optional
            Maturation class to use for filtering cells.
            Only relevant if ``source='filtered_mature'``.
            Default is ``1``.
        min_len : int, optional
            Minimum number of frames (localizations) a track must have to be counted.
            Default is ``5``.
        ch_maturation_selection : str, optional
            Channel to use for selecting maturation category.
            Must be one of ``'ch0'`` or ``'ch1'``.
            Default is ``'ch1'``.
        source : str, optional
            Data source to use. One of:
            
            - ``'tracked'``: counts from the original tracking results
            - ``'filtered'``: counts from filtered tracks outside clusters
            - ``'filtered_mature'``: counts from filtered and maturation-classified tracks
            
            Default is ``'tracked'``.
    
        Returns
        -------
        pandas.DataFrame
            A DataFrame summarizing per-run and per-condition counts, with the following columns:
            
            - ``folder``: str — full path of the run folder  
            - ``condition``: str — experimental condition name  
            - ``colocalized_tracks`` : int — number of co-localized tracks  
            - ``ch0_tracks``: int — number of tracks in channel 0  
            - ``ch1_tracks``: int — number of tracks in channel 1  
            - ``min_len_track``: int — minimum track length used for filtering
    """
        results = []
        for cond_path, cond_name in zip(self._conditions_paths, self.conditions_to_use):
            run_folders = []
            run_pattern = re.compile(r'^Run\d+$')
            for root, dirs, files in os.walk(cond_path):
                matching_dirs = [d for d in dirs if run_pattern.match(d)]
                for d in matching_dirs:
                    full_path = os.path.join(root, d)
                    run_folders.append(full_path)
            result_cond = []
        
            for run_folder in tqdm(run_folders):
                if source == "tracked":
                    try:
                        a = Single_tracked_folder(run_folder, self.ch0_hint, self.ch1_hint).open_files()
                        ch0_stats = a.stats0
                        ch1_stats = a.stats1
                        coloc_stats = a.coloc_stats
                    except Exception as exc:
                        ch0_stats = None
                        ch1_stats = None
                        coloc_stats = None
                        self.failed_folders.append(
                            (run_folder, f"{type(exc).__name__}: {exc}")
                        )
                        print(
                            f"Warning: could not load tracked statistics.\n"
                            f"Folder: {run_folder}\n"
                            f"Error: {type(exc).__name__}: {exc}"
                        )
                else:  # use Combined_analysis
                    a = Combined_analysis(run_folder, ch0_hint = self.ch0_hint, ch1_hint = self.ch1_hint, verbose=False)
                    if a.clusters is None:
                        continue
                    if source == "filtered":
                        ch0_stats = a.tracks_outside_clusters_stats[a.clusters.ch0_wl]
                        ch1_stats = a.tracks_outside_clusters_stats[a.clusters.ch1_wl]
                        coloc_stats = a.cotracks_outside_clusters_stats
                    elif source == "filtered_mature":
                        if ch_maturation_selection == 'ch0':
                            wl_mat = a.clusters.ch0_wl
                        elif ch_maturation_selection == 'ch1':
                            wl_mat = a.clusters.ch1_wl
                        else: 
                            print(f'{ch_maturation_selection} is not a valid ch_maturation_selection')
                        maturation = a.clusters.maturation[wl_mat]
                        if isinstance(maturation, pd.DataFrame):
                            cells = maturation[maturation.category == mature_class]['cell']    
                            if isinstance(a.tracks_outside_clusters_stats[a.clusters.ch0_wl], pd.DataFrame) and not a.tracks_outside_clusters_stats[a.clusters.ch0_wl].empty:
                                ch0_stats = a.tracks_outside_clusters_stats[a.clusters.ch0_wl][a.tracks_outside_clusters_stats[a.clusters.ch0_wl]['cell_id'].isin(cells)]
                            else: 
                                ch0_stats = None
                            if isinstance(a.tracks_outside_clusters_stats[a.clusters.ch1_wl], pd.DataFrame) and not a.tracks_outside_clusters_stats[a.clusters.ch1_wl].empty:
                                ch1_stats = a.tracks_outside_clusters_stats[a.clusters.ch1_wl][a.tracks_outside_clusters_stats[a.clusters.ch1_wl]['cell_id'].isin(cells)]
                            else: 
                                ch1_stats = None
                            if isinstance(a.cotracks_outside_clusters_stats, pd.DataFrame) and not a.cotracks_outside_clusters_stats.empty:
                                coloc_stats = a.cotracks_outside_clusters_stats[a.cotracks_outside_clusters_stats['cell_id'].isin(cells)]
                            else: 
                                coloc_stats = None
                        else:
                            continue
                    else:
                        raise ValueError(f"Unknown source={source}")
                
                    # Decide which set of stats to use
                    if isinstance(ch0_stats, pd.DataFrame) and not ch0_stats.empty:
                        # Count tracks for channel 0
                        ch0_tracks = ch0_stats[ch0_stats.loc_count  >= min_len].shape[0] 
                    else:
                        ch0_tracks = 0
                    if isinstance(ch1_stats, pd.DataFrame) and not ch1_stats.empty:
                        # Count tracks for channel 1
                        ch1_tracks = ch1_stats[ch1_stats.loc_count  >= min_len].shape[0]
                    else: 
                        ch1_tracks = 0
                    if isinstance(coloc_stats, pd.DataFrame) and not coloc_stats.empty:
                        # Count colocalized tracks
                        coloc_count = coloc_stats[coloc_stats['num_frames_coloc'] >= min_len].shape[0]
                    else:
                        coloc_count = 0
                results.append({
                    "folder": run_folder,
                    "condition": cond_name,
                    "colocalized_tracks": coloc_count,
                    "ch0_tracks": ch0_tracks,
                    "ch1_tracks": ch1_tracks,
                    "min_len_track": min_len,
                })
            
            
        self.result_count = pd.DataFrame(results)
        return self.result_count
    def summary_count_number_cotracks(self):
        """
        Summarize the total number of tracks and co-localized tracks per condition.
    
        This method aggregates the per-run results from `count_number_cotracks`
        by experimental condition, producing total counts of co-localized and
        individual channel tracks across all runs belonging to each condition.
    
        The method expects that `count_number_cotracks` has already been executed,
        and that its resulting DataFrame is stored in ``self.result_count``.
    
        Returns
        -------
        pandas.DataFrame
            A DataFrame summarizing total track counts per condition, with the following columns:
            
            - ``condition`` : str — experimental condition name  
            - ``total_colocalized_tracks`` : int — total number of co-localized tracks  
            - ``total_ch0_tracks`` : int — total number of tracks in channel 0  
            - ``total_ch1_tracks`` : int — total number of tracks in channel 1  
        """
        if isinstance(self.result_count, pd.DataFrame):
            aggregated_df = self.result_count.groupby('condition').agg(
                total_colocalized_tracks=pd.NamedAgg(column='colocalized_tracks', aggfunc='sum'),
                total_ch0_tracks=pd.NamedAgg(column='ch0_tracks', aggfunc='sum'),
                total_ch1_tracks=pd.NamedAgg(column='ch1_tracks', aggfunc='sum')
            ).reset_index()
            return aggregated_df
        else: 
            print("run count_number_cotracks first")
    # def combine_spots_clusters(self, min_size=80, ch='ch0', th_method='li_local',
                                   # global_th_mode='max', window_size=15, p=2, q=6, save_videos=False):    
    def get_Ds(self, mature_class=1, min_len=10, ch='ch0', source="tracked"):
        """
        Extract diffusion coefficients from tracked or filtered data.
        
        Parameters
        ----------
        min_len : int
            Minimum track length.
        channel : str
            Channel to use ('ch0' or 'ch1').
        source : str
            One of ["tracked", "filtered", "filtered_mature"].
        mature_class : int
            Which maturation category to use (only relevant if source="filtered_mature").
        """
        all_ds = []
        box = BoxPlotter(xlabel="Conditions", ylabel="Diff. Coeff. (um^2/sec)")
        
        for cond_path, cond_name in zip(self._conditions_paths, self.conditions_to_use):
            print(f"\nAnalyzing {cond_path}...")
            run_folders = []
            run_pattern = re.compile(r'^Run\d+$')
            for root, dirs, files in os.walk(cond_path):
                matching_dirs = [d for d in dirs if run_pattern.match(d)]
                for d in matching_dirs:
                    full_path = os.path.join(root, d)
                    run_folders.append(full_path)
            ds_cond = []
            
            for run_folder in tqdm(run_folders):
                if source == "tracked":
                    try:
                        image = Single_tracked_folder(run_folder, self.ch0_hint, self.ch1_hint).open_files()
                        ds = image.extract_Ds(min_len, ch)
                    except Exception as exc:
                        ds = pd.DataFrame()
                        self.failed_folders.append(
                            (run_folder, f"{type(exc).__name__}: {exc}")
                        )
                        print(
                            f"Warning: diffusion extraction failed.\n"
                            f"Channel: {ch}\n"
                            f"Folder: {run_folder}\n"
                            f"Error: {type(exc).__name__}: {exc}"
                        )
                    
                else:  # use Combined_analysis
                    analysis = Combined_analysis(run_folder, verbose=False)
                    if analysis.tracked is None or analysis.clusters is None:
                        continue
                    
                    if source == "filtered":
                        ds = analysis.extract_Ds_filtered(mature_class=None, min_len=min_len, ch=ch)
                    elif source == "filtered_mature":
                        ds = analysis.extract_Ds_filtered(mature_class=mature_class, min_len=min_len, ch=ch)
                    else:
                        raise ValueError(f"Unknown source={source}")
    
                if isinstance(ds, pd.DataFrame) and not ds.empty:
                    ds.insert(0, 'condition', cond_name)
                    ds.insert(0, 'run', run_folder)
                    all_ds.append(ds)
                    ds_cond.append(ds)
    
            if ds_cond:
                final_cond = pd.concat(ds_cond, ignore_index=True)
                box.add_box(final_cond.D_msd, cond_name)
    
        final_ds = pd.concat(all_ds, ignore_index=True) if all_ds else pd.DataFrame()
        return final_ds, box

    def get_dwell(self, mature_class = 1, min_len=10, frame_rate = 1, max_dist = 250,  
                  ref='ch0', ch_maturation_selection = 'ch1',
                  source="tracked", x0=0, xt=None, y0=0, yt=None):
        """
        Compute and visualize dwell times of co-localized tracks across multiple conditions.
        
        This method extracts dwell times (the time one track spends bound to another)
        from either raw tracked data or filtered datasets, optionally segmented by
        cell maturation class. It aggregates data across all experimental runs and
        produces a histogram of dwell time distributions per condition.
        
        Parameters
        ----------
        mature_class : int, optional
            Maturation class to analyze (only used if ``source='filtered_mature'``). Default is 1.
        min_len : int, optional
            Minimum number of frames a track must persist to be included. Default is 10.
        frame_rate : float, optional
            Frame rate (in seconds per frame) used to convert dwell times to seconds.
            If available, it is automatically determined using `get_time_interval()`.
            Default is 1.
        max_dist : int, optional
            Maximum allowed inter-channel distance (in nanometers) to consider tracks
            as colocalized. Default is 250.
        ref : {'ch0', 'ch1'}, optional
            Reference channel used for dwell time measurement. Default is ``'ch0'``.
        ch_maturation_selection : {'ch0', 'ch1'}, optional
            Channel used to select maturation category when ``source='filtered_mature'``.
            Default is ``'ch1'``.
        source : {'tracked', 'filtered', 'filtered_mature'}, optional
            Source of data to analyze:
            
            - ``'tracked'`` — use directly tracked data  
            - ``'filtered'`` — use filtered tracks outside clusters  
            - ``'filtered_mature'`` — use filtered data limited to selected maturation class  
        
            Default is ``'tracked'``.
        x0, xt : float, optional
            X-axis (time) limits for the dwell time histogram plot. Default is ``0`` and ``None``.
        y0, yt : float, optional
            Y-axis (frequency) limits for the dwell time histogram plot. Default is ``0`` and ``None``.
        
        Returns
        -------     
        - ``final_dwell`` : pandas.DataFrame  
          Combined dwell time data from all analyzed runs and conditions. Columns include:
          ``['condition', 'run', 'colocID', 'track.id_ref', 'track.id_binds', 'cell_id', 'dwell_time']``.
          
        - ``hist`` : HistogramPlotter  
          A histogram object visualizing dwell time distributions per condition.
        """
        all_dwell = []
        try:
            frame_rate = get_time_interval(self.folder)
        except Exception as exc:
            print(
                f"Warning: could not determine the frame rate automatically. {frame_rate} used\n"
                f"Folder: {self.folder}\n"
                f"Error: {type(exc).__name__}: {exc}"
            )
        hist = HistogramPlotter(xlabel="dwell_time(sec)", ylabel="Frequency")

        for cond_path, cond_name in zip(self._conditions_paths, self.conditions_to_use):
            print(f"\nAnalyzing {cond_path}...")
            run_folders = []
            run_pattern = re.compile(r'^Run\d+$')
            for root, dirs, files in os.walk(cond_path):
                matching_dirs = [d for d in dirs if run_pattern.match(d)]
                for d in matching_dirs:
                    full_path = os.path.join(root, d)
                    run_folders.append(full_path)
            dwell_cond = []
            
            
            # pathshdf = glob(cond_path + '/**/**colocsTracks_stats.hdf', recursive=True)
            # paths_locs = list(set(os.path.dirname(file) for file in pathshdf))
            for run_folder in tqdm(run_folders):
                try:
                    if source == "tracked":
                        image = Single_tracked_folder(run_folder).open_files()
                        dwell = image.extract_dwell(frame_rate, min_len, max_dist, ref)                    
                    else:  # use Combined_analysis
                        analysis = Combined_analysis(run_folder, verbose=False)
                        if analysis.tracked is None or analysis.clusters is None:
                            continue
                        
                        if source == "filtered":
                            dwell = analysis.extract_dwell_filtered(mature_class=None, frame_rate = frame_rate,
                                                                 min_len=min_len, max_dist = max_dist, 
                                                                 ref=ref, ch_maturation_selection = ch_maturation_selection)
                        elif source == "filtered_mature":
                            dwell = analysis.extract_dwell_filtered(mature_class=mature_class, frame_rate = frame_rate,
                                                                 min_len=min_len, max_dist = max_dist, 
                                                                 ref=ref, ch_maturation_selection = ch_maturation_selection)
                        else:
                            raise ValueError(f"Unknown source={source}")
                        if isinstance(dwell, pd.DataFrame):
                            dwell.insert(0, 'condition', cond_name)
                            dwell.insert(0, 'run', run_folder)
                            all_dwell.append(dwell)
                            dwell_cond.append(dwell)
                except Exception as exc:
                    self.failed_folders.append(
                        (run_folder, f"{type(exc).__name__}: {exc}")
                    )
                    print(
                        f"Warning: dwell-time extraction failed.\n"
                        f"Folder: {run_folder}\n"
                        f"Error: {type(exc).__name__}: {exc}"
                    )
                    continue

            if dwell_cond:
                final_cond = pd.concat(dwell_cond, ignore_index=True)
                hist.add_data(final_cond.dwell_time, label=cond_name)

        final_dwell = pd.concat(all_dwell, ignore_index=True) if all_dwell else pd.DataFrame()
        hist.set_labels()
        hist.set_xlim(x0, xt)
        hist.set_ylim(y0, yt)
        hist.show_plot()

        return final_dwell, hist
    def count_maturation(self, ch = '488nm'):
        """
        Collect and summarize maturation classification results across all experimental conditions.
    
        This method loads precomputed maturation analysis results from each run folder
        (stored as JSON files) and aggregates them into a single DataFrame.  
        Each entry corresponds to a cell and its assigned maturation category.
    
        Parameters
        ----------
        ch : str, optional
            The fluorescence channel wavelength (e.g., ``'488nm'`` or ``'561nm'``) used
            for maturation analysis.  
            The method looks for JSON files named ``maturation__{ch}.json`` inside each
            run folder’s ``maturation_analysis`` subdirectory.  
            Default is ``'488nm'``.
    
        Returns
        -------
        pandas.DataFrame
            A combined table of maturation data from all runs and conditions.  
            Columns include:
            
            - ``run`` : Name or identifier of the experimental run  
            - ``cell`` : Cell identifier  
            - ``category`` : Maturation category (integer or label, depending on analysis)  
            - ``condition`` : Experimental condition name
        """
        result = []
        for cond_path, cond_name in zip(self._conditions_paths, self.conditions_to_use):
            print(f"\nAnalyzing {cond_path}...")
            run_folders = []
            run_pattern = re.compile(r'^Run\d+$')
            for root, dirs, files in os.walk(cond_path):
                matching_dirs = [d for d in dirs if run_pattern.match(d)]
                for d in matching_dirs:
                    full_path = os.path.join(root, d)
                    run_folders.append(full_path)
            dwell_cond = []
            
            
            # pathshdf = glob(cond_path + '/**/**colocsTracks_stats.hdf', recursive=True)
            # paths_locs = list(set(os.path.dirname(file) for file in pathshdf))
            for run_folder in tqdm(run_folders):
                maturation_json = os.path.join(run_folder, 'maturation_analysis',f"maturation__{ch}.json")
                if os.path.exists(maturation_json):
                    with open(maturation_json, "r") as f:
                        maturation = pd.DataFrame(json.load(f))
                maturation['condition'] = cond_name
                result.append(maturation[['run', 'cell', 'category', 'condition']])
        
        final = pd.concat(result, ignore_index=True) if result else pd.DataFrame()
        return final
                
    def print_failed_folders(self):
        if self.failed_folders:
            print("The following folders failed during processing:")
            for folder, error in self.failed_folders:
                print(f"- {folder}: {error}")
        else:
            print("All folders processed successfully.")
           
    def _count_run_folders_recursive(self, root_folder):
        pattern = re.compile(r'^Run\d+$')
        count = 0

        for dirpath, dirnames, _ in os.walk(root_folder):
            for dirname in dirnames:
                if pattern.match(dirname):
                    count += 1
        return count
    def validate(self):
        print("Just a reminder that most of the time the whole dataset should be analyzed using the same parameters.")
        print("Here are the parameters for the first folder in the dataset that has colocalized tracks:")
        yaml = openyaml(glob(self.folder + r'/**/*_colocsTracks.yaml', recursive=True))
        for i, j in enumerate(yaml.items()):
            print(f"{j[0]}: {j[1]}")


class Dataset_tracked_folder:
    def __init__(self, folder):
        self.folder = folder
        self.conditions = self._get_conditions()
        self._conditions_paths = [os.path.join(self.folder, subfolder) for subfolder in self.conditions]
        self.conditions_to_use = self.conditions
        self.tracked_folders = dict.fromkeys(self.conditions_to_use, 'Not analyzed, use count_tracked()')
        self.result_count = None
    def _get_conditions(self):
        subfolders = [f.name for f in os.scandir(self.folder) if f.is_dir()]
        return subfolders
    def select_conditions(self, indices):
        self.conditions_to_use = [self.conditions[i] for i in indices]
        self._conditions_paths = [os.path.join(self.folder, subfolder) for subfolder in self.conditions_to_use]
    def count_cotracked(self):
        for i in range(len(self._conditions_paths)):
            pathshdf = glob(self._conditions_paths[i] + '/**/**colocsTracks_stats.hdf', recursive=True)
            pathshdf = [os.path.dirname(path) for path in pathshdf]
            num_runs = self._count_run_folders_recursive(self._conditions_paths[i])
            print(f'from {self._conditions_paths[i]}, {len(pathshdf)} runs have colocalized tracks out of {num_runs} runs')
            self.tracked_folders[self.conditions_to_use[i]] = pathshdf
    def count_number_cotracks(self):
        results = []  # This will store the data for each run folder
        run_pattern = re.compile(r'^Run\d+$')  # Regex to identify folders like 'Run0001'
        path_yaml = glob(self.folder + r'/**/*_colocsTracks.yaml', recursive=True)
        general_yaml_file = openyaml(path_yaml)
        ch0_hint = general_yaml_file['ch0']
        ch1_hint = general_yaml_file['ch1']
        min_len_track = general_yaml_file["min_len_track"]
        # Loop through each selected condition and its corresponding path
        for cond_path, cond_name in zip(self._conditions_paths, self.conditions_to_use):
            # Recursively walk through the directory tree under each condition
            for dirpath, dirnames, _ in os.walk(cond_path):
                for dirname in dirnames:
                    try:
                        if run_pattern.match(dirname):  # Check if the folder matches 'RunXXXX'
                            run_folder = os.path.join(dirpath, dirname)
                            # Try to find a YAML file in the run folder (used to get min_len_track)
                            yaml_files = glob(run_folder + '/**/*_colocsTracks.yaml', recursive=True)
                            if yaml_files:
                                yaml_data = openyaml(yaml_files)
                                min_len_track = yaml_data.get("min_len_track", 5)  # Default to 5 if key missing

                            # Load tracking data using 
                            a = Single_tracked_folder(run_folder, ch0_hint, ch1_hint).open_files()
                            print(run_folder)
                            # Count tracks in channel 0 that meet the length threshold
                            if isinstance(a.stats0, pd.DataFrame):
                                ch0_tracks = a.stats0[a.stats0.loc_count >= min_len_track].shape[0]
                            else: 
                                ch0_tracks = 0
                            if isinstance(a.stats1, pd.DataFrame):
                                ch1_tracks = a.stats1[a.stats1.loc_count >= min_len_track].shape[0]
                            else: 
                                ch1_tracks = 0
                            try:
                                # Count colocalized tracks if the attribute exists
                                coloc_count = a.coloc_stats.shape[0] if hasattr(a, 'coloc_stats') else 0
                                
                            except Exception as e:
                                # If loading fails, assume 0 tracks
                                coloc_count = 0

                            # Append the results for this run folder
                            results.append({
                                "folder": run_folder,
                                "condition": cond_name,
                                "colocalized_tracks": coloc_count,
                                "ch0_tracks": ch0_tracks,
                                "ch1_tracks": ch1_tracks,
                                "min_len_track": min_len_track
                            })
                    except Exception as e:
                        print(e)
                        continue

        # Convert the list of dictionaries into a pandas DataFrame
        self.result_count = pd.DataFrame(results)
        return self.result_count
    def summary_count_number_cotracks(self):
        if isinstance(self.result_count, pd.DataFrame):
            aggregated_df = self.result_count.groupby('condition').agg(
                total_colocalized_tracks=pd.NamedAgg(column='colocalized_tracks', aggfunc='sum'),
                total_ch0_tracks=pd.NamedAgg(column='ch0_tracks', aggfunc='sum'),
                total_ch1_tracks=pd.NamedAgg(column='ch1_tracks', aggfunc='sum')
            ).reset_index()
            return aggregated_df
        else: 
            print("run count_number_cotracks first")
    def get_Ds(self, min_len = 10, channel = 'ch0'):
        all_ds = []  # List to collect all DataFrames
        box = BoxPlotter(xlabel="Conditions", ylabel="Diff. Coeff. (um^2/sec)")
        for i, cond in tqdm(zip(self._conditions_paths, self.conditions_to_use), desc='Extracting Ds...\n'):
            print(f'\nAnalyzing {i}...')
            ds_cond = []
            pathshdf = glob(i + '/**/**.hdf', recursive=True)
            paths_locs = list(set(os.path.dirname(file) for file in pathshdf))
            for j in paths_locs:
                image = Single_tracked_folder(j).open_files()
                ds = image.extract_Ds(min_len, channel)
                if isinstance(ds, pd.DataFrame):
                # Add columns at the beginning
                    ds.insert(0, 'condition', cond)
                    ds.insert(0, 'run', j)
                    all_ds.append(ds)  # Accumulate
                    ds_cond.append(ds)
            final_cond = pd.concat(ds_cond, ignore_index=True)
            box.add_box(final_cond.D_msd, cond)
        # Concatenate all DataFrames into one
        final_ds = pd.concat(all_ds, ignore_index=True)
        # box.add_statistical_annotations()
        return final_ds, box
    def get_dwell(self, min_len=10, ref='ch0', x0=0, xt=None, y0=0, yt=None):
        all_dwell = []
        frame_rate = get_time_interval(self.folder)
        hist = HistogramPlotter(xlabel="dwell_time(sec)", ylabel="Frequency")
        for i, cond in tqdm(zip(self._conditions_paths, self.conditions_to_use), desc='Extracting Ds...\n'):
            print(f'\nAnalyzing {i}...')
            dwell_cond = []
            pathshdf = glob(i + '/**/**colocsTracks_stats.hdf', recursive=True)
            paths_locs = list(set(os.path.dirname(file) for file in pathshdf))
            for j in tqdm(paths_locs):
                pathsyaml = glob(j + '/**/**colocsTracks.yaml', recursive=True)
                yaml = openyaml(pathsyaml)
                image = Single_tracked_folder(j).open_files()
                dwell = image.extract_dwell(frame_rate=frame_rate, min_len=min_len, max_dist=yaml['th'], ref=ref)
                if isinstance(dwell, pd.DataFrame):
                    dwell.insert(0, 'condition', cond)
                    dwell.insert(0, 'run', j)
                    all_dwell.append(dwell)
                    dwell_cond.append(dwell)
            dwell_cond = [df for df in dwell_cond if df is not None and not df.empty and not df.isna().all().all()]
            if dwell_cond:
                final_cond = pd.concat(dwell_cond, ignore_index=True)
                hist.add_data(final_cond.dwell_time, label=f"{cond}")
        all_dwell = [df for df in all_dwell if df is not None and not df.empty and not df.isna().all().all()]
        if all_dwell:
            final_dwell = pd.concat(all_dwell, ignore_index=True)
        else:
            final_dwell = pd.DataFrame(columns=['run', 'condition', 'colocID', 'track.id_ref', 'track.id_binds', 'cell_id', 'dwell_time'])
        hist.set_labels()
        hist.set_xlim(x0, xt)
        hist.set_ylim(y0, yt)
        hist.show_plot()
        return final_dwell, hist
    def validate(self):
        print("Just a reminder that most of the time the whole dataset should be analyzed using the same parameters.")
        print("Here are the parameters for the first folder in the dataset that has colocalized tracks:")
        yaml = openyaml(glob(self.folder + r'/**/*_colocsTracks.yaml', recursive=True))
        for i, j in enumerate(yaml.items()):
            print(f"{j[0]}: {j[1]}")
           
    def _count_run_folders_recursive(self, root_folder):
        pattern = re.compile(r'^Run\d+$')
        count = 0

        for dirpath, dirnames, _ in os.walk(root_folder):
            for dirname in dirnames:
                if pattern.match(dirname):
                    count += 1
        return count