import os
import numpy as np
import pandas as pd
from .io_utils import get_nm2px, openyaml, validate_choice
from glob import glob
from picasso.io import TiffMultiMap, load_movie

from .plotting import LinePlotter, TrackPlotter


class Tracked_image:
    """
   Analysis tool for single-particle tracked images.

   This class encapsulates single- or dual-channel imaging data along with
   tracking results, colocalization data, and derived statistics. It provides
   methods to visualize tracks, colocalizations, intensity over time, and
   to extract diffusion coefficients or dwell times for further analysis.

   Attributes
   ----------
   ch0 : TiffMultiMap
       Image stack for channel 0.
   tracks0 : pandas.DataFrame
       Tracks data for channel 0.
   stats0 : pandas.DataFrame
       Track statistics for channel 0.
   ch1 : TiffMultiMap, optional
       Image stack for channel 1 (if dual-channel data).
   tracks1 : pandas.DataFrame, optional
       Tracks data for channel 1.
   stats1 : pandas.DataFrame, optional
       Track statistics for channel 1.
   coloc_tracks : pandas.DataFrame, optional
       Colocalized tracks across channels.
   coloc_stats : pandas.DataFrame, optional
       Statistics for colocalized tracks.
   nm2px : float
       Conversion factor from nanometers to pixels.
   folder : str, optional
       Path to the folder containing image and tracking data.
   result_cluster_analysis : dict
       Placeholder for clustering results per channel.
   summary_cluster_analysis : dict
       Summary of clustering results per channel.
   linked_clusters : dict
       Linked clusters per channel.
   linked_clusters_stats : dict
       Statistics of linked clusters per channel.
   tracks_outside_clusters : dict
       Tracks not associated with any clusters.

   Methods
   -------
   plot_tracks(to_check, channel='ch0')
       Plot selected tracks for the specified channel.
   plot_colocs(to_check)
       Plot colocalized tracks across channels.
   intensity_coloc(to_check, legend_loc='best', legend_0='ch0', legend_1='ch1')
       Plot intensity of colocalizing tracks over time.
   extract_Ds(min_len=10, channel='ch0')
       Extract diffusion coefficients (D_msd) for tracks above a minimum length.
   extract_dwell(frame_rate=1, min_len=10, max_dist=250, ref='ch0')
       Compute dwell times for colocalized tracks based on distance thresholds.
   """
    def __init__(self, ch0, tracks0, stats0, ch1=None, tracks1=None, stats1=None, coloc_tracks=None, coloc_stats=None, nm2px=108, folder = None):
        #checking data types
        assert isinstance(ch0, TiffMultiMap), "'ch0' must be a picasso.io.TiffMultiMap object"
        assert isinstance(tracks0, pd.DataFrame), "'tracks0' must be a pd.DataFrame object"
        assert isinstance(stats0, pd.DataFrame), "'stats0' must be a pd.DataFrame object"
        assert isinstance(nm2px, (int, float))
        # Optional checks
        if ch1 is not None:
            assert isinstance(ch1, TiffMultiMap), "'ch1' must be a picasso.io.TiffMultiMap object"
        if tracks1 is not None:
            assert isinstance(tracks1, pd.DataFrame), "'tracks1' must be a pandas DataFrame"
        if stats1 is not None:
            assert isinstance(stats1, pd.DataFrame), "'stats1' must be a pandas DataFrame"
        if coloc_tracks is not None:
            assert isinstance(coloc_tracks, pd.DataFrame), "'coloc_tracks' must be a pandas DataFrame"
        if coloc_stats is not None:
            assert isinstance(coloc_stats, pd.DataFrame), "'coloc_stats' must be a pandas DataFrame"
        
        self.ch0 = ch0
        self.ch1 = ch1
        self.tracks0 = tracks0
        self.stats0 = stats0
        self.tracks1 = tracks1
        self.stats1 = stats1
        self.coloc_tracks = coloc_tracks
        self.coloc_stats = coloc_stats
        self.nm2px = nm2px
        self.folder = folder
        self.result_cluster_analysis = {'ch0' : None, 'ch1' : None}
        self.summary_cluster_analysis = {'ch0' : None, 'ch1' : None}
        self.linked_clusters = {'ch0' : None, 'ch1' : None}
        self.linked_clusters_stats = {'ch0' : None, 'ch1' : None}
        self.tracks_outside_clusters = {'ch0' : None, 'ch1' : None}
    def plot_tracks(self, to_check, channel='ch0'):
        """
        Plot selected tracks on top of the maximum-intensity projection.

        Parameters
        ----------
        to_check : list or array-like
            List of track IDs to display.
        channel : str, optional
            Channel to plot ('ch0' or 'ch1'). Default is 'ch0'.

        Returns
        -------
        TrackPlotter
            A TrackPlotter object with the plotted tracks.
        """
        validate_choice(channel, {'ch0', 'ch1'}, 'channel')
        if channel == 'ch0':
            image = self.ch0
            tracks = self.tracks0
            stats = self.stats0
        elif channel == 'ch1':
            assert self.ch1 is not None, "ch1 is not provided."
            assert self.tracks1 is not None, "tracks1 is not provided."
            assert self.stats1 is not None, "stats1 is not provided."
            image = self.ch1
            tracks = self.tracks1
            stats = self.stats1
        else:
            raise ValueError("channel must be 'ch0' or 'ch1'")
    
        plotter = TrackPlotter(image, self.nm2px)
        print('Tracks inside ROIs:', stats[stats['track.id'].isin(to_check)]['cell_id'].unique())
        
        contour = None
        if 'contour' in stats.columns and len(stats[stats['track.id'].isin(to_check)]['cell_id'].unique()) == 1:
            contour = stats[stats['track.id'] == to_check[0]]['contour'].values[0]
        
        offset = plotter.show_max_projection(contour)
        for i in to_check:
            if channel == 'ch0':
                plotter.plot_tracks(tracks, i, offset=offset, color='red')
            elif channel == 'ch1':
                plotter.plot_tracks(tracks, i, offset=offset, color='blue')
        plotter.show_plot()
        return plotter   
    def plot_colocs(self, to_check):
        """
        Plot colocalized tracks for given colocalization IDs.

        Parameters
        ----------
        to_check : list or array-like
            List of colocalization IDs to display.

        Returns
        -------
        TrackPlotter
            A TrackPlotter object with the plotted colocalized tracks.
        """
        assert self.coloc_tracks is not None, "coloc_tracks is not provided."
        assert self.coloc_stats is not None, "coloc_stats is not provided."
    
        plotter = TrackPlotter(self.ch0, self.nm2px) #initialize track plotter
        print('Tracks inside ROIs:', self.coloc_stats[self.coloc_stats['colocID'].isin(to_check)]['cell_id'].unique())
        
        #get cell contour, if filtere by ROI
        contour = None 
        if 'contour' in self.coloc_stats.columns and len(self.coloc_stats[self.coloc_stats['colocID'].isin(to_check)]['cell_id'].unique()) == 1:
            contour = self.coloc_stats[self.coloc_stats['colocID'] == to_check[0]]['contour'].values[0]
        #print max projection of the image (or cropped cell) in the plotter
        offset = plotter.show_max_projection(contour)
        #add tracks:
        # - Red: ch0 (reference channel to find colocalizations)
        # - blue: ch1 
        # - purple: colocalized track
        for i in to_check:
            plotter.plot_tracks(self.coloc_tracks, i,id_col = 'colocID', x_col='x_0', y_col='y_0', color='red', offset=offset)
            plotter.plot_tracks(self.coloc_tracks, i,id_col = 'colocID', x_col='x_1', y_col='y_1', color='blue', offset=offset)
            plotter.plot_tracks(self.coloc_tracks, i,id_col = 'colocID', x_col='x', y_col='y', color='purple', offset=offset)
        plotter.show_plot()
        return plotter
    def plot_intensity_coloc(self, to_check, legend_loc = 'best', legend_0 = 'ch0', legend_1 = 'ch1'):
        """
        Plot the intensity of colocalizing tracks over time.

        Vertical dashed lines indicate the period of colocalization.

        Parameters
        ----------
        to_check : int
            ColocID of the track to plot.
        legend_loc : str, optional
            Location of the legend. Default is 'best'.
        legend_0 : str, optional
            Legend label for channel 0. Default is 'ch0'.
        legend_1 : str, optional
            Legend label for channel 1. Default is 'ch1'.

        Returns
        -------
        LinePlotter
            A LinePlotter object with the intensity curves plotted.
        """
        assert self.coloc_tracks is not None, "coloc_tracks is not provided."
        assert self.ch1 is not None, "ch1 is not provided."

        plotter = LinePlotter(title="Intensity Over Time", xlabel="Time (sec)", ylabel="Spot intensity/median BG intensity (AU)")

        track = self.coloc_tracks[self.coloc_tracks.colocID == to_check].copy()
        track['im_int_0'] = None
        track['im_int_1'] = None

        for index, row in track.iterrows():
            if not np.isnan(row.y_0):
                track.at[index, 'im_int_0'] = np.mean(self.ch0[row.t, int(row.y_0/self.nm2px)-1:int(row.y_0/self.nm2px)+1, int(row.x_0/self.nm2px)-1:int(row.x_0/self.nm2px)+1]) / np.median(self.ch0[row.t])
            if not np.isnan(row.y_1):
                track.at[index, 'im_int_1'] = np.mean(self.ch1[row.t, int(row.y_1/self.nm2px)-1:int(row.y_1/self.nm2px)+1, int(row.x_1/self.nm2px)-1:int(row.x_1/self.nm2px)+1]) / np.median(self.ch1[row.t])

        plotter.add_data(track.t * 2, track.im_int_0.rolling(window=1).mean(), label=legend_0, color='red')
        plotter.add_data(track.t * 2, track.im_int_1.rolling(window=1).mean(), label=legend_1, color='blue')

        colocalized_df = track[(track['distance'] <= 300) & pd.notna(track['x']) & pd.notna(track['y'])]
        if not colocalized_df.empty:
            start_time = colocalized_df['t'].min() * 2
            end_time = colocalized_df['t'].max() * 2
            plotter.ax.axvline(x=start_time, color='black', linestyle='--', label='colocalization')
            plotter.ax.axvline(x=end_time, color='black', linestyle='--')
        plotter.set_labels()
        plotter.set_grid()
        plotter.set_ylim(0)
        plotter.show_plot(legend_loc= legend_loc)
        return plotter   
    def extract_Ds(self, min_len=10, channel='ch0'):
        """
        Extract diffusion coefficients (D_msd) for tracks longer than a minimum length.

        Parameters
        ----------
        min_len : int, optional
            Minimum track length to include. Default is 10.
        channel : str, optional
            Channel to extract from ('ch0' or 'ch1'). Default is 'ch0'.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ['track.id', 'cell_id', 'D_msd'] for qualifying tracks.
        """
        validate_choice(channel, {'ch0', 'ch1'}, 'channel')

        if channel == 'ch0':
            stats = self.stats0
        else:
            stats = self.stats1
    
        columns_to_extract = ['track.id', 'cell_id', 'D_msd']
        ds = stats[stats['length'] >= min_len][columns_to_extract]
        ds = ds.dropna(subset=['D_msd'])
        return ds
    def extract_dwell(self, frame_rate = 1, min_len=10, max_dist = 250, ref='ch0'):
        """
       Compute dwell times for colocalized tracks.

       Parameters
       ----------
       frame_rate : float, optional
           Frame interval in seconds. Default is 1.
       min_len : int, optional
           Minimum number of frames for a track to be considered. Default is 10.
       max_dist : float, optional
           Maximum distance (nm) to count as colocalized. Default is 250.
       ref : str, optional
           Reference channel to compute dwell time ('ch0' or 'ch1'). Default is 'ch0'.

       Returns
       -------
       pandas.DataFrame or None
           DataFrame containing columns ['colocID', 'track.id_ref', 'track.id_binds', 'cell_id', 'dwell_time'] 
           for qualifying colocalizations, or None if no valid data is available.
       """
        validate_choice(ref, {'ch0', 'ch1'}, 'ref')
        stats = self.coloc_stats
        tracks = self.coloc_tracks
        if all(isinstance(obj, pd.DataFrame) for obj in [stats, tracks]):
            unique_colocIDs = stats[(stats['num_frames_coloc'] >min_len)]['colocID'].unique()
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

class Single_tracked_folder:
    """
    Class to handle tracked images and associated data in a single folder.

    This class provides functionality to:
    - Load tracked movies, track CSVs, and statistics HDF files.
    - Handle single-channel or dual-channel tracked datasets.
    - Validate the presence of tracked images and colocalized tracks.
    - Retrieve pixel-to-nanometer conversion based on the microscope.

    Attributes
    ----------
    folder : str
        Path to the folder containing tracked movies and associated files.
    ch0_hint : str or None
        Optional hint for the channel 0 laser wavelength (used if no YAML file is present).
    ch1_hint : str or None
        Optional hint for the channel 1 laser wavelength (used if no YAML file is present).
    """
    def __init__(self, folder, ch0_hint = None, ch1_hint = None):
        self.folder = folder
        self.ch0_hint = ch0_hint
        self.ch1_hint = ch1_hint
    def open_files(self):
        """
            Open tracked image files and associated data from the folder.
        
            This method supports three cases:
            1. A coloc YAML file is present, specifying ch0 and ch1 lasers.
            2. Two CSV files are found without coloc YAML (requires `ch0_hint` and `ch1_hint`).
            3. A single CSV file is found without coloc YAML.
        
            Returns
            -------
            Tracked_image
                An instance of Tracked_image containing the loaded movie(s), track data, and statistics.
    """
        yaml_files = glob(self.folder + '/**/**_colocsTracks.yaml', recursive=True)
        yaml_file = openyaml(yaml_files) if yaml_files else False
        if yaml_file:
            ch0_laser = yaml_file['ch0']
            ch1_laser = yaml_file['ch1']
            # print(f'Data has been tracked with the colocalizing tracks algorithm and the reference channel was {ch0_laser}')
            ch0 = load_movie(glob(self.folder + f'/**/**{ch0_laser}nm.tif', recursive=True)[0])[0]
            tracks0 = pd.read_csv(glob(self.folder + f'/**/**{ch0_laser}**_locs_nm_trackpy.csv', recursive=True)[0])
            stats0 =  pd.read_hdf(glob(self.folder + f'/**/**{ch0_laser}**_locs_nm_trackpy_stats.hdf', recursive=True)[0])
            ch1 = load_movie(glob(self.folder + f'/**/**{ch1_laser}nm.tif', recursive=True)[0])[0]
            tracks1 = pd.read_csv(glob(self.folder + f'/**/**{ch1_laser}**_locs_nm_trackpy.csv', recursive=True)[0])
            stats1 = pd.read_hdf(glob(self.folder + f'/**/**{ch1_laser}**_locs_nm_trackpy_stats.hdf', recursive=True)[0])
            coloc_tracks = pd.read_csv(glob(self.folder + '/**/**colocsTracks.csv', recursive=True)[0])
            coloc_stats = pd.read_hdf(glob(self.folder + '/**/**colocsTracks_stats.hdf', recursive=True)[0])
            image1 = Tracked_image(
                                   ch0,
                                   tracks0, 
                                   stats0,
                                   ch1, 
                                   tracks1, 
                                   stats1,
                                   coloc_tracks,
                                   coloc_stats, 
                                   get_nm2px(self.folder),   
                                   self.folder
                                   )
            return image1
        elif not yaml_file and len(glob(self.folder + '/**/**_locs_nm_trackpy.csv', recursive=True)) == 2:
            expected_columns = [
                "track.id", "cell_id", "x_std", "y_std", "length", "loc_count",
                "msd_fit_a", "msd_fit_b", "jumps", "lagtimes", "msd",
                "D_jd0", "A_jd0", "D_jd1", "A_jd1", "D_msd",
                "path", "contour", "area", "centroid"
            ]
            ch0_laser = self.ch0_hint
            ch1_laser = self.ch1_hint
            # print(f'Data has been tracked with the colocalizing tracks algorithm and the reference channel was {ch0_laser}')
            ch0 = load_movie(glob(self.folder + f'/**/**{ch0_laser}nm.tif', recursive=True)[0])[0]
            tracks0 = pd.read_csv(glob(self.folder + f'/**/**{ch0_laser}**_locs_nm_trackpy.csv', recursive=True)[0])
            stats0 = next((pd.read_hdf(f) for f in glob(self.folder + f'/**/**{ch0_laser}**_locs_nm_trackpy_stats.hdf', recursive=True) if os.path.isfile(f)), pd.DataFrame(columns = expected_columns))
            ch1 = load_movie(glob(self.folder + f'/**/**{ch1_laser}nm.tif', recursive=True)[0])[0]
            tracks1 = pd.read_csv(glob(self.folder + f'/**/**{ch1_laser}**_locs_nm_trackpy.csv', recursive=True)[0])
            stats1 = next((pd.read_hdf(f) for f in glob(self.folder + f'/**/**{ch1_laser}**_locs_nm_trackpy_stats.hdf', recursive=True) if os.path.isfile(f)), pd.DataFrame(columns = expected_columns))
            image1 = Tracked_image(
                                   ch0,
                                   tracks0, 
                                   stats0,
                                   ch1, 
                                   tracks1, 
                                   stats1, 
                                   nm2px = get_nm2px(self.folder), 
                                   folder = self.folder
                                   )
            return image1
        elif not yaml_file and len(glob(self.folder + '/**/**_locs_nm_trackpy.csv', recursive=True)) == 1:
            ch0 = load_movie(glob(self.folder + '/**/**nm.tif', recursive=True)[0])[0]
            tracks0 = pd.read_csv(glob(self.folder + '/**/**locs_nm_trackpy.csv', recursive=True)[0])
            stats0 =  pd.read_hdf(glob(self.folder + '/**/**locs_nm_trackpy_stats.hdf', recursive=True)[0])
            image1 = Tracked_image(
                                   ch0,
                                   tracks0, 
                                   stats0,
                                   nm2px = get_nm2px(self.folder), 
                                   folder = self.folder
                                   )
            return image1
        else: 
            print('No tracking data available')          
    def validate(self):
        """
            Validate the contents of the folder by listing images and tracking files.
        
            Prints information about:
            - Available TIFF image files.
            - Tracking CSV files.
            - Colocalized track files, if any.
        
            Notes
            -----
            If colocalized tracks exist, it attempts to read the YAML file to report
            the reference channel used.
    """
        tifs = glob(self.folder + '/**/**nm.tif', recursive=True)
        csvs = glob(self.folder + '/**/**_locs_nm_trackpy.csv', recursive=True)
        coloc_tracks = glob(self.folder + '/**/**_colocsTracks.csv', recursive=True)
        print('The images present are:')
        for i in tifs:
            print(i)
        print('\nThe tracked images are:')
        for i in csvs:
            print(i)
        if coloc_tracks:
            yaml_files = glob(self.folder + '/**/**_colocsTracks.yaml', recursive=True)
            yaml_file = openyaml(yaml_files) if yaml_files else False
            ch0_laser = yaml_file['ch0']
            print(f'\nThe tarcks have been colocalized with {ch0_laser}nm as reference channel')

        else:
            print('The tracks have not been colocalized')   