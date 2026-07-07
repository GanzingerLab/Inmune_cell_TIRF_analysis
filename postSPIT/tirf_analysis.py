import cv2
import joblib
import json
import logging
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import tifffile
import yaml
from collections import defaultdict
from glob import glob
from natsort import natsorted
from picasso.io import TiffMultiMap, load_movie
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.stats import linregress, ttest_ind
from sklearn.metrics import r2_score
from skimage.draw import polygon
from skimage.feature import local_binary_pattern
from skimage.filters import (
    threshold_li,
    threshold_otsu,
)
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_holes,
    remove_small_objects
)
from spit import tools
from spit import colocalize as coloc
from spit import linking as link
from tqdm import tqdm
import trackpy as tp
import pandas.errors
from matplotlib.patches import Patch
import warnings
from pandas.errors import PerformanceWarning
logging.getLogger('trackpy').setLevel(logging.ERROR)


class Plotter:
    """
    A convenience wrapper around Matplotlib for consistent and streamlined plotting.

    The `Plotter` class simplifies the process of creating, labeling, displaying,
    and saving plots with consistent formatting. It provides quick methods for
    setting axis limits, enabling grids, refreshing figures, and handling legends.

    Attributes
    ----------
    title : str
        Title of the plot.
    xlabel : str
        Label for the X-axis.
    ylabel : str
        Label for the Y-axis.
    figsize : touple
        Image size in inches. 
    fig : matplotlib.figure.Figure
        The main Matplotlib figure object.
    ax : matplotlib.axes.Axes
        The main Matplotlib axes object.

    Methods
    -------
    set_labels()
        Apply the plot title and axis labels.
    set_ylim(s, e=None)
        Set the vertical axis (Y) limits.
    set_xlim(s, e=None)
        Set the horizontal axis (X) limits.
    set_grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5)
        Enable or configure gridlines on the plot.
    show_plot(legend_loc='best')
        Display the plot and add a legend if available.
    save_plot(filename, dpi=300)
        Save the current plot to a file.
    refresh(legend_loc='best')
        Refresh the figure for live updates (useful in interactive mode).
    """
    def __init__(self, title="Plot", xlabel="X-axis", ylabel="Y-axis", figsize = (6.4, 4.8)):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fig, self.ax = plt.subplots(figsize=figsize)
    def set_labels(self):
        """
        Apply the title and axis labels to the current plot.

        This method sets the `title`, `xlabel`, and `ylabel` properties of the
        Matplotlib Axes object associated with this Plotter.
        """
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)     
    def set_ylim(self, s, e = None):
        """
        Set the limits of the Y-axis.

        Parameters
        ----------
        s : float
            Lower Y-axis limit.
        e : float, optional
            Upper Y-axis limit. If omitted, Matplotlib will auto-scale.
        """
        self.ax.set_ylim(s, e)
    def set_xlim(self, s, e = None):
        """
        Set the limits of the X-axis.

        Parameters
        ----------
        s : float
            Lower X-axis limit.
        e : float, optional
            Upper X-axis limit. If omitted, Matplotlib will auto-scale.
        """
        self.ax.set_xlim(s, e)
    def show_plot(self, legend_loc='best'):
        """
        Display the current plot, adding a legend if one exists.

        Parameters
        ----------
        legend_loc : str, optional
            Location of the legend. Default is 'best'.

            Valid options include:
                'best', 'upper right', 'upper left', 'lower left', 'lower right',
                'right', 'center left', 'center right', 'lower center', 'upper center'

        Returns
        -------
        fig : matplotlib.figure.Figure
            The displayed Matplotlib Figure object.
        """
        
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc=legend_loc)
        plt.show()
        return self.fig
    def set_grid(self, visible=True, which='both', axis='both', linestyle='--', linewidth=0.5):
        """
       Enable or configure gridlines on the plot.

       Parameters
       ----------
       visible : bool, optional
           Whether to display gridlines. Default is True.
       which : {'major', 'minor', 'both'}, optional
           Which gridlines to show. Default is 'both'.
       axis : {'x', 'y', 'both'}, optional
           Which axes to apply gridlines to. Default is 'both'.
       linestyle : str, optional
           Line style for gridlines. Default is dashed ('--').
       linewidth : float, optional
           Line width for gridlines. Default is 0.5.
       """
        self.ax.grid(visible, which=which, axis=axis, linestyle=linestyle, linewidth=linewidth)
    def save_plot(self, filename, dpi = 300):
        """
       Save the current plot to a file.

       Parameters
       ----------
       filename : str
           File path (including extension, e.g. '.png' or '.pdf') to save the plot.
       dpi : int, optional
           Resolution of the saved figure in dots per inch. Default is 300.
       """
        self.fig.savefig(filename,dpi = dpi, bbox_inches='tight', pad_inches=0.1)
    def refresh(self, legend_loc='best'):
        """
        Refresh the plot for live updates during interactive plotting.

        This method re-renders the plot in real-time, useful when updating plots.

        Parameters
        ----------
        legend_loc : str, optional
            Location of the legend. Default is 'best'.
        """
        plt.ion()  # Ensure interactive mode is on
        plt.figure(self.fig.number)  # Reactivate the figure
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc=legend_loc)
        self.fig.canvas.draw()  # Redraw the canvas
        self.fig.canvas.flush_events()  # Ensure updates are shown

class LinePlotter(Plotter):
    """
    A simple line plotting class extending the base Plotter.

    The `LinePlotter` class provides an easy interface to plot one or more
    continuous data series with automatic labeling and color management.

    Inherits from
    --------------
    Plotter

    Methods
    -------
    add_data(x, y, label="Line", color="blue")
        Add a new line to the plot with the given data, label, and color.
    """
    def add_data(self, x, y, label="Line", color="blue"):
        """
        Add a line to the plot.

        Parameters
        ----------
        x : array-like
            Sequence of X-axis values.
        y : array-like
            Sequence of Y-axis values corresponding to `x`.
        label : str, optional
            Label for the plotted line (used in legend). Default is "Line".
        color : str, optional
            Color of the line. Default is "blue".
        """
        self.ax.plot(x, y, label=label, color=color)

class ScatterPlotter(Plotter):
    """
    A scatter plotting class extending the base Plotter.

    The `ScatterPlotter` class is used for plotting discrete data points
    as scatter plots, with optional labeling and coloring.

    Inherits from
    --------------
    Plotter

    Methods
    -------
    add_data(x, y, label="Scatter", color="green")
        Add a scatter data series to the plot.
    """
    def add_data(self, x, y, label="Scatter", color="green"):
        """
        Add a scatter dataset to the plot.

        Parameters
        ----------
        x : array-like
            X-axis values of points.
        y : array-like
            Y-axis values of points.
        label : str, optional
            Label for the scatter plot in the legend. Default is "Scatter".
        color : str, optional
            Marker color. Default is "green".
        """
        self.ax.scatter(x, y, label=label, color=color)

class HueBoxPlotter:
    """
    Create grouped boxplots with hue categories and jittered data points.

    The `HueBoxPlotter` supports visualization of data grouped by two categorical
    variables: a "group" and a "hue" (subgroup/condition). Each group can have
    multiple hues displayed side by side for comparison.

    Attributes
    ----------
    title : str
        Title of the box plot.
    xlabel : str
        Label for the X-axis.
    ylabel : str
        Label for the Y-axis.
    palette : dict
        Mapping from hue names to colors. If None, defaults to Matplotlib colors.
    data : dict
        Nested dictionary structure {group: {hue: data}}.
    hues : set
        Set of hue names added to the plot.
    groups : list
        Ordered list of group names.
    fig : matplotlib.figure.Figure
        Figure object for the plot.
    ax : matplotlib.axes.Axes
        Axes object for the plot.

    Methods
    -------
    add_box(group, hue, data)
        Add a dataset under a specific group and hue.
    plot()
        Render the grouped boxplot with hue-based coloring and jittered dots.
    """
    def __init__(self, title="Box Plot", xlabel="X-axis", ylabel="Y-axis", palette=None):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.palette = palette if palette else {}
        self.data = {}  # Nested dictionary: {group: {hue: data}}
        self.hues = set()
        self.groups = []
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

    def add_box(self, group, hue, data):
        """
        Add a new dataset for a given group and hue.

        Parameters
        ----------
        group : str
            Group/category name.
        hue : str
            Subgroup or condition name.
        data : array-like
            Numeric data values for this group-hue combination.
        """
        if group not in self.data:
            self.data[group] = {}
            self.groups.append(group)
        self.data[group][hue] = data
        self.hues.add(hue)

    def plot(self):
        """
       Generate the grouped boxplot with optional hue-based color coding.

       This method:
           1. Calculates box positions for each group-hue combination.
           2. Draws boxplots and jittered data points.
           3. Adds group labels and a hue legend.

       Notes
       -----
       - Automatically handles spacing between groups and hues.
       - Colors are taken from `palette` or Matplotlib’s default color cycle.
       """
        positions = []
        box_data = []
        box_colors = []
        tick_labels = []

        hue_list = sorted(self.hues)
        group_spacing = 1.0
        box_width = 0.6
        hue_spacing = box_width / len(hue_list)

        for i, group in enumerate(self.groups):
            base_pos = i * (group_spacing + box_width)
            for j, hue in enumerate(hue_list):
                if hue in self.data[group]:
                    pos = base_pos + j * hue_spacing
                    positions.append(pos)
                    box_data.append(self.data[group][hue])
                    color = self.palette.get(hue, f"C{j}")
                    box_colors.append(color)
            tick_labels.append(group)

        # Plot boxplots
        bp = self.ax.boxplot(box_data, positions=positions, widths=hue_spacing * 0.8,medianprops={'color': 'black'}, patch_artist=True, showfliers=False)

        # Color boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)

        # Add jittered dots
        for pos, data, color in zip(positions, box_data, box_colors):
            x = np.random.normal(pos, hue_spacing * 0.2, size=len(data))
            self.ax.plot(x, data, marker='o', linestyle='None', markersize=1, alpha=0.5, color=color)


        # Set labels and legend
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_xticks([i * (group_spacing + box_width) + box_width / 2 for i in range(len(self.groups))])
        self.ax.set_xticklabels(tick_labels, rotation=45)

        handles = [plt.Line2D([0], [0], color=self.palette.get(hue, f"C{i}"), lw=4) for i, hue in enumerate(hue_list)]
        self.ax.legend(handles, hue_list, title="Condition", loc='best')

        plt.tight_layout()
        plt.show()

class BoxPlotter(Plotter):
    """
    A simple box plot class for visualizing distributions across multiple groups.

    Inherits from
    --------------
    Plotter

    Attributes
    ----------
    data : list of array-like
        List of datasets, one per group.
    labels : list of str
        Labels corresponding to each dataset.

    Methods
    -------
    add_box(data, label=None)
        Add a new dataset and draw its box plot.
    add_jittered_dots()
        Overlay random scatter points to show individual data distribution.
    add_statistical_annotations()
        Perform t-tests between groups and annotate significance levels.
    """
    def __init__(self, title="Box Plot", xlabel="X-axis", ylabel="Y-axis"):
        super().__init__(title, xlabel, ylabel)
        self.data = []
        self.labels = []
    def add_box(self, data, label=None):
        """
        Add a dataset as a box to the plot.

        Parameters
        ----------
        data : array-like
            Numeric data values for the new group.
        label : str, optional
            Group label for the dataset. If None, auto-named as "Group N".
        """
        self.data.append(data)
        if label:
            self.labels.append(label)
        else:
            self.labels.append(f"Group {len(self.data)}")
        self.ax.clear()  # Clear the previous plot
        self.ax.boxplot(self.data, labels=self.labels, showfliers=False)
        self.set_labels()
        self.add_jittered_dots()
    def add_jittered_dots(self):
        """
        Overlay jittered scatter points on top of boxplots.

        Adds random X-axis noise to display individual data points, providing
        a visual indication of data distribution within each group.
        """
        for i, data in enumerate(self.data):
            x = np.random.normal(i + 1, 0.04, size=len(data))
            self.ax.plot(x, data, 'r.', alpha=0.5)
    def add_statistical_annotations(self):
        """
        Perform pairwise t-tests between all boxplot groups and annotate results.

        Adds horizontal bars and p-value text labels between group pairs.

        Notes
        -----
        - Uses independent two-sample t-tests from `scipy.stats.ttest_ind`.
        - P-values are displayed in scientific notation (e.g., 1.23e-04).
        """
        num_groups = len(self.data)
        y_max = max([max(group) for group in self.data])
        y_min = min([min(group) for group in self.data])
        y_range = y_max - y_min
        y_offset = y_range * 0.1

        for i in range(num_groups):
            for j in range(i + 1, num_groups):
                t_stat, p_val = ttest_ind(self.data[i], self.data[j])
                x1, x2 = i + 1, j + 1
                y = y_max + y_offset * (j - i)
                self.ax.plot([x1, x1, x2, x2], [y, y + y_offset, y + y_offset, y], lw=1.5, c='k')
                self.ax.text((x1 + x2) * 0.5, y + y_offset, f"p = {p_val:.3e}", ha='center', va='bottom')

class HistogramPlotter(Plotter):
    """
    A class for creating histograms with optional density normalization.

    Inherits from
    --------------
    Plotter

    Attributes
    ----------
    density : bool
        Whether to normalize histogram heights to form a probability density.
    colors : list of str
        Matplotlib default color cycle used for sequential datasets.
    color_index : int
        Index of the next color to use.

    Methods
    -------
    add_data(data, bins=50, label="Histogram", alpha=0.5)
        Add a histogram for a dataset to the plot.
    """
    def __init__(self, density = False, title="Histogram", xlabel="X-axis", ylabel="Y-axis", figsize=(6.4, 4.8)):
        super().__init__(xlabel=xlabel, ylabel=ylabel, figsize=figsize)
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # default matplotlib colors
        self.color_index = 0
        self.density = density
    def add_data(self, data, bins=50, label="Histogram", alpha=0.5):
        """
       Add a histogram to the plot.

       Parameters
       ----------
       data : array-like
           Numeric dataset to be visualized.
       bins : int, optional
           Number of histogram bins. Default is 50.
       label : str, optional
           Label for the dataset (used in legend). Default is "Histogram".
       alpha : float, optional
           Transparency level of the bars. Default is 0.5.
       """
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        self.ax.hist(data, density = self.density,  bins=bins, label=label, color=color, alpha=alpha)

class TrackPlotter(Plotter):
    """
   A specialized plotter for visualizing particle or molecule trajectories on image projections.

   The `TrackPlotter` class extends the base `Plotter` to display super-resolution
   or microscopy data as maximum-intensity projections with overlaid localization
   or tracking results. Tracks are plotted in nanometers, optionally cropped to a
   defined region of interest (ROI).

   Inherits from
   --------------
   Plotter

   Attributes
   ----------
   image : numpy.ndarray
       3D image stack (frames × height × width) from which a maximum projection is derived.
   px2nm : float
       Conversion factor from pixels to nanometers.
   fig : matplotlib.figure.Figure
       Matplotlib figure object inherited from Plotter.
   ax : matplotlib.axes.Axes
       Axes object for plotting inherited from Plotter.

   Methods
   -------
   show_max_projection(contour=None)
       Display the maximum intensity projection of the 3D image, optionally cropped by a contour.
   plot_tracks(df, tid, id_col='track.id', x_col='x', y_col='y', color='blue', offset=(0, 0))
       Overlay an individual track on the current image.
   save_plot(filename, dpi=300)
       Save the resulting plot to a file.
   """
    def __init__(self, image, px2nm=108, title="Track Plot", xlabel="X (nm)", ylabel="Y (nm)", figsize = (6.4, 6.4)):
        super().__init__(title, xlabel, ylabel, figsize)
        self.image = image
        self.px2nm = px2nm
        self.ax.set_aspect('equal')
    def show_max_projection(self, contour=None):
        """
        Display the maximum intensity projection of the image stack.

        Parameters
        ----------
        contour : numpy.ndarray, optional
            2D array of (x, y) coordinates defining a polygonal ROI contour.
            If provided, the projection is cropped to this region.

        Returns
        -------
        tuple of int
            (x_offset, y_offset) representing the crop origin. Returns (0, 0)
            if no contour is provided.
        """
        img = np.max(self.image, axis=0)
        if contour is not None:
            x0, x1 = int(min(contour[:, 0])), int(max(contour[:, 0]))
            y0, y1 = int(min(contour[:, 1])), int(max(contour[:, 1]))
            img = img[y0:y1, x0:x1]
        self.ax.imshow(img, cmap='gray')
        return (x0 if contour is not None else 0), (y0 if contour is not None else 0)
    def plot_tracks(self, df, tid, id_col='track.id', x_col='x', y_col='y', color='blue', offset=(0, 0)):
        """
        Plot a single trajectory on top of the displayed image.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing track data.
        tid : int or str
            Track identifier corresponding to `id_col` values.
        id_col : str, optional
            Column name for track IDs. Default is 'track.id'.
        x_col : str, optional
            Column name for X coordinates. Default is 'x'.
        y_col : str, optional
            Column name for Y coordinates. Default is 'y'.
        color : str, optional
            Color for the plotted trajectory line. Default is "blue".
        offset : tuple of float, optional
            (x_offset, y_offset) to shift coordinates (useful when plotting cropped images).
        """
        track = df[df[id_col] == tid]
        self.ax.plot(track[x_col]/self.px2nm - offset[0], track[y_col]/self.px2nm - offset[1], color=color)
        self.ax.grid(False)
        self.ax.axis('off')
    def save_plot(self, filename, dpi = 300):
        """
        Save the current track plot to a file.

        Parameters
        ----------
        filename : str
            Path to save the figure (e.g., 'output/track_plot.png').
        dpi : int, optional
            Dots per inch (image resolution). Default is 300.
        """
        self.fig.savefig(filename,dpi = dpi, bbox_inches='tight', pad_inches=0)

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
        if channel == 'ch0':
            stats = self.stats0
        elif channel == 'ch1':
            if self.stats1 is None:
                # Return empty DataFrame with expected columns
                return None
            else:
                stats = self.stats1
        else:
            raise ValueError("channel must be 'ch0' or 'ch1'")
    
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
    def _otsu_threshold(self, original, image, mode = "max"):
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
        yaml_file = self._openyaml()
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
                                   self._get_px2nm(),   
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
                                   nm2px = self._get_px2nm(), 
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
                                   nm2px = self._get_px2nm(), 
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
            yaml_file = self._openyaml()
            ch0_laser = yaml_file['ch0']
            print(f'\nThe tarcks have been colocalized with {ch0_laser}nm as reference channel')

        else:
            print('The tracks have not been colocalized')   
    def _openyaml(self):
        name = glob(self.folder + '/**/**_colocsTracks.yaml', recursive=True)
        if name:
            with open(name[0], 'r') as file:
                data_parts = list(yaml.safe_load_all(file))
            data = {}
            for i in data_parts:
                data.update(i)
            return data
        else:
            return False
    def _get_px2nm(self): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        resultPath  = glob(self.folder + '/**/**result.txt', recursive=True)[0]
        result_txt  = tools.read_result_file(resultPath) #this opens the results.txt file to check the microscope used. 
                #It should be in a folder called paramfile inside the folder where the script is located. 
        if result_txt['Computer'] == 'ANNAPURNA': 
            return 90.16
        elif result_txt['Computer'] == 'K2-BIVOUAC':
            return 108

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
        """
            Compute mean intensity of a local patch around a spot in an image.
        
            Parameters
            ----------
            image : np.ndarray
                2D image array.
            x : float
                x-coordinate of the spot (pixels).
            y : float
                y-coordinate of the spot (pixels).
            window : int, optional
                Half-width of the square patch (total size = 2*window+1), by default 1
        
            Returns
            -------
            float
                Mean intensity in the patch. Returns NaN if patch is empty.
        """
        xi, yi = int(round(x)), int(round(y))
        x_min, x_max = max(0, xi - window), min(image.shape[1], xi + window + 1)
        y_min, y_max = max(0, yi - window), min(image.shape[0], yi + window + 1)
        patch = image[y_min:y_max, x_min:x_max]
        return np.mean(patch) if patch.size > 0 else np.nan

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
        general_yaml_file = self._openyaml(path_yaml)
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
                    except: 
                        ch0_stats = None
                        ch1_stats = None
                        coloc_stats = None
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
                    except: 
                        ds = pd.DataFrame()
                    
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
            If available, it is automatically determined using :meth:`_get_frame_rate`.
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
            frame_rate = self._get_frame_rate()
        except: 
            frame_rate = frame_rate
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
                except Exception as e:
                    self.failed_folders.append((run_folder, str(e)))
                    print(f"Skipping folder {run_folder} due to error: {e}")
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
    def _get_px2nm(self): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        resultPath  = glob(self.folder + '/**/**result.txt', recursive=True)[0]
        result_txt  = tools.read_result_file(resultPath) #this opens the results.txt file to check the microscope used. 
                #It should be in a folder called paramfile inside the folder where the script is located. 
        if result_txt['Computer'] == 'ANNAPURNA': 
            return 90.16
        elif result_txt['Computer'] == 'K2-BIVOUAC':
            return 108
    def _get_frame_rate(self): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        resultPath  = glob(self.folder + '/**/**result.txt', recursive=True)[0]
        result_txt  = tools.read_result_file(resultPath) #this opens the results.txt file to check the microscope used. 
                #It should be in a folder called paramfile inside the folder where the script is located. 
        time_unit = result_txt['Interval']
        time = time_unit.split(' ')[0]
        unit = time_unit.split(' ')[1]
        if unit == 'sec':
            return float(time)
        elif unit == 'ms':
            return int(time)/1000
        elif unit == 'min':
            return int(time)*1000
           
    def _count_run_folders_recursive(self, root_folder):
        pattern = re.compile(r'^Run\d+$')
        count = 0

        for dirpath, dirnames, _ in os.walk(root_folder):
            for dirname in dirnames:
                if pattern.match(dirname):
                    count += 1
        return count
    def _openyaml(self, name):
        # Load YAML content if found
        with open(name[0], 'r') as file:
            data_parts = list(yaml.safe_load_all(file))
        yaml_data = {}
        for part in data_parts:
            yaml_data.update(part)
        return yaml_data
    def validate(self):
        print("Just a reminder that most of the time the whole dataset should be analyzed using the same parameters.")
        print("Here are the parameters for the first folder in the dataset that has colocalized tracks:")
        yaml = self._openyaml(glob(self.folder + r'/**/*_colocsTracks.yaml', recursive=True))
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
        general_yaml_file = self._openyaml(path_yaml)
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
                                yaml_data = self._openyaml(yaml_files)
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
    def get_dwell(self, min_len = 10, ref = 'ch0', x0=0, xt=None, y0=0, yt=None):
        all_dwell = []
        frame_rate = self._get_frame_rate()
        hist = HistogramPlotter( xlabel="dwell_time(sec)", ylabel="Frequency")
        for i, cond in tqdm(zip(self._conditions_paths, self.conditions_to_use), desc='Extracting Ds...\n'):
            print(f'\nAnalyzing {i}...')
            dwell_cond = []
            pathshdf = glob(i + '/**/**colocsTracks_stats.hdf', recursive=True)
            paths_locs = list(set(os.path.dirname(file) for file in pathshdf))
            for j in tqdm(paths_locs):
                pathsyaml = glob(j + '/**/**colocsTracks.yaml', recursive=True)
                yaml = self._openyaml(pathsyaml)
                image = Single_tracked_folder(j).open_files()
                dwell = image.extract_dwell(frame_rate = frame_rate, min_len = min_len, max_dist = yaml['th'], ref = ref)
                if isinstance(dwell, pd.DataFrame):
                # # Add columns at the beginning
                    dwell.insert(0, 'condition', cond)
                    dwell.insert(0, 'run', j)
                    all_dwell.append(dwell)  # Accumulate
                    dwell_cond.append(dwell)
            final_cond = pd.concat(dwell_cond, ignore_index=True)
            hist.add_data(final_cond.dwell_time, label = f"{cond}")
        final_dwell = pd.concat(all_dwell, ignore_index=True)
        hist.set_labels()
        hist.set_xlim(x0, xt)
        hist.set_ylim(y0, yt)
        hist.show_plot()
        # box.add_statistical_annotations()
        return final_dwell, hist
    def validate(self):
        print("Just a reminder that most of the time the whole dataset should be analyzed using the same parameters.")
        print("Here are the parameters for the first folder in the dataset that has colocalized tracks:")
        yaml = self._openyaml(glob(self.folder + r'/**/*_colocsTracks.yaml', recursive=True))
        for i, j in enumerate(yaml.items()):
            print(f"{j[0]}: {j[1]}")
    def _get_px2nm(self): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        resultPath  = glob(self.folder + '/**/**result.txt', recursive=True)[0]
        result_txt  = tools.read_result_file(resultPath) #this opens the results.txt file to check the microscope used. 
                #It should be in a folder called paramfile inside the folder where the script is located. 
        if result_txt['Computer'] == 'ANNAPURNA': 
            return 90.16
        elif result_txt['Computer'] == 'K2-BIVOUAC':
            return 108
    def _get_frame_rate(self): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        resultPath  = glob(self.folder + '/**/**result.txt', recursive=True)[0]
        result_txt  = tools.read_result_file(resultPath) #this opens the results.txt file to check the microscope used. 
                #It should be in a folder called paramfile inside the folder where the script is located. 
        time_unit = result_txt['Interval']
        time = time_unit.split(' ')[0]
        unit = time_unit.split(' ')[1]
        if unit == 'sec':
            return float(time)
        elif unit == 'ms':
            return int(time)/1000
        elif unit == 'min':
            return int(time)*1000
           
    def _count_run_folders_recursive(self, root_folder):
        pattern = re.compile(r'^Run\d+$')
        count = 0

        for dirpath, dirnames, _ in os.walk(root_folder):
            for dirname in dirnames:
                if pattern.match(dirname):
                    count += 1
        return count
    def _openyaml(self, name):
        # Load YAML content if found
        with open(name[0], 'r') as file:
            data_parts = list(yaml.safe_load_all(file))
        yaml_data = {}
        for part in data_parts:
            yaml_data.update(part)
        return yaml_data