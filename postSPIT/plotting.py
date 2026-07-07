import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind


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