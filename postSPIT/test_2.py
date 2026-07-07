"""
Basic package tests for postSPIT.

These tests are intentionally lightweight:
- They check that the reorganized modules import correctly.
- They check that the public classes are exposed from postSPIT.
- They check that key methods still exist on the expected classes.

They do not require real microscopy data.
"""

import inspect


def test_public_imports_from_package_root():
    """The main public classes should be importable from postSPIT."""
    from postSPIT import (
        Plotter,
        LinePlotter,
        ScatterPlotter,
        HueBoxPlotter,
        BoxPlotter,
        HistogramPlotter,
        TrackPlotter,
        Tracked_image,
        Single_tracked_folder,
        Cell_Analyzer,
        Combined_analysis,
        Dataset_combined_analysis,
    )

    assert inspect.isclass(Plotter)
    assert inspect.isclass(LinePlotter)
    assert inspect.isclass(ScatterPlotter)
    assert inspect.isclass(HueBoxPlotter)
    assert inspect.isclass(BoxPlotter)
    assert inspect.isclass(HistogramPlotter)
    assert inspect.isclass(TrackPlotter)
    assert inspect.isclass(Tracked_image)
    assert inspect.isclass(Single_tracked_folder)
    assert inspect.isclass(Cell_Analyzer)
    assert inspect.isclass(Combined_analysis)
    assert inspect.isclass(Dataset_combined_analysis)


def test_module_imports():
    """The new module layout should import cleanly."""
    import postSPIT.plotting
    import postSPIT.tracked
    import postSPIT.cells
    import postSPIT.combined
    import postSPIT.datasets

    assert hasattr(postSPIT.plotting, "Plotter")
    assert hasattr(postSPIT.tracked, "Tracked_image")
    assert hasattr(postSPIT.tracked, "Single_tracked_folder")
    assert hasattr(postSPIT.cells, "Cell_Analyzer")
    assert hasattr(postSPIT.combined, "Combined_analysis")
    assert hasattr(postSPIT.datasets, "Dataset_combined_analysis")


def test_plotter_basic_instantiation():
    """Plotter classes should be instantiable without data files."""
    from postSPIT import (
        Plotter,
        LinePlotter,
        ScatterPlotter,
        BoxPlotter,
        HistogramPlotter,
    )

    plotter = Plotter(title="Test", xlabel="x", ylabel="y")
    assert plotter.title == "Test"
    assert plotter.xlabel == "x"
    assert plotter.ylabel == "y"
    assert plotter.fig is not None
    assert plotter.ax is not None

    line_plotter = LinePlotter()
    assert line_plotter.fig is not None
    assert line_plotter.ax is not None

    scatter_plotter = ScatterPlotter()
    assert scatter_plotter.fig is not None
    assert scatter_plotter.ax is not None

    box_plotter = BoxPlotter()
    assert box_plotter.fig is not None
    assert box_plotter.ax is not None

    histogram_plotter = HistogramPlotter()
    assert histogram_plotter.fig is not None
    assert histogram_plotter.ax is not None


def test_expected_methods_on_tracked_image():
    """
    Tracked_image should contain tracked-particle methods.

    It should not be expected to contain Cell_Analyzer-only methods such as split_cells().
    """
    from postSPIT import Tracked_image

    expected_methods = [
        "plot_tracks",
        "plot_colocs",
        "plot_intensity_coloc",
        "extract_Ds",
        "extract_dwell",
    ]

    for method_name in expected_methods:
        assert hasattr(Tracked_image, method_name), f"Missing method: {method_name}"

    assert not hasattr(
        Tracked_image,
        "split_cells",
    ), "split_cells belongs to Cell_Analyzer, not Tracked_image"


def test_expected_methods_on_cell_analyzer():
    """Cell_Analyzer should contain cell, ROI, cluster, and maturation methods."""
    from postSPIT import Cell_Analyzer

    expected_methods = [
        "split_cells",
        "analyze_clusters_protein",
        "predict_maturation",
        "get_time_interval",
        "_li_threshold",
        "_otsu_threshold",
        "_phansalkar_threshold",
        "_create_mask",
        "_summarize_clusters_per_cell_frame",
    ]

    for method_name in expected_methods:
        assert hasattr(Cell_Analyzer, method_name), f"Missing method: {method_name}"


def test_expected_methods_on_combined_analysis():
    """Combined_analysis should contain spot-cluster integration methods."""
    from postSPIT import Combined_analysis

    expected_methods = [
        "combine_spots_clusters",
        "remove_spots_within_clusters",
        "retrack",
        "recoloc_tracks",
        "extract_Ds_filtered",
        "extract_dwell_filtered",
    ]

    for method_name in expected_methods:
        assert hasattr(Combined_analysis, method_name), f"Missing method: {method_name}"


def test_expected_methods_on_dataset_combined_analysis():
    """Dataset_combined_analysis should contain batch-analysis methods."""
    from postSPIT import Dataset_combined_analysis

    expected_methods = [
        "select_conditions",
        "analyze_clusters_protein",
        "predict_maturation",
        "combine_spots_clusters",
        "retrack",
        "recoloc_tracks",
        "get_Ds",
        "get_dwell",
    ]

    for method_name in expected_methods:
        assert hasattr(Dataset_combined_analysis, method_name), f"Missing method: {method_name}"