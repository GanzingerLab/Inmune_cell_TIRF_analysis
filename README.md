# postSPIT

`postSPIT` is a Python package for post-processing SPIT-based microscopy analysis.

It provides tools for loading tracked SPIT outputs, analyzing TIRF microscopy movies, detecting protein clusters, combining tracked spots with cluster masks, filtering and re-tracking spots, re-running colocalization, extracting diffusion coefficients and dwell times, and plotting analysis results.

## Installation

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

## Usage examples

Import the main analysis classes directly from `postSPIT`:

```python
from postSPIT import (
    Cell_Analyzer,
    Combined_analysis,
    Dataset_combined_analysis,
    Single_tracked_folder,
)
```

### Open tracked SPIT output

```python
from postSPIT import Single_tracked_folder

folder = r"path/to/Run1"

tracked_folder = Single_tracked_folder(
    folder,
    ch0_hint="488nm",
    ch1_hint="561nm",
)

tracked_image = tracked_folder.open_files()
```

### Analyze clusters

```python
from postSPIT import Cell_Analyzer

folder = r"path/to/Run1"

cells = Cell_Analyzer(
    folder,
    ch0_wl="488nm",
    ch1_wl="561nm",
)

clusters_binary, clusters, cluster_stats, linked_clusters, linked_stats = (
    cells.analyze_clusters_protein(
        min_size=80,
        ch="ch0",
        th_method="li_local",
        global_th_mode="max",
        window_size=15,
        p=2,
        q=6,
        save_videos=False,
        overwrite=False,
    )
)
```

### Combine tracked spots with cluster analysis

```python
from postSPIT import Combined_analysis

folder = r"path/to/Run1"

analysis = Combined_analysis(
    folder,
    ch0_hint="488nm",
    ch1_hint="561nm",
)

results = analysis.combine_spots_clusters(
    min_size=80,
    ch="ch0",
    th_method="li_local",
    global_th_mode="max",
    window_size=15,
    p=2,
    q=6,
    save_videos=False,
    overwrite=False,
)
```

### Re-track filtered spots

```python
analysis.retrack(overwrite=False)
```

### Re-run colocalization

```python
analysis.recoloc_tracks(overwrite=False)
```

### Extract filtered diffusion coefficients

```python
ds = analysis.extract_Ds_filtered(
    mature_class=1,
    min_len=10,
    ch="ch0",
)
```

### Extract filtered dwell times

```python
dwell = analysis.extract_dwell_filtered(
    mature_class=1,
    frame_rate=None,
    min_len=10,
    max_dist=250,
    ref="ch0",
    ch_maturation_selection="ch1",
)
```

## Package structure

```text
postSPIT/
├── __init__.py
├── tirf_analysis.py
├── plotting.py
├── tracked.py
├── cells.py
├── combined.py
├── datasets.py
├── io_utils.py
├── image_processing.py
└── stats_utils.py
```

