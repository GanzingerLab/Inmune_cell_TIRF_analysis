# -*- coding: utf-8 -*-
from postSPIT import tirf_analysis as plc
import matplotlib.pyplot as plt


a = plc.Combined_analysis(r'D:\Data\Chi_data\first data\output2\Run00002')
# b = a.tracked.stats0
clusters_binary, result, results_stats, linked_df, linked_stats, spots_filtered = \
    a.combine_spots_clusters(ch = 'ch0', save_videos=False)
clusters_binary, result, results_stats, linked_df, linked_stats, spots_filtered = \
    a.combine_spots_clusters(ch = 'ch1', q = 8,  save_videos=False)
# %%

a.retrack()

# %%
a.recoloc_tracks()
# %%


b  = a.spots_outside_clusters
# c = a.spots_outside_clusters_stats
im = a.clusters.sep_cells

# %%

results_stats.columns

# %%

import matplotlib.pyplot as plt
import seaborn as sns

# Example: results_stats has 'cell_id', 'frame', 'spot_count'
plt.figure(figsize=(10, 6))

# Line plot for each cell
sns.lineplot(
    data=results_stats,
    x='frame',
    y='spot_mean_intensity',
    hue='cell_id',
    marker='o'
)

plt.xlabel('Frame')
plt.ylabel('Number of spots')
plt.title('Number of spots per cell per frame')
plt.legend(title='Cell ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
