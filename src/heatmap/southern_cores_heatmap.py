"""
http://nbviewer.ipython.org/github/ucsd-scientific-python/user-group/blob/master/presentations/20131016/hierarchical_clustering_heatmaps_gridspec.ipynb
"""

import os
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as sch


#input_file_path = os.path.expanduser(sys.argv[1])
input_file_path = os.path.expanduser('~/Dropbox/as_fe_data_sln/minnesota_cores/dissolution_results/southern_counties_tills_2010_dissolution_results.txt')
# similarity should be 'cosine' or 'correlation'
#similarity = sys.argv[2]
similarity = 'cosine'
# from UMRB2_combined_lumps.txt core_df = pd.read_csv(input_file_path, sep='\t', index_col=0)

# from ~/Dropbox/as_fe_data_sln/minnesota_cores/wet_usgs/wet_usgs_results.txt
all_core_df = pd.read_csv(input_file_path, sep='\t', index_col='field_no')
# remove ' ppm' from trace element column headers
new_column_headers = []
for column_header in all_core_df.columns:
    if column_header.endswith('ppm'):
        new_column_header = column_header[:-4]
    else:
        new_column_header = column_header
    new_column_headers.append(new_column_header)
all_core_df.columns = new_column_headers

# remove Mn, Se, Sr they are not informative
core_df = all_core_df.loc[:, 'As':'Se']
core_df.drop(['Mn', 'P', 'Ba', 'Sr'], axis=1, inplace=True)

# keep just OTT3, UMRB2, TG3
##core_df.drop(
##    [c for c in core_df.index if not re.search('OTT3|UMRB2|TG3', c)],
##    axis=0,
##    inplace=True
##)


print(core_df)

figure_size = (10, 10)

# font size for figures
#  14 for UMRB2
plt.rcParams.update({'font.size': 6})
# Arial font
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})


# helper for cleaning up axes by removing ticks, tick labels, frame, etc.
def clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

# look at raw data
axi = plt.imshow(core_df, interpolation='nearest', cmap=plt.cm.Greens)
ax = axi.get_axes()
clean_axis(ax)
plt.show()

# calculate pairwise distances for rows
pairwise_dists = distance.squareform(distance.pdist(core_df, similarity))
# cluster
row_clusters = sch.linkage(pairwise_dists, method='complete')

# calculate pairwise distances for columns
col_pairwise_dists = distance.squareform(distance.pdist(core_df.T, similarity))
# cluster
col_clusters = sch.linkage(col_pairwise_dists, method='complete')

# make dendrograms black rather than letting scipy color them
sch.set_link_color_palette(['black'])

# plot the results
fig = plt.figure(figsize=figure_size)
#fig.suptitle(os.path.split(input_file_path)[1])
heatmapGS = gridspec.GridSpec(2, 2, wspace=0.0, hspace=0.0, width_ratios=[0.25, 1], height_ratios=[0.25, 1])

### col dendrogram ####
col_denAX = fig.add_subplot(heatmapGS[0, 1])
col_denD = sch.dendrogram(col_clusters, color_threshold=np.inf)
clean_axis(col_denAX)

### row dendrogram ###
row_denAX = fig.add_subplot(heatmapGS[1, 0])
row_denD = sch.dendrogram(row_clusters, color_threshold=np.inf, orientation='right')
clean_axis(row_denAX)

### heatmap ###
heatmapAX = fig.add_subplot(heatmapGS[1, 1])
axi = heatmapAX.imshow(core_df.ix[row_denD['leaves'], col_denD['leaves']], interpolation='nearest', aspect='auto',
                       origin='lower', cmap=plt.cm.Greens)
clean_axis(heatmapAX)

## row labels ##
heatmapAX.set_yticks(np.arange(core_df.shape[0]))
heatmapAX.yaxis.set_ticks_position('right')
heatmapAX.set_yticklabels(
    core_df.index[row_denD['leaves']]
    + '  '
    + all_core_df.sample_description[row_denD['leaves']]
    + '  '
    + all_core_df.MGS_provinance[row_denD['leaves']]
)

# remove the tick lines
for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines():
    l.set_markersize(0)

# col labels ##
heatmapAX.set_xticks(np.arange(core_df.shape[1]))
xlabelsL = heatmapAX.set_xticklabels(core_df.columns[col_denD['leaves']])
## rotate labels 90 degrees
##for label in xlabelsL:
##    label.set_rotation(90)
# remove the tick lines
for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines():
    l.set_markersize(0)


fig.tight_layout()

input_dir_path, input_file_name = os.path.split(input_file_path)
input_file_base_name, input_file_ext = os.path.splitext(input_file_name)
output_file_path = os.path.join(input_dir_path, input_file_base_name + '_' + similarity + '.pdf')
print('writing output to {}'.format(output_file_path))
plt.savefig(output_file_path)

plt.show()
