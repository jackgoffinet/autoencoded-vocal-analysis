"""
Plot 3: Mouse and zebra finch clusters.

"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"


from matplotlib import rcParams
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import numpy as np
import os

from ava.data.data_container import DataContainer
from ava.plotting.latent_projection import latent_projection_plot_DC, \
	latent_projection_plot_with_noise_DC
from ava.plotting.rolloff_plot import clustering_performance_plot_splits
from ava.plotting.grid_plot import indexed_grid_plot


MOUSE_COLOR = (154/256,155/256,77/256)
MOUSE_1 = (183/256,105/256,52/256)
MOUSE_2 = (172/256,163/256,63/256)
MOUSE_HIGHLIGHT = (105/256,140/256,224/256)
FINCH_COLOR = (163/256,86/256,141/256)
FINCH_1 = (180/256,80/256,147/256)
FINCH_2 = (112/256,99/256,186/256)
FINCH_HIGHLIGHT = (101/256,228/256,203/256)


if __name__ == '__main__':
	# Define data.
	root = '/media/jackg/Jacks_Animal_Sounds/mice/BM030/'
	proj_dirs = [root+'proj']
	spec_dirs = [root+'specs']
	feature_dirs = [root+'mupet']
	model_fn = root + 'checkpoint_040.tar'

	dc_single_mouse = DataContainer(projection_dirs=proj_dirs, \
		spec_dirs=spec_dirs, model_filename=model_fn, feature_dirs=feature_dirs)

	# latent = dc_single_mouse.request('latent_means')
	# latent_projection_plot_DC(dc_single_mouse, show_axis=True)
	# quit()

	root = '/media/jackg/Jacks_Animal_Sounds/mice/MUPET/'
	proj_dirs = [root+'C57_projections', root+'DBA_projections']
	mupet_feature_dirs = [root+'C57_mupet', root+'DBA_mupet']
	ds_feature_dirs = [root+'C57_deepsqueak', root+'DBA_deepsqueak']
	spec_dirs = [root+'C57_specs', root+'DBA_specs']
	model_fn = root + 'mupet_checkpoint_150.tar'

	dc_mouse = DataContainer(projection_dirs=proj_dirs, \
		feature_dirs=mupet_feature_dirs, spec_dirs=spec_dirs, \
		model_filename=model_fn)

	# from ava.plotting.pairwise_distance_plots import random_walk_plot
	# from_indices=[17047]*3+[17470]*3
	# random_walk_plot(dc_mouse, from_indices=from_indices)
	# quit()


	path = [\
		[411,2747,10441,23196,12051,27193,17736,10275,24884,3018,10957,3561,3441], \
		# [1217,1254,1320,13062,16125,13820,14985,12025,17318], \
		# [485,452,941,5433,6603,1893,2074,2767,8038,8657,18889,22954,17269,21177], \
		[452,485,941,5433,5532,6086,11223,11382,17104,20382,30434,26759,26408], \
		[22319,9900,15785,14053,15329,14680,15991,15209,12603,12396,17047,14398,17470], \
		[27347,30417,19133, 24768, 24348, 23986, 23971, 24136, 19840,17255,18545,18558,26576], \
	]
	# max_len = max(len(i) for i in path)
	# for i in range(len(path)):
	# 	path[i] += [0]*(max_len - len(path[i]))
	# print(path)
	# print()
	# indexed_grid_plot(dc_mouse, path, gap=(8,4))
	# quit()

	root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/syll/'
	proj_dirs = [root+'DIR_proj', root+'UNDIR_proj']
	feature_dirs = [root+'DIR_features', root+'UNDIR_features']
	spec_dirs = [root+'DIR_specs', root+'UNDIR_specs']
	model_fn = root + 'checkpoint_080.tar'

	sap_fields = ['syllable_duration_sap', 'mean_amplitude',
		'mean_pitch', 'mean_FM', 'mean_AM2', 'mean_entropy', 'mean_pitch_goodness',
		'mean_mean_freq', 'pitch_variance', 'FM_variance', 'entropy_variance',
		'pitch_goodness_variance', 'mean_freq_variance', 'AM_variance']

	dc_finch = DataContainer(projection_dirs=proj_dirs, \
		feature_dirs=feature_dirs, spec_dirs=spec_dirs, \
		model_filename=model_fn)


	# Plot.
	params = {
		'axes.labelsize': 7,
		'axes.labelpad': 0.05,
		'legend.fontsize': 7,
		'xtick.labelsize': 7,
		'ytick.labelsize': 7,
		'text.usetex': False,
		'figure.figsize': [6.5, 3.5] # 4, 1.8
	}
	rcParams.update(params)
	fig = plt.figure()

	gsarr = [gridspec.GridSpec(1,1) for _ in range(5)]

	gsarr[0].update(left=0.02, right=0.33, top=0.95, bottom=0.53)
	gsarr[1].update(left=0.35, right=0.65, top=0.95, bottom=0.53)
	gsarr[2].update(left=0.67, right=0.96, top=0.95, bottom=0.50)
	gsarr[3].update(left=0.12, right=0.29, top=0.48, bottom=0.05)
	gsarr[4].update(left=0.35, right=0.95, top=0.48, bottom=0.04)

	axarr = [plt.Subplot(fig, gs[0,0]) for gs in gsarr]

	subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
	# for ax in [axarr[3]]:
		# fig.add_subplot(ax)
	for l, ax in zip(subplot_labels, axarr):
		fig.add_subplot(ax)
		if l not in ['d', 'e', 'f']:
			ax.axis('off')

	colors = [to_rgba(i) for i in [FINCH_COLOR, MOUSE_COLOR]]

	latent_projection_plot_with_noise_DC(dc_single_mouse, alpha=0.15, s=0.3, ax=axarr[1], \
		save_and_close=False, default_color=colors[1], noise_box=[-9,-4.5,-4,2])

	latent_projection_plot_DC(dc_finch, alpha=0.08, s=0.2, ax=axarr[0], \
		save_and_close=False, default_color=colors[0])

	latent_projection_plot_with_noise_DC(dc_single_mouse, alpha=0.25, s=0.3, color_by='cluster', \
		ax=axarr[2], colormap='tab10', save_and_close=False, noise_box=[-9,-4.5,-4,2])


	temp_axarr = [axarr[3]]
	labels = ["Zebra Finch", "Mouse"]
	noise_boxes = [None, [-9,-4.5,-4,2]]
	clustering_performance_plot_splits([dc_finch, dc_single_mouse], labels, axarr=temp_axarr, \
		load_data=True, save_and_close=False, colors=colors, noise_boxes=noise_boxes)
	axarr[3].set_yticks([0.0,0.1,0.2,0.3])


	indexed_grid_plot(dc_mouse, path, ax=axarr[4], gap=(8,4), \
		save_and_close=False)

	plt.text(0.02,0.93,'a',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.34,0.93,'b',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.67,0.93,'c',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.02,0.48,'d',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.32,0.48,'e',transform=fig.transFigure, size=14, weight='bold')


	plt.savefig('plot_4.pdf')
	plt.close('all')


###
