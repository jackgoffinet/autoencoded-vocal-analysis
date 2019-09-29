"""
Plot 2: Nearest Neighbors

"""

from matplotlib import rcParams
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import numpy as np
import os

from ava.data.data_container import DataContainer
from ava.plotting.pairwise_distance_plots import knn_display_DC
from ava.plotting.grid_plot import indexed_grid_plot




if __name__ == '__main__':
	# Define data.
	root = '/media/jackg/Jacks_Animal_Sounds/mice/MUPET/'
	proj_dirs = [root+'C57_projections', root+'DBA_projections']
	mupet_feature_dirs = [root+'C57_mupet', root+'DBA_mupet']
	ds_feature_dirs = [root+'C57_deepsqueak', root+'DBA_deepsqueak']
	spec_dirs = [root+'C57_specs', root+'DBA_specs']
	model_fn = root + 'mupet_checkpoint_150.tar'


	dc = DataContainer(projection_dirs=proj_dirs, \
		feature_dirs=mupet_feature_dirs, spec_dirs=spec_dirs, \
		model_filename=model_fn)


	mupet_fields = ['frequency_bandwidth', 'maximum_frequency', \
		'total_syllable_energy', 'syllable_duration', 'minimum_frequency', \
		'mean_frequency', 'peak_syllable_amplitude', 'starting_frequency', \
		'final_frequency']

	params = {
		'axes.labelsize': 8,
		'axes.labelpad': 0.05,
		'legend.fontsize': 8,
		'xtick.labelsize': 7,
		'ytick.labelsize': 7,
		'text.usetex': False,
		'figure.figsize': [5.0, 1.47]
	}
	rcParams.update(params)
	fig = plt.figure()

	q_indices = [5311, 6396, 8226, 1021, 19457, 20227, 4297, 4600, 31198, 29861, \
			22392, 30227, 15897, 8077, 4838, 23877, 28076, 2484, 3149, 8067, \
			2012][-12:]
	l_indices = [24101, 22439, 24103, 14173, 8836, 4394, 24537, 29593, 4859, \
			5191, 8113, 2011]
	f_indices = [31198, 30227, 22392, 15059, 3149, 27823, 24836, 2484, 28076, \
			8067, 3149, 7620]

	gs1 = gridspec.GridSpec(1,1)
	gs1.update(left=0.22, right=0.98, top=0.92, bottom=0.72)
	gs2 = gridspec.GridSpec(1,1)
	gs2.update(left=0.22, right=0.98, top=0.74, bottom=0.0)

	ax1 = plt.Subplot(fig, gs1[0,0])
	fig.add_subplot(ax1)
	ax2 = plt.Subplot(fig, gs2[0,0])
	fig.add_subplot(ax2)

	# # Think about spacing the top vs. bottom spacings.
	# knn_display_DC(dc, mupet_fields, ax=ax1, indices=indices, gap=(4,8), \
	# 	save_and_close=False)

	indexed_grid_plot(dc, [q_indices], gap=12, ax=ax1, save_and_close=False)
	indexed_grid_plot(dc, [l_indices, f_indices], gap=12, ax=ax2,
		save_and_close=False)

	plt.text(0.02,0.76,'Query\nspectrograms:',transform=fig.transFigure, size=7)
	plt.text(0.02,0.44,'Latent feature\nnearest neighbors:',transform=fig.transFigure, size=7)
	plt.text(0.02,0.22,'MUPET feature\nnearest neighbors:',transform=fig.transFigure, size=7)

	plt.savefig('plot_2.pdf')
	plt.close('all')




###
