"""
'Latent features capture traditional features' plot.

"""

from matplotlib import rcParams
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import os

from ava.data.data_container import DataContainer
from ava.plotting.feature_correlation_plots import \
	_boxplot_DC, pairwise_correlation_plot_DC, feature_pca_plot_DC, \
	correlation_bar_chart_DC
from ava.plotting.latent_projection import latent_projection_plot_DC


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
	root = '/media/jackg/Jacks_Animal_Sounds/mice/MUPET/'
	proj_dirs = [root+'C57_projections', root+'DBA_projections']
	feature_dirs = [root+'C57_mupet', root+'DBA_mupet']
	spec_dirs = [root+'C57_specs', root+'DBA_specs']
	model_fn = root + 'mupet_checkpoint_150.tar'

	dc1 = DataContainer(projection_dirs=proj_dirs, \
		feature_dirs=feature_dirs, spec_dirs=spec_dirs, \
		model_filename=model_fn)

	# NOTE: EDIT THIS!
	root = '/media/jackg/Jacks_Animal_Sounds/mice/MUPET/'
	proj_dirs = [root+'C57_DS_projections', root+'DBA_DS_projections']
	feature_dirs = [root+'C57_deepsqueak', root+'DBA_deepsqueak']
	spec_dirs = [root+'C57_specs', root+'DBA_specs']
	model_fn = root + 'mupet_checkpoint_150.tar'
	dc2 = DataContainer(projection_dirs=proj_dirs, \
		feature_dirs=feature_dirs, spec_dirs=spec_dirs, \
		model_filename=model_fn)

	root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/syll/'
	proj_dirs = [root+'DIR_proj', root+'UNDIR_proj']
	feature_dirs = [root+'DIR_features', root+'UNDIR_features']
	spec_dirs = [root+'DIR_specs', root+'UNDIR_specs']
	model_fn = root + 'checkpoint_080.tar'

	dc3 = DataContainer(projection_dirs=proj_dirs, \
		feature_dirs=feature_dirs, spec_dirs=spec_dirs, \
		model_filename=model_fn)

	# Define layout.
	# import seaborn as sns
	# sns.set_style("whitegrid")

	params = {
		'axes.labelsize': 8,
		'axes.labelpad': 0.05,
		'legend.fontsize': 8,
		'xtick.labelsize': 7,
		'ytick.labelsize': 7,
		'text.usetex': False,
		'figure.figsize': [6.5, 4]
	}
	rcParams.update(params)
	fig = plt.figure()

	gsarr = [gridspec.GridSpec(1,1) for _ in range(10)]

	# 0 1  2
	# 3 4 5 6
	# Projections
	gsarr[0].update(left=0.01, right=0.26, top=1.01, bottom=0.57) # projection
	gsarr[1].update(left=0.26, right=0.51, top=1.01, bottom=0.57) # projection
	gsarr[2].update(left=0.51, right=0.76, top=1.01, bottom=0.57) # projection
	gsarr[4].update(left=.79, right=0.97, top=0.87, bottom=0.62) # matrix
	gsarr[6].update(left=.83, right=0.93, top=0.605, bottom=0.58) #b/w colorbar
	gsarr[3].update(left=0.12, right=0.44, top=0.44, bottom=0.14) # boxplot
	gsarr[5].update(left=.58, right=0.96, top=0.415, bottom=0.12) # PCA

	# Colorbar axes.
	gsarr[7].update(left=0.05, right=0.22, top=0.59, bottom=0.565)
	gsarr[8].update(left=0.30, right=0.47, top=0.59, bottom=0.565)
	gsarr[9].update(left=0.55, right=0.72, top=0.59, bottom=0.565)

	axarr = [plt.Subplot(fig, gs[0,0]) for gs in gsarr]
	# for ax in axarr:
	for ax in [axarr[5]]:
		fig.add_subplot(ax)

	# Plot.
	inset_fields = ['frequency_bandwidth', 'maximum_frequency', 'syllable_duration']

	mupet_fields = inset_fields + ['total_syllable_energy', 'minimum_frequency', 'mean_frequency', \
		'peak_syllable_amplitude', 'starting_frequency', 'final_frequency']

	ds_fields = ['call_length', 'principal_frequency', 'low_freq', 'high_freq',
		'delta_freq', 'frequency_standard_deviation', 'slope', 'sinuosity',
		'mean_power', 'tonality']

	sap_fields = ['syllable_duration_sap', 'mean_amplitude',
		'mean_pitch', 'mean_FM', 'mean_AM2', 'mean_entropy', 'mean_pitch_goodness',
		'mean_mean_freq', 'pitch_variance', 'FM_variance', 'entropy_variance',
		'pitch_goodness_variance', 'mean_freq_variance', 'AM_variance']

	# colors = ['forestgreen', 'magenta', 'firebrick']
	# 178, 223, 138
	colors = [to_rgba(i, alpha=0.9) for i in [MOUSE_1, MOUSE_2, FINCH_COLOR]]

	caxs = [axarr[i] for i in [7,8,9]]
	axs = [axarr[i] for i in [0,1,2]]
	for cax, ax, field in zip(caxs, axs, inset_fields):
		latent_projection_plot_DC(dc1, color_by=field, colorbar=True, s=0.25, \
			alpha=0.3, ax=ax, cax=cax, save_and_close=False)


	pairwise_correlation_plot_DC(dc1, mupet_fields, ax=axarr[4], cax=axarr[6], \
		save_and_close=False)

	correlation_bar_chart_DC([dc1, dc2, dc3], [mupet_fields, ds_fields, sap_fields],
		colors=colors)

	# # _boxplot_DC loads data that's written by correlation_bar_chart_DC
	# _boxplot_DC(colors, hatch='oo', ax=axarr[3], save_and_close=False)
	# patches = [Patch(facecolor='lightgray', edgecolor='k', label=r'Traditional $\rightarrow$ Latent'), \
	# 		Patch(facecolor='lightgray', edgecolor='k', hatch='ooo', label=r'Latent $\rightarrow$ Traditional')]
	# axarr[3].legend(handles=patches, bbox_to_anchor=(0.5, 0.0), fontsize=7, loc='upper center')


	feature_pca_plot_DC([dc1, dc2, dc3], [mupet_fields, ds_fields, sap_fields], \
		colors, ax=axarr[5], save_and_close=False)
	plt.savefig('temp.pdf')

	plt.text(0.04,0.935,'a',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.29,0.935,'b',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.53,0.935,'c',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.77,0.935,'d',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.04,0.46,'e',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.515,0.46,'f',transform=fig.transFigure, size=14, weight='bold')


	plt.savefig('plot_1.pdf')
	plt.close('all')


###
