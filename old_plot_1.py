"""
'Latent features capture traditional features' plot.

"""

from matplotlib import rcParams
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import os

from ava.data.data_container import DataContainer
from ava.plotting.feature_correlation_plots import correlation_bar_chart_DC
from ava.plotting.latent_projection import latent_projection_plot_DC



if __name__ == '__main__':
	# Define data.
	root = '/media/jackg/Jacks_Animal_Sounds/mice/MUPET/'
	proj_dirs = [root+'C57_projections', root+'DBA_projections']
	mupet_feature_dirs = [root+'C57_mupet', root+'DBA_mupet']
	ds_feature_dirs = [root+'C57_deepsqueak', root+'DBA_deepsqueak']
	spec_dirs = [root+'C57_specs', root+'DBA_specs']
	model_fn = root + 'checkpoint_150.tar'


	dc1 = DataContainer(projection_dirs=proj_dirs, \
		feature_dirs=mupet_feature_dirs, spec_dirs=spec_dirs, \
		model_filename=model_fn)

	dc2 = DataContainer(projection_dirs=proj_dirs, \
		feature_dirs=ds_feature_dirs, spec_dirs=spec_dirs, \
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

	# 0 1 2
	# 3 4 5
	# Projections
	gsarr[0].update(left=0.01, right=0.27, top=1.0, bottom=0.55)
	gsarr[1].update(left=0.27, right=0.53, top=1.0, bottom=0.55)
	gsarr[2].update(left=0.54, right=0.73, top=0.94, bottom=0.13)
	gsarr[3].update(left=0.01, right=0.27, top=0.49, bottom=0.08)
	gsarr[4].update(left=0.27, right=0.53, top=0.49, bottom=0.08)
	gsarr[5].update(left=.78, right=0.97, top=0.94, bottom=0.13)
	# Colorbar axes.
	gsarr[6].update(left=0.06, right=0.23, top=0.56, bottom=0.54)
	gsarr[7].update(left=0.31, right=0.48, top=0.56, bottom=0.54)
	gsarr[8].update(left=0.06, right=0.23, top=0.08, bottom=0.06)
	gsarr[9].update(left=0.31, right=0.48, top=0.08, bottom=0.06)

	axarr = [plt.Subplot(fig, gs[0,0]) for gs in gsarr]
	for ax in axarr:
		fig.add_subplot(ax)

	# Plot.
	inset_fields = ['frequency_bandwidth', 'maximum_frequency', \
		'total_syllable_energy', 'syllable_duration']

	caxs = [axarr[i] for i in [6,7,8,9]]
	axs = [axarr[i] for i in [0,1,3,4]]
	for cax, ax, field in zip(caxs, axs, inset_fields):
		latent_projection_plot_DC(dc1, color_by=field, colorbar=True, s=0.25, \
			alpha=0.3, ax=ax, cax=cax, save_and_close=False)

	mupet_fields = inset_fields + ['minimum_frequency', 'mean_frequency', \
		'peak_syllable_amplitude', 'starting_frequency', 'final_frequency']

	ds_fields = ['call_length', 'principal_frequency', 'low_freq', 'high_freq',
		'delta_freq', 'frequency_standard_deviation', 'slope', 'sinuosity',
		'mean_power', 'tonality']

	sap_fields = ['syllable_duration_sap', 'mean_amplitude',
		'mean_pitch', 'mean_FM', 'mean_AM2', 'mean_entropy', 'mean_pitch_goodness',
		'mean_mean_freq', 'pitch_variance', 'FM_variance', 'entropy_variance',
		'pitch_goodness_variance', 'mean_freq_variance', 'AM_variance']

	colors = ['forestgreen', 'magenta', 'firebrick']

	correlation_bar_chart_DC([dc1, dc2, dc3], [mupet_fields, ds_fields, sap_fields], \
		axs=[axarr[2],axarr[5]], colors=colors, top_n=10, bottom_n=5, \
		load_data=False, save_and_close=False) # top_n=8, bottom_n=5

	plt.text(0.05,0.94,'a',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.29,0.94,'b',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.55,0.94,'e',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.05,0.46,'c',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.29,0.46,'d',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.75,0.94,'f',transform=fig.transFigure, size=14, weight='bold')

	# plt.tight_layout()
	plt.savefig('plot_1.pdf')
	plt.close('all')





#
