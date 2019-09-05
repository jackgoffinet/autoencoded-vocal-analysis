"""
Plot 2: Traditional features are highly correlated ...

"""

from matplotlib import rcParams
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import os

from ava.data.data_container import DataContainer
from ava.plotting.feature_correlation_plots import \
	pairwise_correlation_plot_DC, feature_pca_plot_DC




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

	mupet_fields = ['frequency_bandwidth', 'maximum_frequency', \
		'total_syllable_energy', 'syllable_duration', 'minimum_frequency', \
		'mean_frequency', 'peak_syllable_amplitude', 'starting_frequency', \
		'final_frequency']

	ds_fields = ['call_length', 'principal_frequency', 'low_freq', 'high_freq',
		'delta_freq', 'frequency_standard_deviation', 'slope', 'sinuosity',
		'mean_power', 'tonality']

	sap_fields = ['syllable_duration_sap', 'mean_amplitude',
		'mean_pitch', 'mean_FM', 'mean_AM2', 'mean_entropy', 'mean_pitch_goodness',
		'mean_mean_freq', 'pitch_variance', 'FM_variance', 'entropy_variance',
		'pitch_goodness_variance', 'mean_freq_variance', 'AM_variance']

	colors = ['forestgreen', 'magenta', 'firebrick']


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

	gsarr = [gridspec.GridSpec(1,1) for _ in range(3)]

	gsarr[0].update(left=0.05, right=0.46, top=1.0, bottom=0.1)
	gsarr[1].update(left=0.55, right=0.97, top=0.90, bottom=0.15)
	gsarr[2].update(left=0.10, right=0.41, top=0.12, bottom=0.07)

	axarr = [plt.Subplot(fig, gs[0,0]) for gs in gsarr]
	for ax in axarr:
		fig.add_subplot(ax)

	pairwise_correlation_plot_DC(dc1, mupet_fields, ax=axarr[0], cax=axarr[2], \
		save_and_close=False)

	feature_pca_plot_DC([dc1, dc2, dc3], [mupet_fields, ds_fields, sap_fields], \
		colors, ax=axarr[1], save_and_close=False)


	plt.savefig('plot_2.pdf')
	plt.close('all')




###
