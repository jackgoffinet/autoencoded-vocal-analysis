"""
'Latent features capture traditional features' plot.

"""

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import os

from ava.plotting.data_container import DataContainer
from ava.plotting.feature_correlation_plots import correlation_plot_DC, \
		triptych_correlation_plot_DC, two_subplot_correlation_plot_DC
from ava.plotting.latent_projection import latent_projection_plot_DC



if __name__ == '__main__':
	root = '/media/jackg/Jacks_Animal_Sounds/mice/MUPET/'
	proj_dirs = [root+'C57_projections', root+'DBA_projections']
	mupet_feature_dirs = [root+'C57_MUPET_detect', root+'DBA_MUPET_detect']
	ds_feature_dirs = [root+'C57_DS_features', root+'DBA_DS_features']
	spec_dirs = [root+'C57_hdf5s', root+'DBA_hdf5s']

	dc1 = DataContainer(projection_dirs=proj_dirs, \
		feature_dirs=mupet_feature_dirs, spec_dirs=spec_dirs)

	dc2 = DataContainer(projection_dirs=proj_dirs, \
		feature_dirs=ds_feature_dirs, spec_dirs=spec_dirs)

	root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/syll/'
	proj_dirs = [root+'DIR_proj', root+'UNDIR_proj']
	feature_dirs = [root+'DIR_features', root+'UNDIR_features']
	spec_dirs = [root+'DIR_specs', root+'UNDIR_specs']
	model_fn = root + 'checkpoint_080.tar'

	dc3 = DataContainer(projection_dirs=proj_dirs, \
		feature_dirs=ds_feature_dirs, spec_dirs=spec_dirs, \
		model_filename=model_fn)

	import seaborn as sns
	sns.set_style("whitegrid")
	fig = plt.gcf()
	fig.set_size_inches(7.5,4)
	gs1 = gridspec.GridSpec(2, 2,
					width_ratios=[1, 1],
					height_ratios=[1, 1],
					figure=fig,
					)
	gs1.update(bottom=0.05, top=0.95, left=0.06, right=0.58, wspace=0.05, hspace=0.05)

	ax1 = plt.subplot(gs1[0,0])
	ax2 = plt.subplot(gs1[0,1])
	ax3 = plt.subplot(gs1[1,0])
	ax4 = plt.subplot(gs1[1,1])

	gs2 = gridspec.GridSpec(1, 1)
	gs2.update(bottom=0.13, top=0.95, left=0.65, right=0.98, hspace=0.05)
	sns.set_style("darkgrid")
	ax5 = plt.subplot(gs2[0,0])

	inset_fields = ['frequency_bandwidth', 'maximum_frequency', \
		'total_syllable_energy', 'syllable_duration']
	# with sns.axes_style("darkgrid"):
	# 	for ax, field in zip([ax1, ax2, ax3, ax4], inset_fields):
	# 		latent_projection_plot_DC(dc1, color_by=field, colorbar=True,
	# 			ax=ax, save_and_close=False)
	mupet_fields = inset_fields + ['minimum_frequency', 'mean_frequency', \
		'peak_syllable_amplitude', 'starting_frequency', 'final_frequency']


	ds_fields = ['call_length', 'principal_frequency', 'low_freq', 'high_freq',
		'delta_freq', 'frequency_standard_deviation', 'slope', 'sinuosity',
		'mean_power', 'tonality']

	sap_fields = ['syllable_duration_sap', 'mean_amplitude',
		'mean_pitch', 'mean_FM', 'mean_AM2', 'mean_entropy', 'mean_pitch_goodness',
		'mean_mean_freq', 'pitch_variance', 'FM_variance', 'entropy_variance',
		'pitch_goodness_variance', 'mean_freq_variance', 'AM_variance']

	# correlation_plot_DC(dc, fields, ax=ax5, save_and_close=False)
	colors = ['cyan', 'magenta', 'green']
	# triptych_correlation_plot_DC([dc1,dc3], [mupet_fields, sap_fields], \
		# colors=colors, ax=ax5, save_and_close=False)
	two_subplot_correlation_plot_DC([dc1,dc3], [mupet_fields, sap_fields], \
		colors=colors, axs=[ax1,ax2], save_and_close=False)


	# plt.tight_layout()
	plt.savefig('plot_1.pdf')
	plt.close('all')





#
