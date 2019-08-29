"""
'Latent features capture traditional features' plot.

"""

from matplotlib import rcParams
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import os

from ava.plotting.data_container import DataContainer
from ava.plotting.feature_correlation_plots import correlation_plot_DC, \
		triptych_correlation_plot_DC, two_subplot_correlation_plot_DC
from ava.plotting.latent_projection import latent_projection_plot_DC



if __name__ == '__main__':
	# Define data.
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
	gsarr[0].update(left=0.01, right=0.28, top=1.0, bottom=0.55)
	gsarr[1].update(left=0.3, right=0.57, top=1.0, bottom=0.55)
	gsarr[2].update(left=0.65, right=0.97, top=0.94, bottom=0.585)
	gsarr[3].update(left=0.01, right=0.28, top=0.49, bottom=0.08)
	gsarr[4].update(left=0.3, right=0.57, top=0.49, bottom=0.08)
	gsarr[5].update(left=.65, right=0.97, top=0.45, bottom=0.1)
	# Colorbar axes.
	gsarr[6].update(left=0.06, right=0.23, top=0.56, bottom=0.54)
	gsarr[7].update(left=0.35, right=0.52, top=0.56, bottom=0.54)
	gsarr[8].update(left=0.06, right=0.23, top=0.08, bottom=0.06)
	gsarr[9].update(left=0.35, right=0.52, top=0.08, bottom=0.06)

	axarr = [plt.Subplot(fig, gs[0,0]) for gs in gsarr]
	for ax in axarr:
		fig.add_subplot(ax)

	# Plot.
	inset_fields = ['frequency_bandwidth', 'maximum_frequency', \
		'total_syllable_energy', 'syllable_duration']

	caxs = [axarr[i] for i in [6,7,8,9]]
	axs = [axarr[i] for i in [0,1,3,4]]
	for cax, ax, field in zip(caxs, axs, inset_fields):
		latent_projection_plot_DC(dc1, color_by=field, colorbar=True,
			ax=ax, cax=cax, save_and_close=False)

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
	colors = ['cyan', 'magenta']
	labels = ['MUPET', 'SAP']
	# triptych_correlation_plot_DC([dc1,dc3], [mupet_fields, sap_fields], \
		# colors=colors, ax=ax5, save_and_close=False)
	two_subplot_correlation_plot_DC([dc1, dc3], [mupet_fields, sap_fields], \
		colors=colors, axs=[axarr[2],axarr[5]], labels=labels, \
		save_and_close=False)

	for i in [2,5]:
		axarr[i].set_xlim(-0.02,0.51)
		axarr[i].set_ylim(-0.02,0.51)
		axarr[i].plot([0,0.5], [0,0.5], ls='--', c='k')
		axarr[i].spines['right'].set_visible(False)
		axarr[i].spines['top'].set_visible(False)

	plt.text(0.05,0.94,'a',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.3,0.94,'c',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.57,0.94,'e',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.05,0.46,'b',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.3,0.46,'d',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.57,0.46,'f',transform=fig.transFigure, size=14, weight='bold')


	# plt.tight_layout()
	plt.savefig('plot_1.pdf')
	plt.close('all')





#
