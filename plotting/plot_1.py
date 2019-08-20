"""
'Latent features capture traditional features' plot.

"""

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import os

from data_container import DataContainer
from feature_correlation_plots import correlation_plot_DC
from latent_projection import latent_projection_plot_DC



if __name__ == '__main__':
	root = '/media/jackg/Jacks_Animal_Sounds/mice/MUPET/'
	proj_dirs = [root+'C57_projections', root+'DBA_projections']
	feature_dirs = [root+'C57_MUPET_detect', root+'DBA_MUPET_detect']
	spec_dirs = [root+'C57_hdf5s', root+'DBA_hdf5s']

	dc = DataContainer(projection_dirs=proj_dirs, feature_dirs=feature_dirs, \
		spec_dirs=spec_dirs, plots_dir=root+'plots')

	import seaborn as sns
	sns.set_style("whitegrid")
	fig = plt.gcf()
	fig.set_size_inches(7.5,4)
	gs1 = gridspec.GridSpec(2, 2,
					width_ratios=[1, 1],
					height_ratios=[1, 1],
					figure=fig,
					)
	gs1.update(bottom=0.05, top=0.99, left=0.01, right=0.62, wspace=0.05, hspace=0.05)

	ax1 = plt.subplot(gs1[0,0])
	ax2 = plt.subplot(gs1[0,1])
	ax3 = plt.subplot(gs1[1,0])
	ax4 = plt.subplot(gs1[1,1])

	gs2 = gridspec.GridSpec(1, 1)
	gs2.update(bottom=0.13, top=0.95, left=0.65, right=0.98, hspace=0.05)
	sns.set_style("darkgrid")
	ax5 = plt.subplot(gs2[0,0])

	fields = ['frequency_bandwidth', 'maximum_frequency', \
		'total_syllable_energy', 'syllable_duration']
	with sns.axes_style("darkgrid"):
		for ax, field in zip([ax1, ax2, ax3, ax4], fields):
			latent_projection_plot_DC(dc, color_by=field, colorbar=True,
				ax=ax, save_and_close=False)
	fields += ['minimum_frequency', 'mean_frequency', \
		'peak_syllable_amplitude', 'starting_frequency', 'final_frequency']

	correlation_plot_DC(dc, fields, ax=ax5, save_and_close=False)


	# plt.tight_layout()
	plt.savefig(os.path.join(dc.plots_dir,'plot_1.png'))
	plt.close('all')





#
