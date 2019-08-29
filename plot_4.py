"""
Sliding window/song variability multipanel.

"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"


from matplotlib import rcParams
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import numpy as np
import os

from ava.models.window_vae import X_SHAPE
from ava.data.data_container import DataContainer
from ava.plotting.trace_plot import warped_trace_plot_DC, warped_variability_plot_DC, \
		spectrogram_plot
from ava.plotting.latent_projection import latent_projection_plot_DC
from ava.preprocessing.preprocessing import get_spec


if __name__ == '__main__':
	p = {
		'window_length': 0.08,
		'get_spec': get_spec,
		'num_freq_bins': X_SHAPE[0],
		'num_time_bins': X_SHAPE[1],
		'min_freq': 400,
		'max_freq': 10e3,
		'nperseg': 512, # FFT
		'noverlap': 256, # FFT
		'mel': True, # Frequency spacing
		'time_stretch': False,
		'within_syll_normalize': False,
		'spec_min_val': 2.0,
		'spec_max_val': 6.5,
		'max_dur': 1e9, # Big number
	}
	# Make the DataContainers.
	root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/'
	song_filename = root + 'songs/UNDIR/0000.wav'
	audio_dirs = [os.path.join(root, i) for i in ['songs/DIR', 'songs/UNDIR']]
	template_dir = root + 'templates'
	spec_dirs = [root+'h5s']
	proj_dirs = [root+'song_window/proj/']
	model_filename = root + 'song_window/checkpoint_400.tar'
	plots_dir = root + 'song_window/plots/'
	feature_dirs = None
	seg_dirs = None
	finch_dc = DataContainer(projection_dirs=proj_dirs, \
		spec_dirs=spec_dirs, plots_dir=plots_dir, model_filename=model_filename,
		feature_dirs=feature_dirs, audio_dirs=audio_dirs, template_dir=template_dir)


	# Setup the gridspecs.
	params = {
		'axes.labelsize': 8,
		'axes.labelpad': 0.05,
		'legend.fontsize': 8,
		'xtick.labelsize': 8,
		'ytick.labelsize': 8,
		'text.usetex': False,
		'figure.figsize': [6.5, 4] # 4, 1.8
	}
	rcParams.update(params)
	fig = plt.figure()

	gs1 = gridspec.GridSpec(2,1)
	ax1 = fig.add_subplot(gs1[0])
	ax2 = fig.add_subplot(gs1[1])


	gs2 = gridspec.GridSpec(3,1, height_ratios=[3, 4, 4])
	ax3 = fig.add_subplot(gs2[0])
	ax4 = fig.add_subplot(gs2[1], sharex=ax3)
	ax5 = fig.add_subplot(gs2[2], sharex=ax3)

	gs1.update(left=0, right=0.4, top=0.95, bottom=0.05)
	plt.text(0.02,0.96,'a',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.02,0.45,'b',transform=fig.transFigure, size=14, weight='bold')

	gs2.update(left=0.5, right=0.98, top=0.95, bottom=0.05)
	plt.text(0.42,0.96,'c',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.42,0.62,'d',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.42,0.32,'e',transform=fig.transFigure, size=14, weight='bold')

	axarr = [ax1, ax2, ax3, ax4, ax5]
	for ax in axarr:
		fig.add_subplot(ax)

	# # Scatter.
	# latent_projection_plot_DC(finch_dc, ax=ax1, save_and_close=False)

	# Spectrogram.
	spectrogram_plot(song_filename, ax=ax3, save_and_close=False, x_label=False)
	ax3.set_yticks([2,6,10])
	ax3.xaxis.set_ticklabels([])
	# Traces.
	warped_trace_plot_DC(finch_dc, p, ax=ax4, load_warp=True, save_and_close=False, load_traces=True)
	ax4.xaxis.set_ticklabels([])
	# Variability.
	warped_variability_plot_DC(finch_dc, p, ax=ax5, load_warp=True, save_and_close=False, load_traces=True)
	ax5.set_xticks([0,100,200,300,400,500,600,700])
	ax5.xaxis.set_ticklabels(['0', '', '200', '', '400', '', '600', ''])

	plt.savefig('plot_4.pdf')
	plt.close('all')




###
