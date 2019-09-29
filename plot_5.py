"""
Sliding window/song variability multipanel.

"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"


from matplotlib import rcParams
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import numpy as np
import os

from ava.models.vae import X_SHAPE
from ava.data.data_container import DataContainer
from ava.plotting.trace_plot import warped_trace_plot_DC, warped_variability_plot_DC, \
		spectrogram_plot
from ava.plotting.latent_projection import latent_projection_plot_DC
from ava.preprocessing.preprocessing import get_spec



MOUSE_COLOR = (154/256,155/256,77/256)
MOUSE_1 = (183/256,105/256,52/256)
MOUSE_2 = (172/256,163/256,63/256)
MOUSE_HIGHLIGHT = (105/256,140/256,224/256)
FINCH_COLOR = (163/256,86/256,141/256)
FINCH_1 = (180/256,80/256,147/256)
FINCH_2 = (112/256,99/256,186/256)
FINCH_HIGHLIGHT = (101/256,228/256,203/256)


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
	model_filename = root + 'song_window/methods_checkpoint_400.tar'
	plots_dir = root + 'song_window/plots/'
	feature_dirs = None
	seg_dirs = None
	finch_dc = DataContainer(projection_dirs=proj_dirs, \
		spec_dirs=spec_dirs, plots_dir=plots_dir, model_filename=model_filename,
		feature_dirs=feature_dirs, audio_dirs=audio_dirs, template_dir=template_dir)


	root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/fixed_window/'
	audio_dirs = [root + 'undir_subset']
	roi_dirs = [root + 'segs']
	spec_dirs = [root+'h5s']
	proj_dirs = [root+'proj']
	model_filename = root + 'checkpoint_050.tar'
	plots_dir = root
	finch_window_dc = DataContainer(projection_dirs=proj_dirs, audio_dirs=audio_dirs, \
		spec_dirs=spec_dirs, plots_dir=root, model_filename=model_filename)

	# from ava.plotting.tooltip_plot import tooltip_plot_DC
	# tooltip_plot_DC(finch_window_dc)

	root = '/media/jackg/Jacks_Animal_Sounds/mice/BM030/fixed_window/'
	audio_dirs = [root + 'audio']
	roi_dirs = [root + 'segs']
	spec_dirs = [root+'h5s']
	proj_dirs = [root+'proj']
	model_filename = root + 'checkpoint_480.tar'
	plots_dir = root
	mouse_window_dc = DataContainer(projection_dirs=proj_dirs, audio_dirs=audio_dirs, \
		spec_dirs=spec_dirs, plots_dir=root, model_filename=model_filename)

	# Setup the gridspecs.
	params = {
		'axes.labelsize': 7,
		'axes.labelpad': 0.05,
		'legend.fontsize': 7,
		'xtick.labelsize': 7,
		'ytick.labelsize': 7,
		'text.usetex': False,
		'figure.figsize': [6.5, 4] # 4, 1.8
	}
	rcParams.update(params)
	fig = plt.figure()

	# gs1 = gridspec.GridSpec(2,1)
	# ax1 = fig.add_subplot(gs1[0])
	# ax2 = fig.add_subplot(gs1[1])
	gs0 = gridspec.GridSpec(1,1)
	gs1 = gridspec.GridSpec(1,1)
	gs0.update(left=0.0, right=0.45, top=0.98, bottom=0.52)
	gs1.update(left=0.0, right=0.45, top=0.48, bottom=0.02)
	ax0 = plt.Subplot(fig, gs0[0,0])
	ax1 = plt.Subplot(fig, gs1[0,0])

	gs2 = gridspec.GridSpec(3,1, height_ratios=[3, 4, 4])
	gs2.update(left=0.53, right=0.97, top=0.94, bottom=0.10)
	ax3 = fig.add_subplot(gs2[0])
	ax4 = fig.add_subplot(gs2[1], sharex=ax3)
	ax5 = fig.add_subplot(gs2[2], sharex=ax3)

	# axarr = [ax0, ax1, ax3, ax4, ax5]
	# for ax in axarr:
	for ax in [ax3, ax4, ax5]:
		fig.add_subplot(ax)


	# # Scatter.
	# print("scatter 1")
	# latent_projection_plot_DC(mouse_window_dc, default_color=to_rgba(MOUSE_COLOR), \
	# 		s=0.04, alpha=0.1, ax=ax0, save_and_close=False)
	# ax0.annotate('silence', (9,-2.5), fontsize=7)
	# ax0.annotate('shorter\nsyllables', (-2,-9.5), fontsize=7, ha='center')
	# ax0.annotate('longer\nsyllables', (-9,4), fontsize=7, ha='center')
	# ax0.set_ylim(-9.5,None)
	# # ax0.annotate('D', (1,7), fontsize=7)
	# # ax0.annotate('E', (5,3), fontsize=7)

	# print("scatter 2")
	# latent_projection_plot_DC(finch_window_dc, default_color=to_rgba(FINCH_COLOR), \
	# 		s=0.04, alpha=0.1, ax=ax1, save_and_close=False)
	#
	# ax1.annotate('A', (0,-14.5), fontsize=7)
	# ax1.annotate('B', (12.5,-13), fontsize=7)
	# ax1.annotate('C', (13,-1.5), fontsize=7)
	# ax1.annotate('D', (14,7.5), fontsize=7)
	# ax1.annotate('E', (5.8,17.5), fontsize=7)
	# ax1.annotate('F', (-2,9.5), fontsize=7)
	# ax1.annotate('silence', (-13,12.5), fontsize=7)
	# ax1.annotate('linking\nnote', (-19,4), fontsize=7, ha='center')
	# ax1.annotate('introductory\nnotes', (-8,-16.5), fontsize=7, ha='center')
	# ax1.set_xlim(-19, None)



	# Spectrogram.
	def fn_to_group(fn):
		if 'UNDIR' in fn:
			return 0
		return 1

	print("spec")
	spectrogram_plot(song_filename, ax=ax3, sign=1, save_and_close=False, \
		x_label=False)
	ax3.set_yticks([2,6,10])
	ax3.xaxis.set_ticklabels([])
	# ax3.set_title("Latent features capture short-timescale variability", fontsize=8)
	plt.text(0.08,1.04,'A',transform=ax3.transAxes, fontsize=8)
	plt.text(0.21,1.04,'B',transform=ax3.transAxes, fontsize=8)
	plt.text(0.37,1.04,'C',transform=ax3.transAxes, fontsize=8)
	plt.text(0.54,1.04,'D',transform=ax3.transAxes, fontsize=8)
	plt.text(0.66,1.04,'E',transform=ax3.transAxes, fontsize=8)
	plt.text(0.83,1.04,'F',transform=ax3.transAxes, fontsize=8)
	# Traces.
	print("traces")
	colors = [to_rgba(FINCH_COLOR, alpha=0.2), to_rgba(FINCH_HIGHLIGHT, alpha=0.1)]
	warped_trace_plot_DC(finch_dc, p, ax=ax4, load_warp=True, \
			save_and_close=False, load_traces=True, colors=colors, \
			fn_to_group=fn_to_group, unique_groups=[0,1])
	ax4.set_yticks([-5,0,5])
	ax4.xaxis.set_ticklabels([])
	# Variability.
	print("variability")
	colors = [to_rgba(FINCH_COLOR, alpha=0.85), to_rgba(FINCH_HIGHLIGHT, alpha=0.95)]

	def fn_to_group(fn):
		if 'UNDIR' in fn:
			return 0
		return 1

	warped_variability_plot_DC(finch_dc, p, ax=ax5, load_warp=True, \
			save_and_close=False, load_traces=True, colors=colors, lw=1.5, \
			fn_to_group=fn_to_group, unique_groups=[0,1], \
			labels=["Undirected","Directed"])
	for ax in [ax3, ax4, ax5]:
		ax.set_xticks([0,100,200,300,400,500,600])
		ax.xaxis.set_ticklabels(['', '', '', '', '', '', ''])
	ax5.xaxis.set_ticklabels(['0', '', '200', '', '400', '', '600'])
	ax5.set_ylim(0,None)

	box = dict(facecolor='w', edgecolor='w', pad=6, alpha=0.0)
	ax3.set_ylabel("Frequency (kHz)", bbox=box)
	ax4.set_ylabel("Principal Component 1", bbox=box)
	ax5.set_ylabel("Variability Index", bbox=box)

	plt.savefig('temp.pdf')
	quit()

	plt.text(0.02,0.96,'a',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.02,0.45,'b',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.43,0.96,'c',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.43,0.67,'d',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.43,0.35,'e',transform=fig.transFigure, size=14, weight='bold')

	fig.align_ylabels([ax3,ax4,ax5])
	plt.subplots_adjust(bottom=0.15, top=0.85)
	plt.savefig('plot_5.pdf')
	plt.close('all')




###
