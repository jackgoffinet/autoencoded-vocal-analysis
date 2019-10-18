"""
Plot 3: Comparing Vocal Repertoires

"""

from matplotlib import rcParams
import matplotlib
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import os

from ava.data.data_container import DataContainer
from ava.plotting.latent_projection import latent_projection_plot_DC
from ava.plotting.mmd_plots import mmd_matrix_DC, mmd_tsne_DC
from ava.plotting.cluster_pca_plot import cluster_pca_latent_plot_DC, \
	cluster_pca_feature_plot_DC, relative_variability_plot_DC
from ava.plotting.trace_plot import spectrogram_plot


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
	mupet_feature_dirs = [root+'C57_mupet', root+'DBA_mupet']
	ds_feature_dirs = [root+'C57_deepsqueak', root+'DBA_deepsqueak']
	spec_dirs = [root+'C57_specs', root+'DBA_specs']
	model_fn = root + 'mupet_checkpoint_150.tar'

	dc1 = DataContainer(projection_dirs=proj_dirs, \
		feature_dirs=mupet_feature_dirs, spec_dirs=spec_dirs, \
		model_filename=model_fn)

	from ava.plotting.tooltip_plot import tooltip_plot_DC
	tooltip_plot_DC(dc1, num_imgs=2000)
	quit()

	root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/syll/'
	proj_dirs = [root+'DIR_proj', root+'UNDIR_proj']
	feature_dirs = [root+'DIR_features', root+'UNDIR_features']
	spec_dirs = [root+'DIR_specs', root+'UNDIR_specs']
	model_fn = root + 'checkpoint_080.tar'

	sap_fields = ['syllable_duration_sap', 'mean_amplitude',
		'mean_pitch', 'mean_FM', 'mean_AM2', 'mean_entropy', 'mean_pitch_goodness',
		'mean_mean_freq', 'pitch_variance', 'FM_variance', 'entropy_variance',
		'pitch_goodness_variance', 'mean_freq_variance', 'AM_variance']

	dc2 = DataContainer(projection_dirs=proj_dirs, \
		feature_dirs=feature_dirs, spec_dirs=spec_dirs, \
		model_filename=model_fn)


	animals = list(range(3,11)) + list(range(12,24)) + list(range(26,31)) + \
		list(range(44,51)) + list(range(52,53)) + list(range(54,57))
	animal_names = ['BM'+str(animal).zfill(3) for animal in animals]

	root = '/media/jackg/Jacks_Animal_Sounds/mice/Tom_control/'
	audio_dirs = [root+i+'/audio/' for i in animal_names]
	seg_dirs = [root+i+'/segs/' for i in animal_names]
	proj_dirs = [root+i+'/projections/' for i in animal_names]
	spec_dirs = [root+i+'/specs/' for i in animal_names]
	feature_dirs = [root+i+'/mupet/' for i in animal_names]
	model_filename = root + 'checkpoint_060.tar'

	dc_3 = DataContainer(projection_dirs=proj_dirs, feature_dirs=feature_dirs,
		spec_dirs=spec_dirs, plots_dir=root, model_filename=model_filename)

	params = {
		'axes.labelsize': 7,
		'axes.labelpad': 0.05,
		'legend.fontsize': 7,
		'legend.title_fontsize': 7,
		'xtick.labelsize': 7,
		'ytick.labelsize': 7,
		'text.usetex': False,
		'figure.figsize': [6.5, 4]
	}
	rcParams.update(params)
	fig = plt.figure()

	gsarr = [gridspec.GridSpec(1,1) for _ in range(8)]

	# 0 1 2
	# 3 4 5
	# Projections
	gsarr[0].update(left=0.04, right=0.29, top=0.91, bottom=0.60)
	gsarr[1].update(left=0.33, right=0.59, top=0.91, bottom=0.62)
	gsarr[6].update(left=0.65, right=0.97, top=0.94, bottom=0.84)
	gsarr[2].update(left=0.65, right=0.97, top=0.82, bottom=0.59)
	gsarr[3].update(left=0.02, right=0.31, top=0.51, bottom=0.08)
	gsarr[4].update(left=0.37, right=0.59, top=0.47, bottom=0.08)
	gsarr[7].update(left=0.39, right=0.57, top=0.079, bottom=0.06)
	gsarr[5].update(left=0.65, right=0.98, top=0.49, bottom=0.06)
	# gsarr[5].update(left=0.75, right=0.88, top=0.39, bottom=0.16)



	axarr = [plt.Subplot(fig, gs[0,0]) for gs in gsarr]
	for ax in axarr:
		fig.add_subplot(ax)

	bounds = [-15,-5,-5,5]
	colors = [to_rgba(FINCH_COLOR,alpha=0.3), to_rgba(FINCH_HIGHLIGHT,alpha=0.6)]
	cluster_pca_feature_plot_DC(dc2, bounds, sap_fields, colors=colors, s=1.2, \
		ax=axarr[0], save_and_close=False)
	plt.text(0.1,0.94,'SAP Features', \
		transform=fig.transFigure, size=8)

	cluster_pca_latent_plot_DC(dc2, bounds, colors=colors, s=0.9, \
		ax=axarr[1], save_and_close=False)
	plt.text(0.39,0.94,'Latent Features', \
		transform=fig.transFigure, size=8)

	colors = [FINCH_COLOR, FINCH_HIGHLIGHT]
	edgecolor = to_rgba('k', alpha=0.0)
	patches = [Patch(color=colors[0], label="Undirected"), \
			Patch(color=colors[1], label='Directed')]
	axarr[0].legend(handles=patches, bbox_to_anchor=(1.1,-0.32), \
			loc='lower center', ncol=2, title='Social Context:', fontsize=7,
			edgecolor=edgecolor, framealpha=0.0)


	root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/'
	song_filename = root + 'songs/UNDIR/0000.wav'
	spectrogram_plot(song_filename, ax=axarr[6], save_and_close=False, x_label=False)
	#axarr[6].set_yticks([2,10])
	#axarr[6].set_xlabel('Time (ms)')
	axarr[6].axis('off')
	plt.text(0.075,1.04,'A',transform=axarr[6].transAxes, fontsize=8)
	plt.text(0.21,1.04,'B',transform=axarr[6].transAxes, fontsize=8)
	plt.text(0.37,1.04,'C',transform=axarr[6].transAxes, fontsize=8)
	plt.text(0.535,1.04,'D',transform=axarr[6].transAxes, fontsize=8)
	plt.text(0.66,1.04,'E',transform=axarr[6].transAxes, fontsize=8)
	plt.text(0.825,1.04,'F',transform=axarr[6].transAxes, fontsize=8)


	all_bounds = [ \
		[-5,1,-10,3],\
		[-15,-5,-5,5],\
		[1,10,-15,-5],\
		[-3,5,0,7],\
		[5,15,0,10],\
		[-10,-2,6,15],\
	]

	colors = [FINCH_1+(1.0,), FINCH_2+(1.0,)]
	edgecolor = to_rgba('k', alpha=0.0)
	relative_variability_plot_DC(dc2, all_bounds, sap_fields, ax=axarr[2], \
	 		colors=colors, load_data=True, save_and_close=False)


	def fn_func(fn):
		if 'C57' in fn:
			return MOUSE_1
		elif 'DBA' in fn:
			return MOUSE_2
		raise NotImplementedError



	#from ava.plotting.mmd_plots import ALL_RECORDINGS
	#def fn_func(fn):
	#	return ALL_RECORDINGS.index(int(fn.split('/')[-1].split('.')[0]))

	latent_projection_plot_DC(dc1, color_by='filename_lambda', ax=axarr[3], \
			save_and_close=False, s=0.1, alpha=0.25, condition_func=fn_func)

	edgecolor = to_rgba('k', alpha=0.0)
	colors = [MOUSE_1, MOUSE_2]
	patches = [Patch(color=colors[0], label="C57"), Patch(color=colors[1], label='DBA')]
	axarr[3].legend(handles=patches, bbox_to_anchor=(0.5,-0.25), \
			loc='lower center', ncol=2, title='Mouse Strain:', \
			edgecolor=edgecolor, framealpha=0.0, fontsize=7)

	def fn_func(fn):
		"""
		For Tom's mice.
		"""
		fn = os.path.split(fn)[-1]
		mouse_num = int(fn.split('_')[0][2:])
		session_num = fn.split('_')[1]
		if 'day' in session_num:
			session_num = int(session_num[3:])
		elif 's' in session_num:
			session_num = int(session_num[1:])
		else:
			raise NotImplementedError
		return 100*mouse_num + session_num

	fns = ['tom_mmd_matrix.npy', 'tom_mmd_conditions.npy']
	mmd_tsne_DC(dc_3, fn_func, alg='quadratic', max_n=300, \
		load_data=True, ax=axarr[5], save_and_close=False, save_load_fns=fns)
	axarr[5].axis('off')

	fns = ['mupet_mmd_matrix.npy', 'mupet_mmd_conditions.npy']
	from ava.plotting.mmd_plots import _calculate_mmd
	# _calculate_mmd(dc1, fn_func, alg='quadratic', max_n=1000, save_fns=fns)

	mmd_matrix_DC(dc1, fn_func, load_data=True, ax=axarr[4], cmap='Greys', \
			save_and_close=False, save_load_fns=fns, \
			divider_color=to_rgba(MOUSE_1), cax=axarr[7])
	x2, y2 = 16.5, 16.5
	axarr[4].hlines(y2, 16.5, 19.5, colors=to_rgba(MOUSE_1), lw=0.9)
	axarr[4].vlines(x2, 16.47, 19.5, colors=to_rgba(MOUSE_1), lw=0.9)

	labels = ['C57', 'DBA']
	axarr[4].text(0.15, 1.02, labels[0], transform=axarr[4].transAxes)
	axarr[4].text(0.65, 1.02, labels[1], transform=axarr[4].transAxes)
	axarr[4].text(-0.095, 0.66, labels[0], transform=axarr[4].transAxes, \
			rotation=90)
	axarr[4].text(-0.095, 0.16, labels[1], transform=axarr[4].transAxes, \
			rotation=90)


	plt.text(0.02,0.95,'a',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.33,0.95,'b',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.59,0.95,'c',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.02,0.47,'d',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.33,0.47,'e',transform=fig.transFigure, size=14, weight='bold')
	plt.text(0.63,0.47,'f',transform=fig.transFigure, size=14, weight='bold')


	plt.savefig('plot_3.pdf')
	plt.close('all')




###
