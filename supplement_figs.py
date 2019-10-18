"""
Supplementary Figures

"""

from matplotlib import rcParams
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import os
import numpy as np

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
		model_filename=model_fn, plots_dir=root)

	#from ava.plotting.tooltip_plot import tooltip_plot_DC
	#tooltip_plot_DC(dc1)
	#quit()

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
		model_filename=model_fn, plots_dir=root)



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

	dc4 = DataContainer(projection_dirs=proj_dirs, feature_dirs=feature_dirs,
		spec_dirs=spec_dirs, plots_dir=root, model_filename=model_filename)

	#from ava.plotting.latent_projection import latent_projection_plot_DC
	#latent_projection_plot_DC(dc3, show_axis=True, filename='latent_axes.pdf')
	#quit()


	root = '/media/jackg/Jacks_Animal_Sounds/mice/BM030/'
	proj_dirs = [root+'proj']
	spec_dirs = [root+'specs']
	feature_dirs = [root+'mupet']
	model_fn = root + 'checkpoint_040.tar'
	dc5 = DataContainer(projection_dirs=proj_dirs, \
		spec_dirs=spec_dirs, model_filename=model_fn, feature_dirs=feature_dirs)


	mupet_fields = ['frequency_bandwidth', 'maximum_frequency', 'syllable_duration'] + \
		['total_syllable_energy', 'minimum_frequency', 'mean_frequency', \
		'peak_syllable_amplitude', 'starting_frequency', 'final_frequency']
	ds_fields = ['call_length', 'principal_frequency', 'low_freq', 'high_freq',
		'delta_freq', 'frequency_standard_deviation', 'slope', 'sinuosity',
		'mean_power', 'tonality']
	sap_fields = ['syllable_duration_sap', 'mean_amplitude',
		'mean_pitch', 'mean_FM', 'mean_AM2', 'mean_entropy', 'mean_pitch_goodness',
		'mean_mean_freq', 'pitch_variance', 'FM_variance', 'entropy_variance',
		'pitch_goodness_variance', 'mean_freq_variance', 'AM_variance']

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



	# # 1: All feature R2s.
	# rcParams.update({'figure.figsize':[6.5, 8]})
	# fig = plt.figure()
	# from ava.plotting.feature_correlation_plots import correlation_bar_chart_DC
	# colors = [to_rgba(i, alpha=0.9) for i in [MOUSE_1, MOUSE_2, FINCH_COLOR]]
	# correlation_bar_chart_DC([dc1, dc2, dc3], \
	# 		[mupet_fields, ds_fields, sap_fields], \
	# 		colors=colors, \
	# 		load_data=True, \
	# 		save_and_close=False)
	# patches = [Patch(color=colors[0], label="MUPET Features"), \
	# 		Patch(color=colors[1], label='DeepSqueak Features'), \
	# 		Patch(color=colors[2], label='SAP Features')]
	# edgecolor = to_rgba('k', alpha=0.0)
	# plt.gca().legend(handles=patches, bbox_to_anchor=(-0.02,0.005), \
	# 		loc='lower right', fontsize=7, edgecolor=edgecolor)
	# plt.savefig('supp_1.pdf')


	# # 2: Correlation matrices
	# rcParams.update({'figure.figsize':[3.5, 4]})
	# fig = plt.figure()
	# from ava.plotting.feature_correlation_plots import pairwise_correlation_plot_DC, \
	# 	pairwise_latent_feature_correlation_plot_DC
	# _, axarr = plt.subplots(3,2)
	# title = "Traditional Feature\nPairwise Absolute\nCorrelations"
	# pairwise_correlation_plot_DC(dc1, mupet_fields, ax=axarr[0,0], make_cbar=False, \
	# 	title=title, save_and_close=False)
	# title = "Corresponding Latent\nFeature Pairwise\nAbsolute Correlations"
	# pairwise_latent_feature_correlation_plot_DC(dc1, title=title, ax=axarr[0,1])
	#
	# pairwise_correlation_plot_DC(dc2, ds_fields, ax=axarr[1,0], make_cbar=False, \
	# 	save_and_close=False)
	# pairwise_latent_feature_correlation_plot_DC(dc2, ax=axarr[1,1])
	#
	# sap_fields.remove('pitch_variance')
	# pairwise_correlation_plot_DC(dc3, sap_fields, ax=axarr[2,0], make_cbar=False, \
	# 	save_and_close=False)
	# pairwise_latent_feature_correlation_plot_DC(dc3, ax=axarr[2,1])
	# axarr[0,0].set_ylabel('MUPET', labelpad=15)
	# axarr[1,0].set_ylabel('DeepSqueak', labelpad=15)
	# axarr[2,0].set_ylabel('SAP', labelpad=15)
	# plt.savefig('supp_2.pdf')


	# # 3: Variance fall-off.
	# rcParams.update({'figure.figsize':[6, 2]})
	#fig = plt.figure()
	#from ava.plotting.feature_correlation_plots import latent_pc_variances_plot_DC
	#_, axarr = plt.subplots(1,2)
	#latent_pc_variances_plot_DC(dc1, ax=axarr[0], color=MOUSE_COLOR)
	#latent_pc_variances_plot_DC(dc3, ax=axarr[1], color=FINCH_COLOR)
	#axarr[0].set_title("Mouse Syllables", fontsize=8)
	#axarr[0].set_ylabel('Latent Feature Variance')
	#axarr[0].set_xticks(np.arange(9))
	#axarr[0].set_xticklabels(['PC'+str(i) for i in range(1,10)])
	#axarr[1].set_title("Zebra Finch Syllables", fontsize=8)
	#axarr[1].set_ylabel('Latent Feature Variance')
	#axarr[1].set_xticks(np.arange(7))
	#axarr[1].set_xticklabels(['PC'+str(i) for i in range(1,8)])
	#plt.tight_layout()
	#plt.savefig('supp_3.pdf')


	# # 4: Representative NNs.
	#from ava.plotting.pairwise_distance_plots import representative_nn_plot_DC, knn_display_DC
	#from ava.plotting.grid_plot import indexed_grid_plot
	#plt.close('all')

	#rcParams.update({'figure.figsize':[6, 2.5]})
	#fig = plt.gcf()
	##np.random.seed(42)
	##indices = np.random.permutation(14270)[:20] # 31440
	##knn_display_DC(dc3, sap_fields, indices=indices, filename='temp.pdf')
	##quit()
	#mouse_indices = [1188, 2157, 28339, 11270, 18192, 13573, 24606,13559,26996, 11655, 10063, 29466,22266, 28063, 23234, 27885,  1815, 30001,   996, 11848]
	#mouse_l_indices = [4270, 6472, 30650, 23476, 18367, 15586, 21117, 16389, 26957, 11755, 9926, 29524, 16426, 28183, 17818, 27891, 1417, 29852, 19293, 22875]
	#m_indices = [17430, 6472, 29671, 26170, 17540, 13309, 19519, 13551, 20763, 12599, 13455, 29360, 9972, 30366, 12035, 17824, 1417, 13468, 3378, 2055]
	#ds_indices = [23000, 2158, 29015, 15511, 18267, 2974, 21117, 13224, 27063, 11843, 24848, 27944, 19536, 27192, 12143, 8822, 30377, 29882, 25243, 11228]

	#bird_indices = [9039,  2984, 10219, 10872, 10668,  2054,   542,  8910,  1935,  2995,  5914, 10546, 12097, 10184,  4194,  3978,  1039, 12811,  8470, 13994]
	#bird_l_indices = [2565, 7073, 6871, 11679, 8641, 2018, 530, 11885, 1541, 2340, 4715, 12049, 5231, 10645, 6835, 3053, 1087, 12793, 5869, 14042]
	#sap_indices = [2126, 9091, 3529, 6653, 10447, 9616, 1113, 5555, 6722, 2989, 8524, 9395, 12463, 6884, 4362, 3143, 10952, 13960, 2196, 10128]


	#gsarr = [gridspec.GridSpec(1,1) for _ in range(2)]

	#gsarr[0].update(left=0.15, right=0.93, top=0.95, bottom=0.60)
	#gsarr[1].update(left=0.15, right=0.93, top=0.5, bottom=0.05)

	#axarr = [plt.Subplot(fig, gs[0,0]) for gs in gsarr]
	#for ax in axarr:
	#	fig.add_subplot(ax)

	#indices=[bird_indices, bird_l_indices, sap_indices]
	#indexed_grid_plot(dc3, indices, ax=axarr[0], save_and_close=False, gap=(4,8))
	#plt.text(-0.01,0.8,"Query:", transform=axarr[0].transAxes, ha='right', va='center', fontsize=7)
	#plt.text(-0.01,0.46,"Latent NN:", transform=axarr[0].transAxes, ha='right', va='center', fontsize=7)
	#plt.text(-0.01,0.15,"SAP NN:", transform=axarr[0].transAxes, ha='right', va='center', fontsize=7)
	#indices=[mouse_indices, mouse_l_indices, m_indices, ds_indices]
	#indexed_grid_plot(dc1, indices,ax=axarr[1], save_and_close=False, gap=(4,8))
	#plt.text(-0.01,0.85,"Query:", transform=axarr[1].transAxes, ha='right', va='center', fontsize=7)
	#plt.text(-0.01,0.6,"Latent NN:", transform=axarr[1].transAxes, ha='right', va='center', fontsize=7)
	#plt.text(-0.01,0.37,"MUPET NN:", transform=axarr[1].transAxes, ha='right', va='center', fontsize=7)
	#plt.text(-0.01,0.13,"DeepSqueak NN:", transform=axarr[1].transAxes, ha='right', va='center', fontsize=7)
	#plt.savefig('supp_4.pdf')
	#quit()


	# 5: PC scatters for all syllables.
	#from ava.plotting.cluster_pca_plot import cluster_pca_latent_plot_DC, \
	#	cluster_pca_feature_plot_DC
	#from ava.plotting.trace_plot import spectrogram_plot
	#rcParams.update({'figure.figsize':[6.5, 4.5]})
	#fig = plt.figure()
	#gs = fig.add_gridspec(3, 6)
	#spec_ax = fig.add_subplot(gs[0, 1:-1])
	#axarr = [[fig.add_subplot(gs[1,i]) for i in range(6)], [fig.add_subplot(gs[2,i]) for i in range(6)]]

	#root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/'
	#song_filename = root + 'songs/UNDIR/0000.wav'
	#spectrogram_plot(song_filename, ax=spec_ax, save_and_close=False, x_label=False)
	#spec_ax.set_yticks([2,6,10])
	#spec_ax.set_xlabel('Time (ms)')

	#plt.text(0.08,1.04,'A',transform=spec_ax.transAxes, fontsize=7)
	#plt.text(0.21,1.04,'B',transform=spec_ax.transAxes, fontsize=7)
	#plt.text(0.37,1.04,'C',transform=spec_ax.transAxes, fontsize=7)
	#plt.text(0.54,1.04,'D',transform=spec_ax.transAxes, fontsize=7)
	#plt.text(0.66,1.04,'E',transform=spec_ax.transAxes, fontsize=7)
	#plt.text(0.83,1.04,'F',transform=spec_ax.transAxes, fontsize=7)


	#bounds = [ \
	#	[-5,1,-10,3],\
	#	[-15,-5,-5,5],\
	#	[1,10,-15,-5],\
	#	[-3,5,0,7],\
	#	[5,15,0,10],\
	#	[-10,-2,6,15],\
	#]
	#colors = [to_rgba(FINCH_COLOR,alpha=0.1), to_rgba(FINCH_HIGHLIGHT,alpha=0.6)]
	#for i in range(len(bounds)):
	#	cluster_pca_feature_plot_DC(dc3, bounds[i], sap_fields, colors=colors, \
	#		s=1.2, ax=axarr[0][i], save_and_close=False)
	#	axarr[0][i].set_aspect('equal')
	#	cluster_pca_latent_plot_DC(dc3, bounds[i], colors=colors, s=0.9, \
	#		ax=axarr[1][i], save_and_close=False)
	#	axarr[1][i].set_aspect('equal')
	#axarr[0][0].set_ylabel('SAP Feature\nPrincipal\nComponents', fontsize=7, labelpad=15)
	#axarr[1][0].set_ylabel('Latent Feature\nPrincipal\nComponents', fontsize=7, labelpad=15)

	#for i,j in zip(range(6), ['A', 'B', 'C', 'D', 'E', 'F']):
	#	axarr[1][i].set_xlabel('Syllable '+j)
	#	axarr[1][i].xaxis.set_label_coords(0.5, -0.3)
	#plt.text(0.17, 0.07, 'A', transform=fig.transFigure, fontsize=7)
	#plt.text(0.3, 0.07, 'B', transform=fig.transFigure, fontsize=7)
	#plt.text(0.42, 0.07, 'C', transform=fig.transFigure, fontsize=7)
	#plt.text(0.57, 0.07, 'D', transform=fig.transFigure, fontsize=7)
	#plt.text(0.70, 0.07, 'E', transform=fig.transFigure, fontsize=7)
	#plt.text(0.84, 0.07, 'F', transform=fig.transFigure, fontsize=7)
	#plt.savefig('supp_5.pdf')


	# 6: Atlas of USVs
	#Screenshot



	# 7: MMD matrix for Tom's mice
	from ava.plotting.mmd_plots import mmd_matrix_DC
	rcParams.update({'figure.figsize':[6.5, 3]})
	def fn_func(fn):
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
	mmd_matrix_DC(dc4, None, load_data=True, ax=plt.gca(), cmap='Greys', \
		save_and_close=False, save_load_fns=fns)
	plt.savefig('supp_7.pdf')



	# # 8: Unsupervised clustering metrics
	# from ava.plotting.rolloff_plot import clustering_performance_plot_splits
	# from ava.plotting.latent_projection import _clustered_latent_projection_DC
	# rcParams.update({'figure.figsize':[6.5, 6]})
	# fig = plt.gcf()
	# gs = fig.add_gridspec(5, 6)
	# axarr1 = [fig.add_subplot(gs[0, i]) for i in range(6)]
	# axarr2 = [fig.add_subplot(gs[1, i]) for i in range(6)]
	# ax1 = fig.add_subplot(gs[2,:])
	# ax2 = fig.add_subplot(gs[3,:])
	# ax3 = fig.add_subplot(gs[4,:])
	#
	# # gsarr[0].update(left=0.15, right=0.93, top=0.95, bottom=0.60)
	# # gsarr[1].update(left=0.15, right=0.93, top=0.5, bottom=0.05)
	#
	# k_vals = [2,4,6,8,10,12]
	# colors = [to_rgba(i, alpha=0.6) for i in [FINCH_COLOR, MOUSE_COLOR]]
	# labels = ["Zebra Finch", "Mouse"]
	# noise_boxes = [None, [-9,-4.5,-4,2]]
	# dcs = [dc3, dc5]
	#
	# # for i, k_val in enumerate(k_vals):
	# # 	# Clustering metrics.
	# # 	temp_axarr = [axarr[2,i], axarr[3,i], axarr[4,i]]
	# # 	data_fn = 'temp_data/clustering_performance_'+str(k_val)+'.npy'
	# # 	clustering_performance_plot_splits(dcs, labels, axarr=temp_axarr, \
	# # 		load_data=False, n_components=k_val, save_and_close=False, \
	# # 		colors=colors, data_fn=data_fn, axis_labels=False, legend=False)
	#
	# #
	# plot_colors = []
	# X1, Y1, X2, Y2, X3, Y3 = [], [], [], [], [], []
	# for i, k_val in enumerate(k_vals):
	# 	data_fn = 'temp_data/clustering_performance_'+str(k_val)+'.npy'
	# 	result = np.load(data_fn, allow_pickle=True).item()
	# 	for j in range(10):
	# 		for k in range(2):
	# 			X1.append(0.25*(np.random.rand()-0.5)+i)
	# 			X2.append(0.25*(np.random.rand()-0.5)+i)
	# 			X3.append(0.25*(np.random.rand()-0.5)+i)
	# 			temp1 = result[(labels[k],j)]
	# 			temp2 = result[(labels[k]+'_fake',j)]
	# 			Y1.append(temp1[0] - temp2[0])
	# 			Y2.append(temp1[1] - temp2[1])
	# 			Y3.append(temp2[2] - temp1[2])
	# 			plot_colors.append(colors[k])
	#
	# ax1.scatter(X1, Y1, c=plot_colors)
	# ax2.scatter(X2, Y2, c=plot_colors)
	# ax3.scatter(X3, Y3, c=plot_colors)
	# for ax in [ax1, ax2, ax3]:
	# 	ax.axhline(y=0.0, c='k', alpha=0.5, ls='--')
	# 	ax.spines['right'].set_visible(False)
	# 	ax.spines['top'].set_visible(False)
	# 	ax.spines['bottom'].set_visible(False)
	# ax1.set_xticks([])
	# ax2.set_xticks([])
	# ax3.set_xticks(np.arange(len(k_vals)))
	# ax3.set_xticklabels([r"$k="+str(k_val)+"$" for k_val in k_vals])
	#
	# for i, k_val in enumerate(k_vals):
	# 	# Visualize clustering.
	# 	data_fn = 'temp_data/gmm_'+str(k_val)+'.gz'
	# 	_clustered_latent_projection_DC(dc3, data_fn, "Zebra Finch", ax=axarr1[i])
	# 	_clustered_latent_projection_DC(dc5, data_fn, "Mouse", \
	# 			ax=axarr2[i], noise_box=[-9,-4.5,-4,2])
	#
	# ax1.set_ylabel(r"$\Delta$"+" Silhouette\nCoefficient", fontsize=7)
	# ax2.set_ylabel(r'$\Delta$'+' Calinski-\nHarabasz Index', fontsize=7)
	# ax3.set_ylabel(r'$-\Delta$'+' Davies-\nBouldin Index', fontsize=7)
	#
	# axarr1[0].set_ylabel("Zebra Finch\nClusters", fontsize=7, color=FINCH_COLOR, labelpad=10)
	# axarr2[0].set_ylabel("Mouse\nClusters", fontsize=7, color=MOUSE_COLOR, labelpad=10)
	#
	# plt.text(0.15, 0.05, 'Underclustered', transform=fig.transFigure, size=8)
	# plt.text(0.52, 0.05, '<---->', transform=fig.transFigure, size=8)
	# plt.text(0.78, 0.05, 'Overclustered', transform=fig.transFigure, size=8)
	#
	# plt.savefig('supp_8.pdf')
	# quit()



	# # 9: Interps to and from random mouse USVs
	# from ava.plotting.pairwise_distance_plots import random_walk_plot, bridge_plot_DC
	# rcParams.update({'figure.figsize':[6, 2]})
	# # _, axarr = plt.subplots(2,1)
	# #random_walk_plot(dc3, k=10, n=25, ax=axarr[0], save_and_close=False, gap=(8,4))
	# #random_walk_plot(dc1, k=10, n=25, ax=axarr[1], save_and_close=False, gap=(8,4))
	# bridge_plot_DC(dc3, ax=plt.gca(), save_and_close=False, gap=(8,4))
	# # bridge_plot_DC(dc1, ax=axarr[1], save_and_close=False, gap=(8,4))
	# plt.tight_layout()
	# plt.savefig('supp_9.pdf')


	# 10: VAE diagram






###
