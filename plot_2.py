"""
Plot 2: Mouse and zebra finch clusters.

"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"


from matplotlib import rcParams
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
import numpy as np
import os

from plotting.data_container import DataContainer
from plotting.latent_projection import latent_projection_plot_DC
from plotting.rolloff_plot import clustering_performance_plot_splits
from plotting.grid_plot import indexed_grid_plot



if __name__ == '__main__':
	# Zebra Finch data container.
	root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/'
	hdf5_dirs = [os.path.join(root, i) for i in ['DIR_SAP_h5', 'UNDIR_SAP_h5']]
	proj_dirs = [root + 'syll_proj_DIR', root + 'syll_proj_UNDIR']
	feature_dirs = [root + i for i in ['DIR_SAP_features', 'UNDIR_SAP_features']]
	model_filename = root + 'checkpoint_080.tar'
	dc_finch = DataContainer(projection_dirs=proj_dirs, \
		spec_dirs=hdf5_dirs, plots_dir=None, model_filename=model_filename,
		feature_dirs=feature_dirs)
	finch_latent = dc_finch.request('latent_means')

	# Mouse data container.
	root = '/media/jackg/Jacks_Animal_Sounds/mice/MUPET/'
	hdf5_dirs = [os.path.join(root, i) for i in ['C57_hdf5s', 'DBA_hdf5s']]
	proj_dirs = [root + 'C57_projections', root + 'DBA_projections']
	dc_mouse = DataContainer(spec_dirs=hdf5_dirs, projection_dirs=proj_dirs)
	mouse_latent = dc_mouse.request('latent_means')

	latents = [finch_latent, mouse_latent]
	labels = ["Zebra Finch", "Mouse"]

	# Plot.
	params = {
		'axes.labelsize': 9,
		'axes.labelpad': 0.05,
		'legend.fontsize': 10,
		'xtick.labelsize': 10,
		'ytick.labelsize': 7,
		'text.usetex': False,
		'figure.figsize': [6.5, 4] # 4, 1.8
	}
	rcParams.update(params)
	fig = plt.figure()

	gsarr = [gridspec.GridSpec(1,1) for _ in range(7)]
	axarr = [plt.Subplot(fig, gs[0,0]) for gs in gsarr]
	gsarr[0].update(left=0, right=0.4, top=0.98, bottom=0.67)
	plt.text(0.05,0.92,'a',transform=fig.transFigure, size=14, weight='bold')
	gsarr[1].update(left=0, right=0.4, top=0.64, bottom=0.40)
	plt.text(0.05,2/3,'b',transform=fig.transFigure, size=14, weight='bold')
	gsarr[2].update(left=0, right=0.4, top=1/3+0.05, bottom=0.05)
	plt.text(0.05,1/3,'c',transform=fig.transFigure, size=14, weight='bold')
	gsarr[3].update(left=0.45, right=0.55, top=0.92, bottom=0.5)
	plt.text(0.385,0.94,'d',transform=fig.transFigure, size=14, weight='bold')
	gsarr[4].update(left=0.65, right=0.75, top=0.92, bottom=0.5)
	plt.text(0.585,0.94,'e',transform=fig.transFigure, size=14, weight='bold')
	gsarr[5].update(left=0.85, right=0.95, top=0.92, bottom=0.5)
	plt.text(0.805,0.94,'f',transform=fig.transFigure, size=14, weight='bold')
	gsarr[6].update(left=0.39, right=0.98, top=0.5, bottom=-0.02)
	plt.text(0.385,0.45,'g',transform=fig.transFigure, size=14, weight='bold')





	# ax1 = plt.Subplot(fig, gs0[0,0])
	# ax2 = plt.Subplot(fig, gs0[1,0])
	# ax3 = plt.Subplot(fig, gs0[2,0])
	#
	# ax4 = plt.Subplot(fig, gs1[0,0])
	# ax5 = plt.Subplot(fig, gs1[0,1])
	# ax6 = plt.Subplot(fig, gs1[0,2])
	#
	# ax7 = plt.Subplot(fig, gs1[1,:])

	subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
	x, y = [1,2], [5,4]
	for l, ax in zip(subplot_labels, axarr):
		fig.add_subplot(ax)
		# ax.text(-0.1, 1.1, l, transform=ax.transAxes, size=14, weight='bold')
		if l not in ['d', 'e', 'f']:
			ax.axis('off')
		# ax.plot(x,y)

	latent_projection_plot_DC(dc_finch, alpha=0.15, s=0.3, ax=axarr[1], \
		save_and_close=False)

	latent_projection_plot_DC(dc_mouse, alpha=0.2, color_by='cluster', \
		ax=axarr[2], colormap='tab10', save_and_close=False)

	temp_axarr = [axarr[3], axarr[4], axarr[5]]
	clustering_performance_plot_splits(latents, labels, axarr=temp_axarr, \
		save_and_close=False)
	axarr[3].set_yticks([0.0,0.2,0.4])
	axarr[4].set_yticks([0,200,400])
	axarr[5].set_yticks([0,1,2])

	path = [\
		[4821,4815,4826,6927,3161,6464,3706,16665,16211], \
		[4821,5226,25449,26994,17912,17068,17724,18971,19138], \
		[5569,22696,17911,17988,16957,17104,19994,20163,18309], \
	]
	indexed_grid_plot(dc_mouse, path, ax=axarr[6], save_and_close=False)

	# gs0.tight_layout(fig, rect=[0, 0, 0.4, 1])
	# gs1.tight_layout(fig, rect=[0.4, 0, 1, 1])
	plt.savefig('clustering.png')
	plt.close('all')


###
