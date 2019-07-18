"""
Plot a latent mean projection.

"""
__author__ = "Jack Goffinet"
__date__ = "July 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import numpy as np

from .data_container import PRETTY_NAMES



def embedding_plot_DC(dc, embedding_type='latent_mean_umap', \
	color_by=None, title=None, filename='latent.pdf', colorbar=False):
	"""
	Parameters
	----------
	"""
	embedding = dc.request(embedding_type)
	if color_by is None:
		color = 'b'
	else:
		color = dc.request(color_by)
	if title is None:
		if color_by is not None:
			title = PRETTY_NAMES[color_by]
		else:
			title = PRETTY_NAMES[embedding_type]
	filename = os.path.join(dc.plots_dir, filename)
	projection_plot(embedding, color=color, title=title, \
		save_filename=filename, colorbar=colorbar)


def projection_plot(embedding, color='b', title="",
	save_filename='latent.pdf', colorbar=False):
	"""

	Parameters
	----------
	embedding : numpy.ndarray
		...

	color : str or numpy.ndarray, optional
	"""
	X, Y = embedding[:,0], embedding[:,1]
	fig, ax = plt.subplots()
	cax = ax.scatter(X, Y, c=color, alpha=0.5, s=0.9)
	ax.set_aspect('equal')
	if len(title) > 0:
		ax.set_title(title)
	ax.axis('off')
	if colorbar:
		min_val, max_val = np.min(color), np.max(color)
		ticks = [int(round(i)) for i in [0.8*min_val+0.2*max_val, 0.5*(min_val+max_val), 0.8*max_val+0.2*min_val]]
		cbar = fig.colorbar(cax, fraction=0.046, orientation="horizontal", ticks=ticks)
		cbar.solids.set_edgecolor("face")
		cbar.solids.set_rasterized(True)
		cbar.ax.set_xticklabels([str(int(round(t))) for t in ticks])
	save_dir = os.path.split(save_filename)[0]
	if save_dir != '' and not os.path.exists(save_dir):
		os.makedirs(save_dir)
	plt.tight_layout()
	plt.savefig(save_filename)
	plt.close('all')
