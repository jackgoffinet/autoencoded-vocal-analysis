"""
Plot a latent mean projection.

"""
__author__ = "Jack Goffinet"
__date__ = "July-August 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import numpy as np

from .data_container import PRETTY_NAMES



def latent_projection_plot_DC(dc, embedding_type='latent_mean_umap', \
	color_by=None, title=None, filename='latent.pdf', colorbar=False, \
	colormap='viridis', alpha=0.5, s=0.9, ax=None, save_and_close=True, \
	show_axis=False):
	"""
	Parameters
	----------
	"""
	embedding = dc.request(embedding_type)
	if color_by is None:
		color = 'b'
	else:
		color = dc.request(color_by)
	if title is None and color_by is not None:
		title = PRETTY_NAMES[color_by]
	if dc.plots_dir is not None:
		filename = os.path.join(dc.plots_dir, filename)
	projection_plot(embedding, color=color, title=title, \
		save_filename=filename, colorbar=colorbar, colormap=colormap, \
		alpha=alpha, s=s, ax=ax, save_and_close=save_and_close, show_axis=show_axis)


def projection_plot(embedding, color='b', title="",
	save_filename='latent.pdf', colorbar=False, colormap='viridis', alpha=0.6, \
	s=0.9, ax=None, save_and_close=True, show_axis=False):
	"""

	Parameters
	----------
	embedding : numpy.ndarray
		...

	color : str or numpy.ndarray, optional
	"""
	X, Y = embedding[:,0], embedding[:,1]
	if ax is None:
		ax = plt.gca()
	# fig, ax = plt.subplots()
	cax = ax.scatter(X, Y, c=color, alpha=alpha, s=s, cmap=colormap)
	ax.set_aspect('equal')
	if title is not None and len(title) > 0:
		ax.set_xlabel(title, fontdict={'fontsize':10}) # Really a title
	if not show_axis:
		ax.get_xaxis().set_ticks([])
		ax.get_yaxis().set_ticks([])
		ax.spines['right'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
	else:
		ax.grid(True)
	if colorbar:
		min_val, max_val = np.min(color), np.max(color)
		ticks = [int(round(i)) for i in [0.8*min_val+0.2*max_val, 0.5*(min_val+max_val), 0.8*max_val+0.2*min_val]]
		fig = plt.gcf()
		cax2 = fig.add_axes([0.27, 0.8, 0.5, 0.05])
		cbar = fig.colorbar(cax, cax=cax2, fraction=0.046, orientation="horizontal", ticks=ticks) # was fig.colorbar
		cbar.solids.set_edgecolor("face")
		cbar.solids.set_rasterized(True)
		cbar.ax.set_xticklabels([str(int(round(t))) for t in ticks])
	save_dir = os.path.split(save_filename)[0]
	if save_dir != '' and not os.path.exists(save_dir):
		os.makedirs(save_dir)
	if save_and_close:
		plt.tight_layout()
		plt.savefig(save_filename)
		plt.close('all')



if __name__ == '__main__':
	pass


###
