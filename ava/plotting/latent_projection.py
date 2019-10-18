"""
Plot a latent mean projection.

"""
__author__ = "Jack Goffinet"
__date__ = "July-October 2019"


from numba.errors import NumbaPerformanceWarning
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import numpy as np
import joblib
import umap
import warnings

from ava.data.data_container import PRETTY_NAMES



def latent_projection_plot_DC(dc, embedding_type='latent_mean_umap', \
	color_by=None, title=None, filename='latent.pdf', colorbar=False, \
	colormap='viridis', alpha=0.5, s=0.9, ax=None, cax=None, shuffle=True, \
	save_and_close=True, show_axis=False, default_color='b', \
	condition_func=None):
	"""
	Parameters
	----------
	"""
	embedding = dc.request(embedding_type)
	if color_by is None:
		color = default_color
	elif color_by == 'filename_lambda':
		assert condition_func is not None
		fns = dc.request('audio_filenames')
		color = [condition_func(fn) for fn in fns]
	else:
		color = dc.request(color_by)
	if title is None and color_by not in [None, 'filename_lambda']:
		title = PRETTY_NAMES[color_by]
	if dc.plots_dir is not None:
		filename = os.path.join(dc.plots_dir, filename)
	projection_plot(embedding, color=color, title=title, \
		save_filename=filename, colorbar=colorbar, colormap=colormap, \
		shuffle=shuffle, alpha=alpha, s=s, ax=ax, cax=cax, \
		save_and_close=save_and_close, show_axis=show_axis)


def latent_projection_plot_with_noise_DC(dc, embedding_type='latent_mean_umap', \
	color_by=None, title=None, filename='latent.pdf', colorbar=False, \
	colormap='viridis', alpha=0.5, s=0.9, ax=None, cax=None, shuffle=True, \
	save_and_close=True, show_axis=False, default_color='b', \
	condition_func=None, noise_box=None):
	assert noise_box is not None
	embedding = dc.request(embedding_type)
	indices = []
	x1, x2, y1, y2 = noise_box
	for i in range(len(embedding)):
		if embedding[i,0] < x1 or embedding[i,0] > x2 or \
				embedding[i,1] < y1 or embedding[i,1] > y2:
			indices.append(i)
	indices = np.array(indices, dtype='int')
	try:
		default_color = np.array(default_color)[indices]
	except:
		pass
	latent = dc.request('latent_means')[indices]
	transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
		metric='euclidean', random_state=42)
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
		embedding = transform.fit_transform(latent)
	if color_by is None:
		color = default_color
	elif color_by == 'filename_lambda':
		assert condition_func is not None
		fns = dc.request('audio_filenames')[indices]
		color = [condition_func(fn) for fn in fns]
	else:
		color = dc.request(color_by)[indices]
	if title is None and color_by not in [None, 'filename_lambda']:
		title = PRETTY_NAMES[color_by]
	if dc.plots_dir is not None:
		filename = os.path.join(dc.plots_dir, filename)
	projection_plot(embedding, color=color, title=title, \
		save_filename=filename, colorbar=colorbar, colormap=colormap, shuffle=shuffle, \
		alpha=alpha, s=s, ax=ax, cax=cax, save_and_close=save_and_close, \
		show_axis=show_axis)


def projection_plot(embedding, color='b', title="",
	save_filename='latent.pdf', colorbar=False, shuffle=True, \
	colormap='viridis', alpha=0.6, s=0.9, ax=None, cax=None, \
	save_and_close=True, show_axis=False):
	"""

	Parameters
	----------
	embedding : numpy.ndarray
		...

	color : str or numpy.ndarray, optional
	"""
	X, Y = embedding[:,0], embedding[:,1]
	if shuffle:
		np.random.seed(42)
		perm = np.random.permutation(len(X))
		np.random.seed(None)
		X, Y = X[perm], Y[perm]
		try:
			color = np.array(color)[perm]
		except IndexError:
			pass
	if ax is None:
		ax = plt.gca()
	im = ax.scatter(X, Y, c=color, alpha=alpha, s=s, cmap=colormap)
	ax.set_aspect('equal')
	if title is not None and len(title) > 0:
		ax.set_xlabel(title, fontdict={'fontsize':8}) # Really a title
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
		ticks = [int(round(i)) for i in [0.8*min_val+0.2*max_val, \
			0.5*(min_val+max_val), 0.8*max_val+0.2*min_val]]
		fig = plt.gcf()
		if cax is None:
			cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
		cbar = fig.colorbar(im, cax=cax, fraction=0.046, \
			orientation="horizontal", ticks=ticks)
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


def _clustered_latent_projection_DC(dc, data_fn, label, ax=None, noise_box=None):
	"""


	"""
	clusterer = joblib.load(data_fn)[label]
	latent = dc.request('latent_means')
	labels = clusterer.predict(latent)
	cmap = matplotlib.cm.get_cmap('tab10')
	colors = cmap(labels)
	if noise_box is None:
		latent_projection_plot_DC(dc, color_by=None, alpha=0.5, s=0.9, ax=ax,
			save_and_close=False, default_color=colors)
	else:
		latent_projection_plot_with_noise_DC(dc, color_by=None, alpha=0.5, s=0.9, ax=ax,
			save_and_close=False, default_color=colors, noise_box=noise_box)



if __name__ == '__main__':
	pass


###
