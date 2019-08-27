"""
Clusterwise PCA plots.

"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from sklearn.decomposition import PCA


def cluster_pca_plot_DC(dc, bounds, filename='cluster_pca.pdf'):
	"""

	"""
	latent = dc.request('latent_means')
	embed = dc.request('latent_mean_umap')
	fns = dc.request('audio_filenames')
	indices = indices_in_bounds(embed, bounds)
	latent = latent[indices]
	fns = fns[indices]
	assert len(indices) > 2
	transform = PCA(n_components=2, copy=False, whiten=True, random_state=42)
	embed = transform.fit_transform(latent)
	colors = np.array(['b' if 'UNDIR' in str(fn) else 'darkorange' for fn in fns])
	perm = np.random.permutation(len(colors))
	embed = embed[perm]
	colors = colors[perm]
	plt.scatter(embed[:,0], embed[:,1], c=colors, s=3.5, alpha=0.5)
	plt.axis('off')
	plt.savefig(os.path.join(dc.plots_dir, filename))
	plt.close('all')


def cluster_pca_feature_plot_DC(dc, bounds, fields, filename='cluster_pca.pdf'):
	"""Same as above but with traditional features."""
	embed = dc.request('latent_mean_umap')
	fns = dc.request('audio_filenames')
	field_data = {}
	for field in fields:
		field_data[field] = dc.request(field)
	field_arr = np.zeros((len(field_data[fields[0]]), len(fields)))
	for i in range(len(fields)):
		field_arr[:,i] = field_data[fields[i]]
	indices = indices_in_bounds(embed, bounds)
	fns = fns[indices]
	field_arr = field_arr[indices]
	assert len(indices) > 2
	transform = PCA(n_components=2, copy=False, whiten=True, random_state=42)
	embed = transform.fit_transform(field_arr)
	colors = np.array(['b' if 'UNDIR' in str(fn) else 'darkorange' for fn in fns])
	perm = np.random.permutation(len(colors))
	embed = embed[perm]
	colors = colors[perm]
	plt.scatter(embed[:,0], embed[:,1], c=colors, s=3.5, alpha=0.5)
	plt.axis('off')
	plt.savefig(os.path.join(dc.plots_dir, filename))
	plt.close('all')


def indices_in_bounds(embed, bounds):
	indices = []
	for i, coord in enumerate(embed):
		if bounds[0] < coord[0] and \
				bounds[1] > coord[0] and \
				bounds[2] < coord[1] and \
				bounds[3] > coord[1]:
			indices.append(i)
	return np.array(indices, dtype='int')


if __name__ == '__main__':
	pass

###
