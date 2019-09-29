"""
Plot the PCs of particular clusters.

"""
__author__ = "Jack Goffinet"
__date__ = "August-September 2019"

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
plt.switch_backend('agg')
import numpy as np
import os
from scipy.stats import zscore
from sklearn.decomposition import PCA


def cluster_pca_latent_plot_DC(dc, bounds, colors=('b', 'darkorange'), \
	s=0.9, alpha=0.9, ax=None, save_and_close=True, filename='cluster_pca.pdf'):
	"""
	Plot the first two latent PCs of a given cluster.

	Parameters
	----------
	dc : ...
		...
	...

	"""
	latent = dc.request('latent_means')
	embed = dc.request('latent_mean_umap')
	fns = dc.request('audio_filenames')
	indices = _indices_in_bounds(embed, bounds)
	latent = latent[indices]
	fns = fns[indices]
	assert len(indices) > 2
	transform = PCA(n_components=2, random_state=42)
	embed = transform.fit_transform(latent)
	c = np.array([colors[0] if 'UNDIR' in str(fn) else colors[1] for fn in fns])
	perm = np.random.permutation(len(c))
	embed = embed[perm]
	c = c[perm]
	if ax is None:
		ax = plt.gca()
	ax.scatter(embed[:,0], embed[:,1], c=c, s=s, alpha=alpha)
	ax.axis('off')
	if save_and_close:
		plt.savefig(os.path.join(dc.plots_dir, filename))
		plt.close('all')


def cluster_pca_feature_plot_DC(dc, bounds, fields, colors=('b','darkorange'), \
	s=0.9, alpha=0.7, ax=None, save_and_close=True,filename='cluster_pca.pdf'):
	"""
	Plot the first two traditional feature PCs of a given cluster.

	Paramters
	---------
	dc : ...
		...
	...

	"""
	embed = dc.request('latent_mean_umap')
	fns = dc.request('audio_filenames')
	field_data = {}
	for field in fields:
		field_data[field] = dc.request(field)
	field_arr = []
	for i in range(len(fields)):
		temp = zscore(field_data[fields[i]])
		if not np.isnan(temp).any():
			field_arr.append(temp)
	field_arr = np.stack(field_arr).T
	indices = _indices_in_bounds(embed, bounds)
	fns = fns[indices]
	field_arr = field_arr[indices]
	assert len(indices) > 2
	embed = PCA(n_components=2, random_state=42).fit_transform(field_arr)
	c = np.array([colors[0] if 'UNDIR' in str(fn) else colors[1] for fn in fns])
	perm = np.random.permutation(len(c))
	embed = embed[perm]
	c = c[perm]
	if ax is None:
		ax = plt.gca()
	ax.scatter(embed[:,0], embed[:,1], c=c, s=s, alpha=alpha)
	ax.axis('off')
	if save_and_close:
		plt.savefig(os.path.join(dc.plots_dir, filename))
		plt.close('all')


def relative_variability_plot_DC(dc, bounds_list, fields, load_data=False, \
	colors=['goldenrod', 'seagreen'], ax=None, save_and_close=True, \
	filename='variability.pdf'):
	""" """
	loaded_data = False
	if load_data:
		try:
			d = np.load('temp_data/variability_bar_data.npy', allow_pickle=True).item()
			x_vals = d['x_vals']
			bar_heights = d['bar_heights']
			loaded_data = True
		except:
			pass
	if not loaded_data:
		# Collect data.
		latent = dc.request('latent_means')
		embed = dc.request('latent_mean_umap')
		filenames = dc.request('audio_filenames')
		field_data = {}
		for field in fields:
			field_data[field] = dc.request(field)
		field_array = []
		for i in range(len(fields)):
			temp = zscore(field_data[fields[i]])
			if not np.isnan(temp).any():
				field_array.append(temp)
		field_array = np.stack(field_array).T
		# Calculate relative variability for each cluster.
		bar_heights = []
		x_vals = []
		x_val = 0
		for bounds in bounds_list:
			indices = _indices_in_bounds(embed, bounds)
			fns = filenames[indices]
			field_arr = field_array[indices]
			latent_subset = latent[indices]
			undir_indices = [i for i in range(len(fns)) if 'UNDIR' in fns[i] ]
			trad_1 = _variability_index(field_arr, undir_indices)
			latent_1 = _variability_index(latent_subset, undir_indices)
			dir_indices = [i for i in range(len(fns)) if 'UNDIR' not in fns[i]]
			trad_2 = _variability_index(field_arr, dir_indices)
			latent_2 = _variability_index(latent_subset, dir_indices)
			bar_heights.append(trad_2/trad_1)
			bar_heights.append(latent_2/latent_1)
			x_vals += [x_val,x_val+1]
			x_val += 2.5
		np.save('temp_data/variability_bar_data.npy', {'x_vals': x_vals, 'bar_heights': bar_heights})
	if ax is None:
		ax = plt.gca()
	for y in [0.25,0.5,0.75,1.0]:
		ax.axhline(y=y, c='k', ls='--', lw=0.8, alpha=0.4, zorder=2)
	ax.bar(x_vals, bar_heights, color=colors*(len(x_vals)//2), zorder=3)
	ax.set_ylabel('Relative Variability Index', fontsize=7, labelpad=3)
	ax.set_xlabel('Syllable', fontsize=7, labelpad=2)
	ax.set_title('Variability Reduction of Directed Song', fontsize=8)
	ax.set_ylim(0,None)
	ax.set_yticks([0.0,0.5,1.0])
	ax.set_xticks([0.5 + 2.5*i for i in range(6)])
	ax.set_xticklabels(["A", "B", "C", "D", "E", "F"])
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	# ax.xaxis.get_major_ticks().set_visible(False)
	# ax.xaxis.set_major_formatter(plt.NullFormatter())
	patches = [Patch(color=colors[0], label="SAP Features"), \
		Patch(color=colors[1], label='Latent Features')]
	ax.legend(handles=patches, loc='lower right', fontsize=7, ncol=1, \
		framealpha=0.9, edgecolor=to_rgba('k', alpha=0.0))
	if save_and_close:
		plt.savefig(filename)
		plt.close('all')


def _indices_in_bounds(embed, bounds):
	indices = []
	for i, coord in enumerate(embed):
		if bounds[0] < coord[0] and \
				bounds[1] > coord[0] and \
				bounds[2] < coord[1] and \
				bounds[3] > coord[1]:
			indices.append(i)
	return np.array(indices, dtype='int')


# def _variability_index(arr, indices):
# 	"""Helper function."""
# 	mean_val = np.mean(arr[indices], axis=0)
# 	return np.mean(np.sum(np.power(arr[indices] - mean_val, 2), axis=1))


def _variability_index(arr, indices):
	"""Helper function."""
	# Calculate medoid.
	n = len(indices)
	matrix = np.zeros((n, n))
	for i in range(n-1):
		for j in range(i+1,n):
			if i != j:
				dist = np.sqrt(np.sum(np.power(arr[indices[i]] - arr[indices[j]],2)))
				matrix[i,j] = dist
				matrix[j,i] = dist
	dist_sums = np.sum(matrix, axis=0)
	medoid_index = np.argmin(dist_sums)
	return np.median(matrix[:,medoid_index])

if __name__ == '__main__':
	pass


###
