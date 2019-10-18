"""
Pairwise feature distance plots.

"""
__author__ = "Jack Goffinet"
__date__ = "July 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

from ava.plotting.grid_plot import grid_plot


EPSILON = 1e-12


def pairwise_distance_scatter_DC(dc, fields, n=10**4, scaling='mad', \
	ax=None, save_and_close=True, filename='feature_distance.pdf'):
	"""
	Plot feature distance vs. latent distance.

	"""
	filename = os.path.join(dc.plots_dir, filename)
	# Request data and scale features.
	latent = dc.request('latent_means')
	features = np.zeros((len(latent), len(fields)))
	for i, field in enumerate(fields):
		feature = dc.request(field)
		if scaling is None:
			pass
		elif scaling == 'mad':
			feature -= np.median(feature)
			feature /= np.median(np.abs(feature)) + EPSILON
		elif scaling == 'z-score':
			feature -= np.mean(feature)
			feature /= np.std(feature) + EPSILON
		else:
			raise NotImplementedError
		features[:,i] = feature
	# Get random pairwise distances.
	np.random.seed(42)
	syll_pairs = np.random.randint(len(latent), size=(n,2))
	np.random.seed(None)
	latent_distances, feature_distances = np.zeros(n), np.zeros(n)
	for i in range(n):
		latent_distances[i] = \
			euclidean(latent[syll_pairs[i,0]], latent[syll_pairs[i,1]])
		feature_distances[i] = \
			euclidean(features[syll_pairs[i,0]], features[syll_pairs[i,1]])
	# Plot.
	X, y = latent_distances.reshape(-1,1), feature_distances.reshape(-1,1)
	pairwise_distance_scatter(X, y, ax=ax, save_and_close=save_and_close, \
		filename=filename)



def pairwise_distance_scatter(X, y, ax=None, save_and_close=True, \
	filename='feature_distance.pdf'):
	"""
	Plot pairwise distances.

	Parameters
	----------
	X : ...
	"""
	reg = LinearRegression(fit_intercept=False).fit(X, y)
	r2 = reg.score(X, y)
	coeff = reg.coef_.flatten()[0]
	# Plot.
	if ax is None:
		ax = plt.gca()
	x_val = 0.9 * np.max(X)
	ax.plot([0,x_val], [0,coeff*x_val], ls='--', c='k', alpha=0.8)
	ax.scatter(X.flatten(), y.flatten(), s=0.9, alpha=0.6)
	ax.set_title("Pairwise feature distances")
	ax.set_xlabel("Latent feature distance")
	ax.set_ylabel("Traditional feature distance")
	if save_and_close:
		plt.tight_layout()
		plt.savefig(filename)
		plt.close('all')


def knn_display_DC(dc, fields, indices=None, n=5, scaling='z-score', \
	gap=(2,4), ax=None, save_and_close=True, filename='knn_display.pdf'):
	"""
	Plot nearest neighbors in feature space that are distant in latent space.

	Note that this is a multiobjective problem, and it's solved here in an
	ad-hoc way.

	Parameters
	----------
	dc : ...
		...
	fields : list of str
		...
	indices : list of str
		...
	n : int
		Ignored if indices is not ``None``. Defaults to ``5``.
	scaling : {str, None}, optional
		...
	filename : str, optional
		Defaults to ``'knn_display.pdf'``.

	"""
	# Request data.
	latent = dc.request('latent_means')
	specs = dc.request('specs')
	features = get_feature_array(dc, fields, scaling=scaling)
	# Calculate nearest neighbors for traditional features.
	nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(features)
	f_distances, f_indices = nbrs.kneighbors(features)
	f_distances, f_indices = f_distances[:,1], f_indices[:,1] # Remove self-neighbors.
	# And for latent features.
	nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(latent)
	l_distances, l_indices = nbrs.kneighbors(latent)
	l_distances, l_indices = l_distances[:,1], l_indices[:,1] # Remove self-neighbors.
	# Find corresponding distances in latent space.
	if indices is None:
		latent_distances = np.zeros(len(features))
		for i in range(len(features)):
			latent_distances[i] = euclidean(latent[i], latent[f_indices[i]])
		latent_distances /= np.max(latent_distances) + EPSILON
		f_distances /= np.max(f_distances) + EPSILON
		# Find neighbors close in feature space, but distant in latent space.
		objective = f_distances - latent_distances
		i = 2
		indices = np.argsort(objective)[i*10:(i+1)*10]
	print("indices:", indices)
	print("l_indices", [l_indices[i] for i in indices])
	print("f_indices", [f_indices[i] for i in indices])
	# Plot spectrograms.
	query_specs = np.array([specs[i] for i in indices])
	lnn_specs = np.array([specs[l_indices[i]] for i in indices])
	fnn_specs = np.array([specs[f_indices[i]] for i in indices])
	plot_specs = np.stack([query_specs, lnn_specs, fnn_specs])
	grid_plot(plot_specs, gap=gap, ax=ax, save_and_close=save_and_close, \
		filename=filename)


def representative_nn_plot_DC(dc, fields, ax=None, n=20, seed=42):
	"""


	"""
	latent = dc.request('latent_means')
	print("latent", latent.shape)
	specs = dc.request('specs')
	features = get_feature_array(dc, fields, scaling='z-score')
	nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(features)
	f_distances, f_indices = nbrs.kneighbors(features)
	f_distances, f_indices = f_distances[:,1], f_indices[:,1] # Remove self-neighbors.
	# And for latent features.
	nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(latent)
	l_distances, l_indices = nbrs.kneighbors(latent)
	l_distances, l_indices = l_distances[:,1], l_indices[:,1] # Remove self-neighbors.
	np.random.seed(seed)
	indices = np.random.permutation(len(latent))
	np.random.seed(None)
	result = [[], [], []]
	collected = 0
	i = 0
	while collected < n:
		if l_indices[indices[i]] != f_indices[indices[i]]:
			result[0].append(specs[indices[i]])
			result[1].append(specs[l_indices[indices[i]]])
			result[2].append(specs[f_indices[indices[i]]])
			collected += 1
		i += 1
	if ax is None:
		ax = plt.gca()
	grid_plot(np.stack(result), ax=ax, gap=(8,4), save_and_close=False)



def bridge_plot_DC(dc, from_indices=None, to_indices=None, size=5, n=20, ax=None, save_and_close=True, \
	gap=(8,4), filename='bridge_plot.pdf'):
	"""
	Find smooth paths between example spectrograms.

	Parameters
	----------
	dc : ava.data.data_container.DataContainer
		...
	from_indices : list of ints
		...
	to_indices : list of ints
		...
	"""
	latent = dc.request('latent_means')
	specs = dc.request('specs')
	result = []
	if from_indices is None:
		from_indices = np.random.randint(len(latent), size=size)
		to_indices = np.random.randint(len(latent), size=size)

	for i in range(len(from_indices)):
		latent_1 = latent[from_indices[i]]
		latent_2 = latent[to_indices[i]]
		temp_specs = [specs[from_indices[i]]]
		for p in np.linspace(0,1,n,endpoint=True)[1:-1]:
			target_latent = latent_2 * p + latent_1 * (1-p)
			index = np.argmin(np.array([euclidean(target_latent, l) for l in latent]))
			temp_specs.append(specs[index])
			print(index)
		temp_specs.append(specs[to_indices[i]])
		print()
		result.append(np.array(temp_specs))
	result = np.array(result)
	if ax is None:
		ax=plt.gca()
	grid_plot(result, gap=gap, ax=ax, save_and_close=save_and_close, filename=os.path.join(dc.plots_dir, filename))


def random_walk_plot(dc, from_indices=None, size=5, k=5, n=20, ax=None, save_and_close=True, \
	gap=4, filename='spec_walk.pdf'):
	"""
	Plot spectrograms along a random walk in a latent nearest neighbors graph.
	"""
	latent = dc.request('latent_means')
	specs = dc.request('specs')
	if from_indices is None:
		np.random.seed(42)
		from_indices = np.random.randint(len(latent), size=size)
		np.random.seed(None)
	result = []
	print(from_indices)
	for i in range(len(from_indices)):
		compiled = np.zeros(n, dtype='int')
		compiled[0] = from_indices[i]
		for j in range(1,n):
			target = latent[compiled[j-1]]
			distances = np.array([euclidean(target, l) for l in latent])
			distances[compiled[:j]] = 1e6 # large number
			compiled[j] = np.random.choice(np.argsort(distances)[:k])
		print(compiled)
		result.append(np.array([specs[j] for j in compiled]))
	if ax is None:
		ax = plt.gca()
	grid_plot(np.array(result), ax=ax, save_and_close=save_and_close, \
		gap=gap, filename=os.path.join(dc.plots_dir, filename))


def indexed_grid_plot(dc, indices, filename='grid.pdf'):
	"""
	TO DO: use this to access grid_plot.
	"""
	specs = dc.request('specs')
	result = []
	for row in indices:
		result.append([specs[j] for j in row])
	grid_plot(np.array(result), os.path.join(dc.plots_dir, filename))


def plot_paths_on_projection(dc, indices, filename='paths_scatter.pdf'):
	"""
	Parameters
	----------
	"""
	embed = dc.request('latent_mean_umap')
	plt.scatter(embed[:,0], embed[:,1], s=0.9, alpha=0.6)
	plt.axis('off')
	for j,path in enumerate(indices):
		# if j == 0:
			# plt.plot(embed[np.array(path)[:4],0], embed[np.array(path)[:4],1], c='r')
		# else:
		plt.plot(embed[np.array(path),0], embed[np.array(path),1], c='r')
		# plt.scatter(embed[np.array(path),0], embed[np.array(path),1], c='k')
		# for i in range(len(path)):
			# plt.text(embed[path[i]][0], embed[path[i]][1], str(i)+','+str(j), fontsize=5)
	plt.savefig(os.path.join(dc.plots_dir, filename))
	plt.close('all')


def get_feature_array(dc, fields, scaling='mad'):
	"""

	Parameters
	----------
	dc: ..
		...
	fields: ...
		...
	scaling: str, optional
		...

	Returns
	-------
	features : numpy.ndarray
		Scaled features.

	NOTE: move to data_container.py?
	"""
	for i, field in enumerate(fields):
		feature = dc.request(field)
		if i == 0:
			features = np.zeros((len(feature), len(fields)))
		if scaling is None:
			pass
		elif scaling == 'mad':
			feature -= np.median(feature)
			feature /= np.median(np.abs(feature)) + EPSILON
		elif scaling == 'z-score':
			feature -= np.mean(feature)
			feature /= np.std(feature) + EPSILON
		else:
			raise NotImplementedError
		features[:,i] = feature
	return features


if __name__ == '__main__':
	pass


###
