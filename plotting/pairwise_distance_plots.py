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

from plotting.grid_plot import grid_plot


EPSILON = 1e-12


def pairwise_distance_scatter_DC(dc, fields, n=10**4, scaling='mad', \
	filename='feature_distance.pdf'):
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
	pairwise_distance_scatter(X, y, filename=filename)



def pairwise_distance_scatter(X, y, filename='feature_distance.pdf'):
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
	plt.title("Pairwise feature distances")
	plt.xlabel("Latent feature distance")
	plt.ylabel("Traditional feature distance")
	x_val = 0.9 * np.max(X)
	plt.plot([0,x_val], [0,coeff*x_val], ls='--', c='k', alpha=0.8)
	plt.scatter(X.flatten(), y.flatten(), s=0.9, alpha=0.6)
	plt.tight_layout()
	plt.savefig(filename)
	plt.close('all')


def knn_display_DC(dc, fields, indices=None, n=5, scaling='mad', filename='knn_display.pdf'):
	"""
	Plot nearest neighbors in feature space that are distant in latent space.

	Note that this is a multiobjective problem, and it's solved here in an
	ad-hoc way.

	Parameters
	----------
	dc :

	fields : list of strings

	scaling : str or None, optional

	filename : str, optional

	"""
	# Request data.
	latent = dc.request('latent_means')
	specs = dc.request('specs')
	features = get_feature_array(dc, fields, scaling=scaling)
	# Calculate nearest neighbors.
	nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(features)
	f_distances, f_indices = nbrs.kneighbors(features)
	f_distances, f_indices = f_distances[:,1], f_indices[:,1] # Remove self-neighbors.
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
		indices = np.argsort(objective)[:n]
	# Plot spectrograms.
	query_specs = np.array([specs[i] for i in indices])
	lnn_specs = np.array([specs[l_indices[i]] for i in indices])
	fnn_specs = np.array([specs[f_indices[i]] for i in indices])
	plot_specs = np.stack([fnn_specs, lnn_specs, query_specs])
	grid_plot(plot_specs, os.path.join(dc.plots_dir, filename))


def bridge_plot(dc, from_indices, to_indices, n=8, filename='bridge_plot.pdf'):
	"""
	Find smooth paths between example spectrograms.

	Parameters
	----------
	dc :
		...
	from_indices : list of ints
		...
	to_indices : list of ints
		...
	"""
	latent = dc.request('latent_means')
	specs = dc.request('specs')
	result = []
	print(from_indices)
	print(to_indices)
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
		result.append(temp_specs)
	grid_plot(np.array(result), os.path.join(dc.plots_dir, filename))


def random_walk_plot(dc, from_indices=None, n=20, filename='spec_walk.pdf'):
	"""
	Plot spectrograms along a random walk in a latent nearest neighbors graph.
	"""
	latent = dc.request('latent_means')
	specs = dc.request('specs')
	if from_indices is None:
		from_indices = np.random.randint(len(latent), size=8)
	result = []
	print(from_indices)
	for i in range(len(from_indices)):
		compiled = np.zeros(n, dtype='int')
		compiled[0] = from_indices[i]
		for j in range(1,n):
			target = latent[compiled[j-1]]
			distances = np.array([euclidean(target, l) for l in latent])
			distances[compiled[:j]] = 1e6 # large number
			compiled[j] = np.random.choice(np.argsort(distances)[:4])
		print(compiled)
		result.append([specs[j] for j in compiled])
	grid_plot(np.array(result), os.path.join(dc.plots_dir, filename))


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
	dc

	fields

	scaling

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
