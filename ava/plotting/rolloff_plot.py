"""
Rolloff curve plots for k-means clustering.

TO DO: clean this!
"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"


from itertools import product
import os
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
from scipy.spatial import distance
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def rolloff_plot_DCS(dcs, labels, filename='rolloff.pdf'):
	"""
	Parameters
	----------
	dcs : list of
		...

	"""
	latents = [dc.request('latent_means') for dc in dcs]
	filename=os.path.join(dcs[0].plots_dir, filename)
	rolloff_plot(latents, labels, filename)


def rolloff_plot(latents, labels, best_of=4, filename='rolloff.pdf'):
	"""
	Parameters
	----------
	latents : ...
		...

	"""
	errors = {}
	for latent, label in zip(latents, labels):
		temp_errors = []
		print(label)
		for num_c in range(1,16):
			temp_temp_errors = []
			for i in range(best_of):
				clusterer = GaussianMixture(n_components=num_c, init_params='random').fit(latent)

				temp_temp_errors.append(clusterer.score(latent))
				# y_pred = KMeans(n_clusters=num_c).fit_predict(latent)
				# temp_temp_errors.append(mean_within_cluster_error(y_pred, latent))
			temp_errors.append(max(temp_temp_errors))
		errors[label] = temp_errors
	np.save('errors.npy', errors)
	errors = np.load('errors.npy', allow_pickle=True).item()
	for label in errors.keys():
		# max_error = max(errors[label])
		plt.plot(list(range(1,16)), errors[label], label=label)
	plt.legend()
	plt.savefig(filename)
	plt.close('all')


def clustering_performance_plot_splits(latents, labels, n_components=6, \
	num_fake_shuffles=10, axarr=None, save_and_close=True, load_data=False, \
	filename='clustering.pdf', colors=['b', 'darkorange']):
	"""
	Parameters
	----------

	latents_1 :

	"""
	jitter = 0.25
	latent_nums = range(len(latents))
	result = {}

	data_loaded = False
	if load_data:
		try:
			result = np.load('temp_data/clustering_performance.npy', allow_pickle=True).item()
			data_loaded = True
		except:
			print("Unable to load data!")

	if not data_loaded:
		for latent_num in latent_nums:
			latent = latents[latent_num]
			kf = KFold(n_splits=10, shuffle=True)
			fold = 0
			for train_index, test_index in kf.split(latent):
				print(fold)
				clusterer = GaussianMixture(n_components=n_components, \
					n_init=5, covariance_type='full').fit(latent[train_index])
				c_labels = clusterer.predict(latent[test_index])
				sil_score = metrics.silhouette_score(latent[test_index], c_labels, metric='euclidean')
				ch_score = metrics.calinski_harabasz_score(latent[test_index], c_labels)
				db_score = metrics.davies_bouldin_score(latent[test_index], c_labels)
				result[(labels[latent_num], fold)] = [sil_score, ch_score, db_score]
				fake_latent_train = make_fake_latent(latent[train_index])
				fake_latent = make_fake_latent(latent[test_index])
				clusterer = GaussianMixture(n_components=n_components, \
					n_init=5, covariance_type='full').fit(fake_latent_train)
				c_labels = clusterer.predict(fake_latent)
				sil_score = metrics.silhouette_score(fake_latent, c_labels, metric='euclidean')
				ch_score = metrics.calinski_harabasz_score(fake_latent, c_labels)
				db_score = metrics.davies_bouldin_score(fake_latent, c_labels)
				result[(labels[latent_num]+'_fake', fold)] = [sil_score, ch_score, db_score]
				fold += 1
		np.save('temp_data/clustering_performance.npy', result)

	if axarr is None:
		_, axarr = plt.subplots(1, 3)
	np.random.seed(42)
	for j in range(len(axarr)):
		for i, label in enumerate(labels):
			x_vals = [0.0*i + jitter*np.random.rand() for _ in range(10)]
			y_vals = np.array([result[(label,k)][j] for k in range(10)])
			y_vals -= np.array([result[(label+'_fake',k)][j] for k in range(10)])
			if j == 2:
				y_vals = -y_vals
			axarr[j].axhline(y=0, c='k', ls='--', lw=0.8)
			axarr[j].scatter(x_vals, y_vals, c=colors[i], alpha=0.8)
			axarr[j].set_xticks([], [])
	np.random.seed(None)
	# axarr[0].set_ylabel(r'$\Delta$ Silhouette Coefficient')
	axarr[0].set_ylabel("Goodness of Clustering\n"+r"($\Delta$ Silhouette Coefficient)", \
		labelpad=2)
	if len(axarr) > 1:
		axarr[1].set_ylabel(r'$\Delta$ Calinski-Harabasz Index')
		if len(axarr) > 2:
			axarr[2].set_ylabel(r'$-\Delta$ Davies-Bouldin Index')
	if labels is not None:
		edgecolor = to_rgba('k', alpha=0.0)
		patches = [Patch(color=colors[0], label=labels[0]), \
			Patch(color=colors[1], label=labels[1])]
		axarr[0].legend(handles=patches, edgecolor=edgecolor, framealpha=0.0, \
			loc='center right', ncol=1)
	sns.despine(bottom=True)
	if save_and_close:
		plt.savefig(filename)
		plt.close('all')


def make_fake_latent(latent, seed=None):
	"""Fit a normal distribution by MLE and sample."""
	loc = np.mean(latent, axis=0)
	cov = np.cov(latent, rowvar=False)
	np.random.seed(seed)
	result = np.random.multivariate_normal(loc, cov, size=len(latent))
	np.random.seed(None)
	return result


def get_noise(d=3.0, n=3000, ndim=2, seed=42):
	np.random.seed(42)
	noise = np.random.normal(loc=0.0, scale=1.0, size=(n,ndim))
	noise[:,0] *= d
	np.random.seed(None)
	return noise


def mean_within_cluster_error(labels, points):
	error = 0.0
	for label in np.unique(labels):
		indices = np.argwhere(labels == label).flatten()
		mean = np.mean(points[indices], axis=0)
		error += np.sum(np.power(points[indices] - mean, 2))
	return error / len(labels)


def make_fake_latent(latent, seed=None):
	"""Fit a normal distribution by MLE and sample."""
	loc = np.mean(latent, axis=0)
	cov = np.cov(latent, rowvar=False)
	np.random.seed(seed)
	result = np.random.multivariate_normal(loc, cov, size=len(latent))
	np.random.seed(None)
	return result


if __name__ == '__main__':
	spherical = get_noise(d=1.0)
	mixture_5 = []
	for i in range(5):
		mixture_5.append(get_noise(d=1.0, n=3000)+np.array([i*10.0,0.0]))
	mixture_5 = np.concatenate(mixture_5, axis=0)
	mixture_3 = []
	for i in range(3):
		mixture_3.append(get_noise(d=1.0, n=1000)+np.array([i*10.0,0.0]))
	mixture_3 = np.concatenate(mixture_3, axis=0)
	rolloff_plot([mixture_5, mixture_3, spherical], ['mixture_5', 'mixture_3', 'spherical'])



###
