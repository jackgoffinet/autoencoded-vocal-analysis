"""
Rolloff curve plots for k-means clustering.


"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"


from itertools import product
import os
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

	"""
	latents = [dc.request('latent_means') for dc in dcs]
	filename=os.path.join(dcs[0].plots_dir, filename)
	rolloff_plot(latents, labels, filename)


def rolloff_plot(latents, labels, best_of=4, filename='rolloff.pdf'):
	"""
	Paramters
	---------
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
	num_fake_shuffles=10, axarr=None, save_and_close=True, filename='clustering.pdf'):
	"""
	Parameters
	----------

	latents_1 :

	"""
	jitter = 0.25
	latent_nums = range(len(latents))
	result = {}

	# for latent_num in latent_nums:
	# 	latent = latents[latent_num]
	# 	kf = KFold(n_splits=10, shuffle=True)
	# 	fold = 0
	# 	for train_index, test_index in kf.split(latent):
	# 		print(fold)
	# 		clusterer = GaussianMixture(n_components=n_components, \
	# 			n_init=5, covariance_type='full').fit(latent[train_index])
	# 		c_labels = clusterer.predict(latent[test_index])
	# 		sil_score = metrics.silhouette_score(latent[test_index], c_labels, metric='euclidean')
	# 		ch_score = metrics.calinski_harabasz_score(latent[test_index], c_labels)
	# 		db_score = metrics.davies_bouldin_score(latent[test_index], c_labels)
	# 		result[(labels[latent_num], fold)] = [sil_score, ch_score, db_score]
	# 		fake_latent_train = make_fake_latent(latent[train_index])
	# 		fake_latent = make_fake_latent(latent[test_index])
	# 		clusterer = GaussianMixture(n_components=n_components, \
	# 			n_init=5, covariance_type='full').fit(fake_latent_train)
	# 		c_labels = clusterer.predict(fake_latent)
	# 		sil_score = metrics.silhouette_score(fake_latent, c_labels, metric='euclidean')
	# 		ch_score = metrics.calinski_harabasz_score(fake_latent, c_labels)
	# 		db_score = metrics.davies_bouldin_score(fake_latent, c_labels)
	# 		result[(labels[latent_num]+'_fake', fold)] = [sil_score, ch_score, db_score]
	# 		fold += 1
	# np.save('result.npy', result)

	result = np.load('result.npy', allow_pickle=True).item()
	if axarr is None:
		_, axarr = plt.subplots(1, 3)
	np.random.seed(42)
	for j in range(3):
		for i, label in enumerate(labels):
			x_vals = [0.6*i + jitter*np.random.rand() for _ in range(10)]
			y_vals = np.array([result[(label,k)][j] for k in range(10)])
			y_vals -= np.array([result[(label+'_fake',k)][j] for k in range(10)])
			if j == 2:
				y_vals = -y_vals
			axarr[j].axhline(y=0, c='k', ls='--', lw=0.8)
			axarr[j].scatter(x_vals, y_vals, alpha=0.8)
			axarr[j].set_xticks([], [])
	np.random.seed(None)
	axarr[0].set_ylabel(r'$\Delta$ Silhouette Coefficient')
	axarr[1].set_ylabel(r'$\Delta$ Calinski-Harabasz Index')
	axarr[2].set_ylabel(r'$-\Delta$ Davies-Bouldin Index')
	sns.despine(bottom=True)
	if save_and_close:
		plt.savefig(filename)
		plt.close('all')


def clustering_performance_plot_zscore(latents, labels, n_components=6, \
	num_fake_shuffles=10, filename='clustering.pdf'):
	"""
	Parameters
	----------

	latents_1 :

	"""
	jitter = 0.25
	latent_nums = range(len(latents))
	result = {}

	# for latent_num in latent_nums:
	# 	latent = latents[latent_num]
	# 	kf = KFold(n_splits=5)
	# 	fold = 0
	# 	for train_index, test_index in kf.split(latent):
	# 		print(fold)
	# 		clusterer = GaussianMixture(n_components=n_components, \
	# 			n_init=5, covariance_type='full').fit(latent[train_index])
	# 		c_labels = clusterer.predict(latent[test_index])
	# 		sil_score = metrics.silhouette_score(latent[test_index], c_labels, metric='euclidean')
	# 		ch_score = metrics.calinski_harabasz_score(latent[test_index], c_labels)
	# 		db_score = metrics.davies_bouldin_score(latent[test_index], c_labels)
	# 		result[(labels[latent_num], fold)] = [sil_score, ch_score, db_score]
	# 		# fake_latent = make_fake_latent(latent[train_index])
	# 		fold += 1
	# 	fold = 0
	# 	for i in range(num_fake_shuffles):
	# 		print(i)
	# 		fake_latent = make_fake_latent(latent)
	# 		kf = KFold(n_splits=5)
	# 		for train_index, test_index in kf.split(fake_latent):
	# 			clusterer = GaussianMixture(n_components=n_components, \
	# 				n_init=5, covariance_type='full').fit(fake_latent[train_index])
	# 			c_labels = clusterer.predict(fake_latent[test_index])
	# 			sil_score = metrics.silhouette_score(fake_latent[test_index], c_labels, metric='euclidean')
	# 			ch_score = metrics.calinski_harabasz_score(fake_latent[test_index], c_labels)
	# 			db_score = metrics.davies_bouldin_score(fake_latent[test_index], c_labels)
	# 			result[(labels[latent_num]+'_fake', fold)] = [sil_score, ch_score, db_score]
	# 			break
	# 		fold += 1
	# np.save('result.npy', result)
	# quit()
	result = np.load('result.npy', allow_pickle=True).item()
	f, axarr = plt.subplots(1, 3)
	for j in range(3):
		for i, label in enumerate(labels):
			y_vals_1 = np.array([result[(label,k)][j] for k in range(5)])
			x_vals_1 = [i + jitter*(np.random.rand()-0.5) for _ in y_vals_1]
			y_vals_2 = np.array([result[(label+'_fake',k)][j] for k in range(num_fake_shuffles)])
			mean = np.mean(y_vals_2)
			std = np.std(y_vals_2, ddof=1)
			if j == 2:
				std = -std
			y_vals_1 -= mean
			y_vals_1 /= std
			axarr[j].scatter(x_vals_1, y_vals_1)
			axarr[j].axhline(y=0)
	plt.savefig(filename)
	plt.close('all')


def clustering_performance_plot(latents, labels, n_components=6, \
	filename='clustering.pdf'):
	"""
	Parameters
	----------

	latents_1 :

	"""
	jitter = 0.25
	latent_nums = range(len(latents))
	result = {}

	for latent_num in latent_nums:
		latent = latents[latent_num]
		kf = KFold(n_splits=5)
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
			# fake_latent = make_fake_latent(latent[train_index])
			fold += 1
		fold = 0
		for i in range(10):
			print(i)
			fake_latent = make_fake_latent(latent)
			kf = KFold(n_splits=5)
			for train_index, test_index in kf.split(fake_latent):
				clusterer = GaussianMixture(n_components=n_components, \
					n_init=5, covariance_type='full').fit(fake_latent[train_index])
				c_labels = clusterer.predict(fake_latent[test_index])
				sil_score = metrics.silhouette_score(fake_latent[test_index], c_labels, metric='euclidean')
				ch_score = metrics.calinski_harabasz_score(fake_latent[test_index], c_labels)
				db_score = metrics.davies_bouldin_score(fake_latent[test_index], c_labels)
				result[(labels[latent_num]+'_fake', fold)] = [sil_score, ch_score, db_score]
				break
			fold += 1
	np.save('result.npy', result)
	# quit()
	result = np.load('result.npy', allow_pickle=True).item()
	f, axarr = plt.subplots(1, 3)
	for j in range(3):
		for i, label in enumerate(labels):
			y_vals_1 = [result[(label,k)][j] for k in range(5)]
			x_vals_1 = [2*i + jitter*(np.random.rand()-0.5) for _ in y_vals_1]
			axarr[j].scatter(x_vals_1, y_vals_1)
			y_vals_2 = [result[(label+'_fake',k)][j] for k in range(5)]
			x_vals_2 = [2*i+1 + jitter*(np.random.rand()-0.5) for _ in y_vals_2]
			axarr[j].scatter(x_vals_2, y_vals_2)
			for k in range(5):
				plt.plot([x_vals_1[k], x_vals_2[k]], [y_vals_1[k], y_vals_2[k]])
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


def compute_bic(kmeans,X):
	"""
	Copied from:
	https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans

	Computes the BIC metric for a given clusters

	Parameters:
	-----------------------------------------
	kmeans: List of clustering object from scikit learn

	X :  multidimension np array of data points

	Returns:
	-----------------------------------------
	BIC value
	"""
	# assign centers and labels
	centers = [kmeans.cluster_centers_]
	labels  = kmeans.labels_
	#number of clusters
	m = kmeans.n_clusters
	# size of the clusters
	n = np.bincount(labels)
	#size of data set
	N, d = X.shape
	#compute variance for all clusters beforehand
	cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]],
			 'euclidean')**2) for i in range(m)])

	const_term = 0.5 * m * np.log(N) * (d+1)

	BIC = np.sum([n[i] * np.log(n[i]) -
		n[i] * np.log(N) -
		((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
		((n[i] - 1) * d/ 2) for i in range(m)]) - const_term
	return(BIC)


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
