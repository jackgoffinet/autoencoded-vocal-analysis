"""
Estimate Jensen-Shannon distances.

D_KL(P||Q) = E_P log(P/Q)
D_JS(P||Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M)
where M = 0.5 * (P + Q)

"""
__author__ = "Jack Goffinet"
__date__ = "January 2019"

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA


def helper(kde_1, kde_2, n=10**4):
	"""Helper function"""
	samples = kde_1.resample(size=n)
	log_p_1 = kde_1.logpdf(samples)
	log_p_2 = kde_2.logpdf(samples)
	log_sum = np.logaddexp(log_p_1, log_p_2)
	return np.mean(log_p_1) - np.mean(log_sum) + np.log(2.0)


def estimate_jensen_shannon_distance(cloud_1, cloud_2, num_dims=5, n=10**4, pca=None):
	assert(num_dims <= cloud_1.shape[1])
	# First take the first <num_dims> principal components.
	if pca is None:
		pca = PCA(n_components=num_dims, random_state=0)
		pca.fit(np.concatenate((cloud_1, cloud_2), axis=0))
	cloud_1 = pca.transform(cloud_1)
	cloud_2 = pca.transform(cloud_2)
	# Make KDEs.
	kde_1 = gaussian_kde(cloud_1.T, bw_method='scott')
	kde_2 = gaussian_kde(cloud_2.T, bw_method='scott')
	# Calculate KL-divergences.
	jsd = helper(kde_1, kde_2, n=n)
	jsd += helper(kde_2, kde_1, n=n)
	jsd /= 2.0
	return np.sqrt(jsd) # np.sqrt converts divergence to distance.



if __name__ == '__main__':
	d = np.load('BM005_latent_by_session.npy')
	pca = PCA(n_components=5, random_state=0)
	pca.fit(np.concatenate((d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9]), axis=0))
	results = np.zeros((10,10))
	for i in range(10):
		for j in range(i+1,10,1):
			jsd = estimate_jensen_shannon_distance(np.array(d[i]), np.array(d[j]), num_dims=5, pca=pca)
			results[i,j] = jsd
			results[j,i] = jsd
	np.save('distance_matrix.npy', results)
