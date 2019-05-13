"""
Instantaneous plots.

"""
__author__ = "Jack Goffinet"
__date__ = "May 2019"


import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw

import matplotlib.pyplot as plt
plt.switch_backend('agg')


def apply_pca(latent_paths):
	pca = PCA(n_components=8)
	original_shape = latent_paths.shape
	latent_paths = pca.fit_transform(latent_paths.reshape(-1,latent_paths.shape[-1]))
	return latent_paths.reshape(original_shape)


def fit_taus(latent_paths, ts):
	"""DTW time fits."""
	# Order by PCA dimensions.
	latent_paths = apply_pca(latent_paths)
	# Find tau fits.
	def l1(u,v):
		return np.sum(np.abs(u-v))
	target_path = np.median(latent_paths, axis=0)
	taus = np.zeros((len(latent_paths), len(ts)))
	for i in range(latent_paths.shape[0]):
		_, path = fastdtw(target_path, latent_paths[i], radius=2, dist=euclidean)
		for p in path:
			taus[i,p[1]] = ts[p[0]]
	return latent_paths, taus, target_path


# def fit_taus(latent_paths, ts):
# 	"""Quadratic time fits."""
# 	# Order by PCA dimensions.
# 	latent_paths = apply_pca(latent_paths)
# 	# Find tau fits.
# 	target_path = np.median(latent_paths, axis=0)
# 	target_interp = interp1d(ts, target_path, axis=0, fill_value='extrapolate', assume_sorted=True)
# 	coeffs = np.zeros((len(latent_paths),2))
# 	ts_2 = np.power(ts, 2)
# 	for i in range(latent_paths.shape[0]):
# 		def loss(coeff):
# 			q_ts = coeff[0] * ts_2 + coeff[1] * ts
# 			return np.sum(np.abs(target_interp(q_ts) - latent_paths[i]))
# 		res = minimize(loss, np.array([0.0,1.0]))
# 		coeffs[i] = res.x
# 	taus = np.zeros((len(latent_paths), len(ts)))
# 	for i in range(len(taus)):
# 		taus[i] = coeffs[i,0] * ts_2 + coeffs[i,1] * ts
# 	return latent_paths, taus, target_path


def subtract_target(latent_paths, taus, target_path, ts):
	target_interp = interp1d(ts, target_path, axis=0, fill_value='extrapolate', assume_sorted=True)
	for i in range(latent_paths.shape[0]):
		for j in range(latent_paths.shape[1]):
			latent_paths[i,j] -= target_interp(taus[i,j])
	return latent_paths


def plot_paths(latent_paths, ts):
	latent_paths, taus, target_path = fit_taus(latent_paths, ts)

	# latent_paths = subtract_target(latent_paths, taus, target_path, ts)
	# latent_paths = apply_pca(latent_paths)

	_, axarr = plt.subplots(3,1, sharex=True)
	for j in range(3):
		for i in range(len(latent_paths)):
				axarr[j].plot(taus[i], latent_paths[i,:,j], c='b', lw=0.5, alpha=0.2)
		axarr[j].plot(ts, target_path[:,j], c='r', lw=1)
		axarr[j].set_ylabel('PC'+str(j+1))
	axarr[2].set_xlabel('tau = f(t)')
	axarr[0].set_title("red291 instantaneous song variability")
	plt.savefig('temp.pdf')
	plt.close('all')



if __name__ == '__main__':
	pass



###
