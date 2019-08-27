"""
Instantaneous plots.

"""
__author__ = "Jack Goffinet"
__date__ = "May 2019"

import h5py
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean

# from fastdtw import fastdtw




def get_imaging_info(loader):
	"""Get filenames and times from the loader & match them w/ spike data."""
	spike_data = np.load('09182018_data.npy').item()
	binary_spikes, real_spikes = [], []
	# For each collected song...
	count = 0
	for i in range(len(loader.dataset)):
		sample = loader.dataset.__getitem__(i)
		filename, file_time = sample['filename'], sample['file_time']
		# See if there'scorresponding imaging.
		flag = True
		for j in range(len(spike_data['filenames'])):
			if spike_data['filenames'][j] in filename and \
						abs(spike_data['times'][j] - file_time) < 0.5:
				binary_spikes.append(spike_data['binary_spikes'][j])
				real_spikes.append(spike_data['real_spikes'][j])
				flag = False
				count += 1
				break
		if flag:
			binary_spikes.append(None)
			real_spikes.append(None)
	print("portion identified:", count / len(loader.dataset))
	return {'binary':binary_spikes, 'real':real_spikes}



def apply_pca(latent_paths):
	pca = PCA(n_components=latent_paths.shape[-1])
	original_shape = latent_paths.shape
	latent_paths = pca.fit_transform(latent_paths.reshape(-1,latent_paths.shape[-1]))
	return latent_paths.reshape(original_shape)


# def fit_taus(latent_paths, ts):
# 	"""DTW time fits."""
# 	# Find tau fits.
# 	target_path = latent_paths[0] # Pick some actual path to be the target path.
# 	taus = np.zeros((len(latent_paths), len(ts)))
# 	for i in range(latent_paths.shape[0]):
# 		_, path = fastdtw(target_path, latent_paths[i], radius=2, dist=euclidean)
# 		for p in path:
# 			taus[i,p[1]] = ts[p[0]]
# 	return taus, target_path


def resample_paths(latent_paths, ts, taus):
	result = np.zeros(latent_paths.shape)
	for i in range(len(latent_paths)):
		interp = interp1d(taus[i], latent_paths[i], axis=0, fill_value='extrapolate', assume_sorted=True)
		result[i] = interp(ts)
	return result

# def fit_taus(latent_paths, ts):
# 	"""Quadratic time fits."""
# 	# Find tau fits.
# 	target_path = np.median(latent_paths, axis=0)
# 	# target_path = latent_paths[0]
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


def fit_taus(latent_paths, ts):
	"""Linear time fits."""
	# Find tau fits.
	target_path = latent_paths[0]
	target_interp = interp1d(ts, target_path, axis=0, fill_value='extrapolate', assume_sorted=True)
	coeffs = np.zeros((len(latent_paths),2))
	# ts_2 = np.power(ts, 2)
	for i in range(latent_paths.shape[0]):
		def loss(coeff):
			q_ts = coeff[1] * ts + coeff[0]
			return np.sum(np.abs(target_interp(q_ts) - latent_paths[i]))
		res = minimize(loss, np.array([0.0,1.0]))
		coeffs[i] = res.x
	taus = np.zeros((len(latent_paths), len(ts)))
	for i in range(len(taus)):
		taus[i] = coeffs[i,1] * ts + coeffs[i,0]
	return taus, target_path


def subtract_target(latent_paths, taus, target_path, ts):
	target_interp = interp1d(ts, target_path, axis=0, fill_value='extrapolate', assume_sorted=True)
	for i in range(latent_paths.shape[0]):
		for j in range(latent_paths.shape[1]):
			latent_paths[i,j] -= target_interp(taus[i,j])
	return latent_paths


# def plot_paths(latent_paths, ts):
# 	taus, target_path = fit_taus(latent_paths, ts)
#
# 	mean_path = np.mean(latent_paths, axis=0)
# 	# for i in range(len(latent_paths)):
# 		# latent_paths[i] -= mean_path
#
# 	latent_paths = subtract_target(latent_paths, taus, mean_path, ts) # subtract target/mean?
# 	latent_paths = apply_pca(latent_paths)
#
# 	_, axarr = plt.subplots(5,1, sharex=True)
# 	axarr[0].imshow(np.load('blk215_spec.npy'), origin='lower', aspect='auto', \
# 			extent=[np.min(ts), np.max(ts), 300, 12000])
# 	for j in range(4):
# 		for i in range(len(latent_paths)):
# 				axarr[j+1].plot(taus[i], latent_paths[i,:,j], c='b', lw=0.5, alpha=0.05)
# 		axarr[j+1].plot(ts, target_path[:,j], c='r', lw=1)
# 		axarr[j+1].set_ylabel('PC'+str(j+1))
# 	axarr[-1].set_xlabel('tau = f(t)')
# 	axarr[0].set_title("blk215 instantaneous song variability")
# 	plt.savefig('temp.pdf')
# 	plt.close('all')


def plot_paths_imaging(latent_paths, ts, loader, unit_num=4, filename='temp.pdf'):
	taus, target_path = fit_taus(latent_paths, ts)
	latent_paths = resample_paths(latent_paths, ts, taus)

	latent_paths -= np.mean(latent_paths, axis=0)
	# latent_paths /= np.std(latent_paths, axis=0)
	latent_paths = apply_pca(latent_paths)

	cmap = matplotlib.cm.get_cmap('viridis')

	imaging_info = get_imaging_info(loader)
	imaging_n = len(imaging_info['binary'])

	_, axarr = plt.subplots(3,1, sharex=True)
	axarr[0].imshow(np.load('blk215_spec.npy'), origin='lower', aspect='auto', \
			extent=[np.min(ts), np.max(ts), 300, 12000])
	spikes = np.array([spike for spike in imaging_info['real'] if spike is not None])
	max_spike = np.sqrt(np.max(spikes[:,unit_num]))

	no_spike_indices, spike_indices = [], []
	for i in range(imaging_n):
		if imaging_info['real'][i] is None:
			continue
		if imaging_info['real'][i][unit_num] > 0:
			spike_indices.append(i)
		else:
			no_spike_indices.append(i)
	no_spike_indices, spike_indices = np.array(no_spike_indices), np.array(spike_indices)
	if min(len(no_spike_indices), len(spike_indices)) < 10:
		return

	for j in range(2):
		# for i in range(imaging_n):
		# 	spikes = imaging_info['real'][i]
		# 	spike = 0.0 if spikes is None else np.sqrt(spikes[unit_num])/max_spike
		# 	alpha = 0.0 if spikes is None else 0.5
		# 	rgba = cmap(spike)
		# 	# c = 'r' if spikes is not None and spikes[unit_num] == 1 else 'k'
		# 	# alpha = 0.4 if c == 'r' else 0.1
		# 	axarr[j+1].plot(ts, latent_paths[i,:,j], c=rgba, lw=0.5, alpha=alpha)
		# for i in range(imaging_n, len(latent_paths)):
		# 	axarr[j+1].plot(ts, latent_paths[i,:,j], c='k', lw=0.5, alpha=0.01)

		# Quantiles
		index_set, colors = [no_spike_indices, spike_indices], ['b', 'r']
		for indices, color in zip(index_set, colors):
			temp = np.quantile(latent_paths[indices,:,j], 0.5, axis=0)
			axarr[j+1].plot(ts, temp, c=color, ls='-', alpha=0.8, lw=0.5)

			std = np.std(latent_paths[indices,:,j], axis=0, ddof=1)
			temp_1 = temp - std
			temp_2 = temp + std
			# temp_1 = np.quantile(latent_paths[indices,:,j], 0.25, axis=0)
			# temp_2 = np.quantile(latent_paths[indices,:,j], 0.75, axis=0)
			axarr[j+1].fill_between(ts, temp_1, temp_2, color=color, alpha=0.1)

		axarr[j+1].set_ylabel('PC'+str(j+1))
	axarr[-1].set_xlabel('tau = f(t)')
	axarr[0].set_title("blk215 instantaneous song variability")
	plt.savefig(filename)
	plt.close('all')



if __name__ == '__main__':
	pass



###
