"""
Plot traces for sliding window analysis.


TO DO: Put a spectrogram up top, also some measure of variability
"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.optimize import minimize
from scipy.signal import stft
from sklearn.decomposition import PCA
import torch

from models.window_vae import X_SHAPE, VAE



def trace_plot_DC(dc, p, song_filename, num_segs=200, \
	filename='inst_traces.pdf'):
	"""
	Parameters
	----------
	dc

	get_spec

	p
	"""
	# Collect audio chunks.
	segment_audio = dc.request('segment_audio')
	audio_chunks = []
	audio_filenames = []
	k = 0
	for audio_dir in dc.audio_dirs:
		for audio_fn in segment_audio[audio_dir]:
			temp = segment_audio[audio_dir][audio_fn]
			audio_chunks += temp
			k += 1
			audio_filenames += [os.path.join(audio_dir, audio_fn)] * len(temp)
	# # Find samplerate.
	# fs, _ = wavfile.read(audio_filenames[0])
	# # Set up the model.
	# model = VAE()
	# model.load_state(dc.model_filename)
	# # Collect latent means.
	# latent_paths = np.zeros((len(audio_chunks), num_segs, model.z_dim))
	# for i, audio_chunk in enumerate(audio_chunks):
	# 	arr = np.zeros((num_segs, X_SHAPE[0], X_SHAPE[1]))
	# 	chunk_dur = len(audio_chunk) / fs
	# 	start_ts = np.linspace(0,chunk_dur-p['window_length'], num_segs)
	# 	for j, start_t in enumerate(start_ts):
	# 		end_t = start_t + p['window_length']
	# 		arr[j] = p['get_spec'](start_t, end_t, audio_chunk, p, fs=fs)[0]
	# 	arr = torch.from_numpy(arr).type(torch.FloatTensor).to(model.device)
	# 	arr_latent = model.encode(arr)[0].detach().cpu().numpy()
	# 	latent_paths[i] = arr_latent
	# # Align the time series.
	# print(latent_paths.shape)
	# np.save('latent_paths.npy', latent_paths)
	latent_paths = np.load('latent_paths.npy')

	# latent_paths = subtract_mean(latent_paths)
	transform = PCA(n_components=1, random_state=42)
	shape = latent_paths.shape
	latent_paths = transform.fit_transform(latent_paths.reshape(-1,32))
	latent_paths = latent_paths.reshape(shape[0], shape[1], 1)
	# Plot the traces.
	fig = plt.figure()
	axes = fig.subplots(nrows=2, ncols=1, sharex=True) # , sharex=True
	ts = np.linspace(0.05,0.64-0.05, latent_paths.shape[1])
	print(np.min(ts), np.max(ts))
	# taus, target_path = fit_taus(latent_paths, ts)
	undir_indices = [i for i in range(len(audio_filenames)) if 'UNDIR' in \
		audio_filenames[i]]
	dir_indices = [i for i in range(len(audio_filenames)) if 'UNDIR' not in \
		audio_filenames[i]]
	cs = ['b' if 'UNDIR' in fn else 'r' for fn in audio_filenames]
	lws = ['0.2' if 'UNDIR' in fn else '0.35' for fn in audio_filenames]
	for i in undir_indices:
		axes[1].plot(ts, latent_paths[i,:,0], c='b', alpha=0.1, lw=0.2)
	for i in dir_indices:
		axes[1].plot(ts, latent_paths[i,:,0], color='darkorange', alpha=0.4, lw=0.2)
	axes[1].set_ylim(-5, 5)
	axes[1].spines['right'].set_visible(False)
	axes[1].spines['top'].set_visible(False)
	axes[1].yaxis.set_ticks_position('left')
	axes[1].xaxis.set_ticks_position('bottom')
	axes[1].set_ylabel('PC 1')
	# Plot a spectrogram.
	ts = np.linspace(0,0.64, latent_paths.shape[1])
	fs, audio = wavfile.read(song_filename)
	f, t, spec = stft(audio, fs=fs, nperseg=512, noverlap=256+128+64)
	i1, i2 = np.searchsorted(f, 400), np.searchsorted(f, 12e3)
	spec = spec[i1:i2]
	f = f[i1:i2]
	spec = np.log(np.abs(spec) + 1e-12)
	extent=[np.min(t),np.max(t),np.min(f),np.max(f)]
	axes[0].imshow(spec, extent=[np.min(ts), np.max(ts), np.min(f)/1e3, np.max(f)/1e3], \
		origin='lower', vmin=np.quantile(spec,0.2), vmax=np.quantile(spec,0.99))
	# axes[0].axis('off')
	axes[0].spines['right'].set_visible(False)
	axes[0].spines['top'].set_visible(False)
	axes[0].set_ylabel('Frequency (kHz)')
	axes[0].set_aspect('auto')
	plt.xlabel('Time (s)')
	plt.savefig(os.path.join(dc.plots_dir, 'trace_plot.png'))


def inst_variability_plot_DC(dc, song_filename, filename='inst_variability.png'):
	"""
	TO DO: clean this up!
	"""
	# Collect audio chunks.
	segment_audio = dc.request('segment_audio')
	audio_chunks = []
	audio_filenames = []
	k = 0
	for audio_dir in dc.audio_dirs:
		for audio_fn in segment_audio[audio_dir]:
			temp = segment_audio[audio_dir][audio_fn]
			audio_chunks += temp
			k += 1
			audio_filenames += [os.path.join(audio_dir, audio_fn)] * len(temp)
	# NOTE: HERE
	latent_paths = np.load('latent_paths.npy')
	undir_indices = [i for i in range(len(audio_filenames)) if 'UNDIR' in \
		audio_filenames[i]]
	undir_indices = np.array(undir_indices, dtype='int')
	dir_indices = [i for i in range(len(audio_filenames)) if 'UNDIR' not in \
		audio_filenames[i]]
	dir_indices = np.array(dir_indices, dtype='int')
	dir_means = np.zeros(latent_paths.shape[1])
	undir_means = np.zeros(latent_paths.shape[1])
	# dir_hi = np.zeros(latent_paths.shape[1])
	# dir_lo = np.zeros(latent_paths.shape[1])
	# undir_hi = np.zeros(latent_paths.shape[1])
	# undir_lo = np.zeros(latent_paths.shape[1])
	for j in range(latent_paths.shape[1]):
		mean = np.mean(latent_paths[dir_indices,j,:], axis=0)
		l2_sum = np.sum(np.power(latent_paths[dir_indices,j,:] - mean, 2))
		std = np.sqrt(l2_sum / (len(dir_indices) - 1))
		dir_means[j] = std
		mean = np.mean(latent_paths[undir_indices,j,:], axis=0)
		l2_sum = np.sum(np.power(latent_paths[undir_indices,j,:] - mean, 2))
		std = np.sqrt(l2_sum / (len(undir_indices) - 1))
		undir_means[j] = std
	fig = plt.figure()
	axes = fig.subplots(nrows=2, ncols=1, sharex=True)
	# Plot a spectrogram.
	ts = np.linspace(0,0.64, latent_paths.shape[1])
	fs, audio = wavfile.read(song_filename)
	f, t, spec = stft(audio, fs=fs, nperseg=512, noverlap=256+128+64)
	i1, i2 = np.searchsorted(f, 400), np.searchsorted(f, 12e3)
	spec = spec[i1:i2]
	f = f[i1:i2]
	spec = np.log(np.abs(spec) + 1e-12)
	extent=[np.min(t),np.max(t),np.min(f),np.max(f)]
	axes[0].imshow(spec, extent=[np.min(ts), np.max(ts), np.min(f)/1e3, np.max(f)/1e3], \
		origin='lower', vmin=np.quantile(spec,0.2), vmax=np.quantile(spec,0.99))
	# axes[0].axis('off')
	axes[0].spines['right'].set_visible(False)
	axes[0].spines['top'].set_visible(False)
	axes[0].set_ylabel('Frequency (kHz)')
	axes[0].set_aspect('auto')
	# Plot the variability
	ts = np.linspace(0.05,0.64-0.05, latent_paths.shape[1])
	axes[1].plot(ts, dir_means, c='darkorange', label='Directed')
	axes[1].plot(ts, undir_means, c='b', label='Undirected')
	axes[1].set_xlabel('Time (s)')
	axes[1].set_ylabel('Variability Index')
	axes[1].set_xlim(0,0.64)
	axes[1].spines['right'].set_visible(False)
	axes[1].spines['top'].set_visible(False)
	axes[1].yaxis.set_ticks_position('left')
	axes[1].xaxis.set_ticks_position('bottom')
	axes[0].set_title('Within-song Variability')
	plt.legend(loc='best')
	plt.savefig(os.path.join(dc.plots_dir, filename))
	plt.close('all')


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


def subtract_mean(latent_paths):
	for i in range(latent_paths.shape[1]):
		for j in range(latent_paths.shape[2]):
			latent_paths[:,i,j] -= np.mean(latent_paths[:,i,j])
	return latent_paths


if __name__ == '__main__':
	pass


###
