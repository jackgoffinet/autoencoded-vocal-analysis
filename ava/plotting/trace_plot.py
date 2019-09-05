"""
Plot traces for sliding window analysis.

TO DO: clean this up!
"""
__author__ = "Jack Goffinet"
__date__ = "August-September 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from scipy.io import wavfile
from scipy.signal import stft
from sklearn.decomposition import PCA
import torch

from ava.models.vae import X_SHAPE, VAE
from ava.models.window_vae_dataset import WarpedWindowDataset, numpy_to_tensor



def trace_plot(traces, audio_filenames, ts, ax=None, save_and_close=True, \
	filename='traces.png'):
	"""
	Plot latent traces.

	Parameters
	----------
	traces : numpy.ndarray
		...
	audio_filenames : list of str
		...
	ts : numpy.ndarray
		...
	ax : ...
		...
	save_and_close : bool
		Defaults to ``True``.
	filename : str
		...

	"""
	transform = PCA(n_components=1, random_state=42)
	shape = traces.shape
	traces = transform.fit_transform(traces.reshape(-1,32))
	traces = traces.reshape(shape[0], shape[1], 1)
	# Plot the traces.
	if ax is None:
		ax = plt.gca()
	print(np.min(ts), np.max(ts))
	undir_indices = np.array([i for i in range(len(audio_filenames)) if \
			'UNDIR' in audio_filenames[i]], dtype='int')
	dir_indices = np.array([i for i in range(len(audio_filenames)) if \
			'UNDIR' not in audio_filenames[i]], dtype='int')

	ax.plot(ts, traces[undir_indices,:,0].T, c='b', alpha=0.08, lw=0.2)
	# ax.plot(ts, traces[dir_indices,:,0].T, c='darkorange', alpha=0.15, lw=0.2)
	# ax.set_ylim(-5, 5)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	ax.set_ylabel('Principal Component 1')
	if save_and_close:
		plt.savefig(filename)
		plt.close('all')


def spectrogram_plot(song_filename, ax=None, save_and_close=True, \
	x_label=True, filename='spec.pdf'):
	"""
	Plot a spectrogram.

	Parameters
	----------
	song_filename : str
		...
	ax : ...
		...
	save_and_close : bool
		Defaults to ``True``.
	x_label : bool
		Defaults to ``True``.
	filename : str
		Defaults to ``'spec.pdf'``.

	"""
	# Get the spectrogram.
	fs, audio = wavfile.read(song_filename)
	f, t, spec = stft(audio, fs=fs, nperseg=512, noverlap=256+128+64)
	t = t * 1e3
	i1, i2 = np.searchsorted(f, 400), np.searchsorted(f, 12e3)
	spec = spec[i1:i2]
	f = f[i1:i2]
	spec = np.log(np.abs(spec) + 1e-12)
	extent=[np.min(t),np.max(t),np.min(f)/1e3,np.max(f)/1e3]
	# Plot.
	if ax is None:
		ax = plt.gca()
	ax.imshow(spec, extent=extent, \
		origin='lower', vmin=np.quantile(spec,0.2), vmax=np.quantile(spec,0.99))
	# axes[0].axis('off')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_ylabel('Frequency (kHz)')
	ax.set_aspect('auto')
	ax.set_xlim(extent[0], extent[1])
	if x_label:
		ax.set_xlabel('Time (ms)')
	if save_and_close:
		plt.savefig(filename)
		plt.close('all')



def variability_plot(traces, audio_fns, template_dur, window_length, ax=None, \
	save_and_close=True, filename='variability.pdf'):
	"""
	Parameters
	----------
	"""
	# Calculate variability index.
	undir_indices = [i for i in range(len(audio_fns)) if 'UNDIR' in \
		audio_fns[i]]
	undir_indices = np.array(undir_indices, dtype='int')
	dir_indices = [i for i in range(len(audio_fns)) if 'UNDIR' not in \
		audio_fns[i]]
	dir_indices = np.array(dir_indices, dtype='int')
	dir_variability = np.zeros(treaces.shape[1])
	undir_variability = np.zeros(treaces.shape[1])
	for j in range(traces.shape[1]):
		mean = np.mean(traces[dir_indices,j,:], axis=0)
		l2_sum = np.sum(np.power(traces[dir_indices,j,:] - mean, 2))
		std = np.sqrt(l2_sum / (len(dir_indices) - 1))
		dir_variability[j] = std
		mean = np.mean(traces[undir_indices,j,:], axis=0)
		l2_sum = np.sum(np.power(traces[undir_indices,j,:] - mean, 2))
		std = np.sqrt(l2_sum / (len(undir_indices) - 1))
		undir_variability[j] = std
	# Plot.
	if ax is None:
		ax = plt.gca()
	ts = np.linspace(window_length/2,template_dur-window_length/2,traces.shape[1])
	ts = ts * 1e3
	ax.plot(ts, undir_variability, c='b', label='Undirected', alpha=0.2)
	ax.plot(ts, dir_variability, c='darkorange', label='Directed', alpha=0.2)
	ax.set_xlabel('Time (ms)')
	ax.set_ylabel('Variability Index')
	ax.set_xlim(0,template_dur*1e3)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	if save_and_close:
		plt.savefig(filename)
		plt.close('all')


def warped_trace_plot_DC(dc, p, num_points=200, latent_dim=32, ax=None, \
	save_and_close=True, load_warp=False, load_traces=False, \
	filename='warped_traces.pdf'):
	"""

	"""
	d = _get_warped_data_from_DC(dc, p, num_points, latent_dim, \
			load_warp=load_warp, load_traces=load_traces)
	trace_plot(d['traces'], d['audio_fns'], d['ts'], ax=ax, \
			save_and_close=save_and_close, filename=filename)


def warped_variability_plot_DC(dc, p, num_points=200, latent_dim=32, ax=None, \
	save_and_close=True, load_warp=False, load_traces=False, \
	filename='inst_variability.pdf'):
	"""
	Parameters
	----------
	"""
	d = _get_warped_data_from_DC(dc, p, num_points, latent_dim, \
			load_warp=load_warp, load_traces=load_traces)
	traces = d['traces']
	audio_fns = d['audio_fns']
	# Compute variability index.
	undir_indices = [i for i in range(len(audio_fns)) if 'UNDIR' in \
		audio_fns[i]]
	undir_indices = np.array(undir_indices, dtype='int')
	dir_indices = [i for i in range(len(audio_fns)) if 'UNDIR' not in \
		audio_fns[i]]
	dir_indices = np.array(dir_indices, dtype='int')
	undir_variability = np.zeros(traces.shape[1])
	dir_variability = np.zeros(traces.shape[1])
	for j in range(traces.shape[1]):
		mean = np.mean(traces[dir_indices,j,:], axis=0)
		l2_sum = np.sum(np.power(traces[dir_indices,j,:] - mean, 2))
		std = np.sqrt(l2_sum / (len(dir_indices) - 1))
		dir_variability[j] = std
		mean = np.mean(traces[undir_indices,j,:], axis=0)
		l2_sum = np.sum(np.power(traces[undir_indices,j,:] - mean, 2))
		std = np.sqrt(l2_sum / (len(undir_indices) - 1))
		undir_variability[j] = std
	if ax is None:
		ax = plt.gca()
	ax.plot(d['ts'], undir_variability, c='b', label='Undirected')
	ax.plot(d['ts'], dir_variability, c='darkorange', label='Directed')
	print(np.min(d['ts']), np.max(d['ts']))
	ax.set_xlabel('Time (ms)')
	ax.set_ylabel('Variability Index')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	if save_and_close:
		plt.savefig(os.path.join(dc.pots_dir, filename))
		plt.close('all')


def _get_warped_data_from_DC(dc, p, num_points, latent_dim, load_warp=False, \
	load_traces=False):
	"""
	Get traces, filenames, and template duration.

	Parameters
	----------
	dc : ava.data.data_container.DataContainer
		...

	"""
	if load_traces:
		try:
			return np.load('temp_data/sliding_window_data.npy', allow_pickle=True).item()
		except:
			pass
	assert dc.audio_dirs is not None
	assert dc.template_dir is not None
	assert dc.model_filename is not None
	data = {}
	# List audio files.
	audio_fns = []
	for audio_dir in dc.audio_dirs:
		audio_fns += [os.path.join(audio_dir, i) for i in \
			sorted(os.listdir(audio_dir)) if i[-4:] == '.wav']
	data['audio_fns'] = audio_fns
	# Get warps.
	dset = WarpedWindowDataset(audio_fns, dc.template_dir, p, load_warp=load_warp)
	data['template_dur'] = dset.template_dur
	# Make the model.
	model = VAE()
	model.load_state(dc.model_filename)
	# Get traces.
	quantiles = np.linspace(0,1,num_points) # NOTE: HERE!
	traces = np.zeros((len(audio_fns), num_points, latent_dim))
	print("Making traces...")
	for i, audio_fn in enumerate(dset.audio_filenames):
		if i % 100 == 0:
			print(i)
		spec_arr = np.zeros((num_points,) + X_SHAPE)
		for j, quantile in enumerate(quantiles):
			spec = dset.get_specific_item(audio_fn, quantile)
			spec_arr[j] = spec
		with torch.no_grad():
			spec_arr = numpy_to_tensor(spec_arr).to(model.device)
			temp, _, _ = model.encode(spec_arr)
		traces[i] = temp.detach().cpu().numpy().reshape(num_points,latent_dim)
	data['traces'] = traces
	ts = np.linspace(0,dset.template_dur-p['window_length'], num_points)
	ts = (ts + 0.5 * p['window_length']) * 1e3
	data['ts'] = ts
	np.save('temp_data/sliding_window_data.npy', data)
	return data


if __name__ == '__main__':
	pass


###
