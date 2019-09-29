"""
Plot traces for sliding window analysis.

TO DO: clean this up!
"""
__author__ = "Jack Goffinet"
__date__ = "August-September 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.patches import Patch
import numpy as np
import os
from scipy.io import wavfile
from scipy.signal import stft
from sklearn.decomposition import PCA
import torch

from ava.models.vae import X_SHAPE, VAE
from ava.models.window_vae_dataset import WarpedWindowDataset, numpy_to_tensor



def spectrogram_plot(song_filename, ax=None, save_and_close=True, \
	sign=1, x_label=True, filename='spec.pdf'):
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
	spec *= sign
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



# def variability_plot(traces, audio_fns, template_dur, window_length, ax=None, \
# 	save_and_close=True, filename='variability.pdf'):
# 	"""
# 	Parameters
# 	----------
# 	"""
# 	# Calculate variability index.
# 	undir_indices = [i for i in range(len(audio_fns)) if 'UNDIR' in \
# 		audio_fns[i]]
# 	undir_indices = np.array(undir_indices, dtype='int')
# 	dir_indices = [i for i in range(len(audio_fns)) if 'UNDIR' not in \
# 		audio_fns[i]]
# 	dir_indices = np.array(dir_indices, dtype='int')
# 	dir_variability = np.zeros(treaces.shape[1])
# 	undir_variability = np.zeros(treaces.shape[1])
# 	for j in range(traces.shape[1]):
# 		mean = np.mean(traces[dir_indices,j,:], axis=0)
# 		l2_sum = np.sum(np.power(traces[dir_indices,j,:] - mean, 2))
# 		std = np.sqrt(l2_sum / (len(dir_indices) - 1))
# 		dir_variability[j] = std
# 		mean = np.mean(traces[undir_indices,j,:], axis=0)
# 		l2_sum = np.sum(np.power(traces[undir_indices,j,:] - mean, 2))
# 		std = np.sqrt(l2_sum / (len(undir_indices) - 1))
# 		undir_variability[j] = std
# 	# Plot.
# 	if ax is None:
# 		ax = plt.gca()
# 	ts = np.linspace(window_length/2,template_dur-window_length/2,traces.shape[1])
# 	ts = ts * 1e3
# 	ax.plot(ts, undir_variability, c='b', label='Undirected', alpha=0.2)
# 	ax.plot(ts, dir_variability, c='darkorange', label='Directed', alpha=0.2)
# 	ax.set_xlabel('Time (ms)')
# 	ax.set_ylabel('Variability Index')
# 	# ax.set_xlim(0,template_dur*1e3)
# 	ax.spines['right'].set_visible(False)
# 	ax.spines['top'].set_visible(False)
# 	ax.yaxis.set_ticks_position('left')
# 	ax.xaxis.set_ticks_position('bottom')
# 	if save_and_close:
# 		plt.savefig(filename)
# 		plt.close('all')


def warped_trace_plot_DC(dc, p, num_points=200, latent_dim=32, ax=None, \
	save_and_close=True, load_warp=False, load_traces=False, \
	colors=['b', 'darkorange'], filename='warped_traces.pdf', \
	load_fn='temp_data/sliding_window_data.npy', \
	warp_fns=['temp_data/x_knots.npy', 'temp_data/y_knots.npy'],
	fn_to_group=None, unique_groups=[0,1], labels=[None, None]):
	"""
	Parameters
	----------

	"""
	d = _get_warped_data_from_DC(dc, p, num_points, latent_dim, \
			load_warp=load_warp, load_traces=load_traces, load_fn=load_fn, \
			warp_fns=warp_fns)
	trace_plot(d['traces'], d['audio_fns'], d['ts'], ax=ax, \
			fn_to_group=fn_to_group, save_and_close=save_and_close, \
			unique_groups=unique_groups, colors=colors, labels=labels, \
			filename=filename)


def trace_plot(traces, audio_fns, ts, ax=None, save_and_close=True, \
	colors=['b', 'darkorange'], labels=[None, None], fn_to_group=None, \
	unique_groups=[0,1], filename='traces.png'):
	"""
	Plot latent traces.

	Parameters
	----------
	traces : numpy.ndarray
		...
	audio_fns : list of str
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
	assert fn_to_group is not None
	transform = PCA(n_components=1, random_state=42)
	shape = traces.shape
	traces = transform.fit_transform(traces.reshape(-1,32))
	traces = traces.reshape(shape[0], shape[1], 1)

	# Collect indices.
	trace_groups = np.array([fn_to_group(fn) for fn in audio_fns], dtype='int')
	group_indices = []
	for group in unique_groups:
		indices = np.argwhere(trace_groups == group).flatten()
		group_indices.append(indices)

	# Plot the traces.
	if ax is None:
		ax = plt.gca()

	# undir_indices = np.array([i for i in range(len(audio_filenames)) if \
	# 		'UNDIR' in audio_filenames[i]], dtype='int')
	# dir_indices = np.array([i for i in range(len(audio_filenames)) if \
	# 		'UNDIR' not in audio_filenames[i]], dtype='int')

	for i in range(len(colors)):
		ax.plot(ts, traces[group_indices[i],:,0].T, c=colors[i], lw=0.2, label=labels[i])

	# ax.set_ylim(-5, 5)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	ax.set_ylabel('Principal Component 1')
	if save_and_close:
		plt.savefig(filename)
		plt.close('all')


def warped_variability_plot_DC(dc, p, num_points=200, latent_dim=32, ax=None, \
	save_and_close=True, load_warp=False, load_traces=False, \
	colors=['b', 'darkorange'], fn_to_group=None, unique_groups=[0,1], lw=1.0, \
	filename='inst_variability.pdf', labels=[None, None],
	load_fn='temp_data/sliding_window_data.npy', \
	warp_fns=['temp_data/x_knots.npy', 'temp_data/y_knots.npy'], legend=False):
	"""
	Parameters
	----------
	dc : ...
		...
	p : ...
		...
	"""
	assert fn_to_group is not None
	d = _get_warped_data_from_DC(dc, p, num_points, latent_dim, \
			load_warp=load_warp, load_traces=load_traces, load_fn=load_fn,
			warp_fns=warp_fns)
	traces = d['traces']
	audio_fns = d['audio_fns']
	# Compute variability index.
	trace_groups = np.array([fn_to_group(fn) for fn in audio_fns], dtype='int')
	group_indices = []
	group_variability = []
	for group in unique_groups:
		indices = np.argwhere(trace_groups == group).flatten()
		group_indices.append(indices)
		group_variability.append(np.zeros(traces.shape[1]))

	for i in range(len(group_indices)):
		indices = group_indices[i]
		for j in range(traces.shape[1]):
			median = np.zeros(traces.shape[2])
			for k in range(traces.shape[2]):
				median[k] = np.median(traces[indices,j,k])
			deviations = np.sum(np.power(traces[indices,j,:] - median, 2), \
					axis=-1)
			group_variability[i][j] = np.median(deviations)

	if ax is None:
		ax = plt.gca()
	for i in range(len(colors)):
		group_variability[i][:3] = np.nan # NOTE: TEMP!
		ax.plot(d['ts'], group_variability[i], c=colors[i], label=labels[i], lw=lw)
	ax.set_xlabel('Time (ms)')
	ax.set_ylabel('Variability Index')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	if legend:
		patches = [Patch(color=colors[i], label=labels[i]) \
			for i in range(len(colors))]
		ax.legend(handles=patches, loc='best', fontsize=7)
	if save_and_close:
		plt.savefig(os.path.join(dc.plots_dir, filename))
		plt.close('all')


def _get_warped_data_from_DC(dc, p, num_points, latent_dim, load_warp=False, \
	load_traces=False, load_fn='temp_data/sliding_window_data.npy',
	warp_fns=['temp_data/x_knots.npy', 'temp_data/y_knots.npy']):
	"""
	Get traces, filenames, and template duration.

	Parameters
	----------
	dc : ava.data.data_container.DataContainer
		...

	"""
	if load_traces:
		try:
			return np.load(load_fn, allow_pickle=True).item()
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
	dset = WarpedWindowDataset(audio_fns, dc.template_dir, p, load_warp=load_warp,
		warp_fns=warp_fns)
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
	np.save(load_fn, data)
	return data



if __name__ == '__main__':
	pass


###
