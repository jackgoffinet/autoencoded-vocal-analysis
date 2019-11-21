"""
Make a movie out of a shotgun VAE projection and an audio file.

"""
__date__ = "Novemeber 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from scipy.io import wavfile
from scipy.spatial.distance import euclidean
import torch
from torch.utils.data import Dataset, DataLoader

from ava.models.vae import VAE



def shotgun_movie_DC(dc, audio_file, p, output_dir='temp', fps=30, \
	shoulder=0.05, c='b', alpha=0.2, s=0.9, marker_c='r', marker_s=40.0, \
	marker_marker='*'):
	"""
	Make a shotgun VAE projection movie with the given audio file.

	This will write a series of images to ``output_dir``. To make the ``.mp4``,
	a couple extra commands will have to be entered in the command line:

	::

	$ ffmpeg -r <fps> -i <output_dir>/%04d.png -i "<audio_file>" -c:a aac -strict -2 -shortest -y output.mp4


	Parameters
	----------
	dc : ava.data.data_container.DataContainer
		See ava.data.data_container.
	audio_file : str
		Path to audio file.
	p : dict
		Preprocessing parameters. Must contain keys: ``'fs'``, ``'get_spec'``,
		``'num_freq_bins'``, ``'num_time_bins'``, ``'nperseg'``, ``'noverlap'``,
		``'window_length'``, ``'min_freq'``, ``'max_freq'``, ``'spec_min_val'``,
		``'spec_max_val'``, ``'mel'``, ...
	output_dir : str, optional
		Directory where output images are written. Defaults to ``'temp'``.
	fps : int, optional
		Frames per second. Defaults to ``20``.
	shoulder : float, optional
		The movie will start this far into the audio file and stop this far from
		the end. This removes weird edge effect of making spectrograms. Defaults
		to ``0.05``.
	c : str, optional
		Passed to ``matplotlib.pyplot.scatter`` for background points. Defaults
		to ``'b'``.
	alpha : float, optional
		Passed to ``matplotlib.pyplot.scatter`` for background points. Defaults
		to ``0.2``.
	s : float, optional
		Passed to ``matplotlib.pyplot.scatter`` for background points. Defaults
		to ``0.9``.
	marker_c : str, optional
		Passed to ``matplotlib.pyplot.scatter`` for the marker. Defaults to
		``'r'``.
	marker_s : float, optional
		Passed to ``matplotlib.pyplot.scatter`` for the marker. Defaults to
		``40.0``.
	marker_marker : str, optional
		Passed to ``matplotlib.pyplot.scatter`` for the marker. Defaults to
		``'r'``.
	"""
	assert dc.model_filename is not None
	# Read the audio file.
	fs, audio = wavfile.read(audio_file)
	assert fs == p['fs'], "found fs="+str(fs)+", expected "+str(p['fs'])
	# Make spectrograms.
	specs = []
	dt = 1/fps
	onset = shoulder
	while onset + p['window_length'] < len(audio)/fs - shoulder:
		offset = onset + p['window_length']
		target_times = np.linspace(onset, offset, p['num_time_bins'])
		# Then make a spectrogram.
		spec, flag = p['get_spec'](onset-shoulder, offset+shoulder, audio, p, \
				fs=fs, target_times=target_times)
		assert flag
		specs.append(spec)
		onset += dt
	# Make a DataLoader out of these spectrograms.
	specs = np.stack(specs)
	loader = DataLoader(SimpleDataset(specs))
	# Get latent means.
	model = VAE()
	model.load_state(dc.model_filename)
	latent = model.get_latent(loader)
	# Get original latent and embeddings.
	original_embed = dc.request('latent_mean_umap')
	original_latent = dc.request('latent_means')
	# Find nearest neighbors in latent space to determine embeddings.
	new_embed = np.zeros((len(latent),2))
	for i in range(len(latent)):
		index = np.argmin([euclidean(latent[i], j) for j in original_latent])
		new_embed[i] = original_embed[index]
	# Calculate x and y limits.
	xmin = np.min(original_embed[:,0])
	ymin = np.min(original_embed[:,1])
	xmax = np.max(original_embed[:,0])
	ymax = np.max(original_embed[:,1])
	x_pad = 0.05 * (xmax - xmin)
	y_pad = 0.05 * (ymax - ymin)
	xmin, xmax = xmin - x_pad, xmax + x_pad
	ymin, ymax = ymin - y_pad, ymax + y_pad
	# Save images.
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	for i in range(len(new_embed)):
		plt.scatter(original_embed[:,0], original_embed[:,1], c=[c]*len(original_embed), \
				alpha=alpha, s=s)
		plt.scatter([new_embed[i,0]], [new_embed[i,1]], s=marker_s, \
				marker=marker_marker, c=marker_c)
		plt.xlim(xmin, xmax)
		plt.ylim(ymin, ymax)
		plt.axis('off')
		fn = str(i).zfill(4) + '.png'
		fn = os.path.join(output_dir, fn)
		plt.savefig(fn)
		plt.close('all')



class SimpleDataset(Dataset):
	def __init__(self, specs):
		self.specs = specs

	def __len__(self):
		return self.specs.shape[0]

	def __getitem__(self, index):
		return torch.from_numpy(self.specs[index]).type(torch.FloatTensor)


if __name__ == '__main__':
	pass



###
