"""
Segment song bouts using linear acoustic feature templates.


TO DO:
	- Align the examplar spectrograms?
"""
__date__ = "April-August 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from scipy.ndimage.filters import gaussian_filter
import os
import umap
import warnings

from ava.plotting.tooltip_plot import tooltip_plot

# Silence numpy.loadtxt when reading empty files.
warnings.filterwarnings("ignore", category=UserWarning)

EPSILON = 1e-12
NUM_MAD = 1.8



def get_spec(fs, audio, p):
	"""Not many options here."""
	f, t, spec = stft(audio, fs=fs, nperseg=p['nperseg'], noverlap=p['noverlap'])
	i1 = np.searchsorted(f, p['min_freq'])
	i2 = np.searchsorted(f, p['max_freq'])
	spec = spec[i1:i2]
	spec = np.log(np.abs(spec) + EPSILON)
	spec -= p['spec_min_val']
	spec /= p['spec_max_val'] - p['spec_min_val']
	spec[spec<0.0] = 0.0
	spec[spec>1.0] = 1.0
	return spec, t[1]-t[0]


def get_template(feature_dir, p):
	"""
	Create a linear template given exemplar spectrograms.

	Parameters
	----------
	feature_dir : str
		Directory containing multiple audio files to average together.
	p : dict
		Parameters.

	Returns
	-------
	template : np.ndarray
		...
	dt : float
		...
	"""
	filenames = [os.path.join(feature_dir, i) for i in os.listdir(feature_dir) \
		if is_audio_file(i)]
	specs = []
	for i, filename in enumerate(filenames):
		fs, audio = wavfile.read(filename)
		assert fs == p['fs'], "Found samplerate="+str(fs)+\
			", expected "+str(p['fs'])
		spec, dt = get_spec(fs, audio, p)
		spec = gaussian_filter(spec, (0.5,0.5))
		specs.append(spec)
	min_time_bins = min(spec.shape[1] for spec in specs)
	specs = np.array([i[:,:min_time_bins] for i in specs])
	template = np.mean(specs, axis=0) # Average over all the templates.
	template -= np.mean(template)
	template /= np.sum(np.power(template, 2)) + EPSILON
	return template, dt


def segment_files(audio_dirs, segment_dirs, template, p):
	"""Write segments"""
	result = {}
	for audio_dir, segment_dir in zip(audio_dirs, segment_dirs):
		audio_fns = [os.path.join(audio_dir, i) for i in os.listdir(audio_dir) \
			if is_audio_file(i)]
		for audio_fn in audio_fns:
			segments = segment_file(audio_fn, template, p)
			result[audio_fn] = segments
			segment_fn = audio_fn[:-4] + '.txt'
			segment_fn = os.path.join(segment_dir, segment_fn)
			np.savetxt(segment_fn, segments, fmt='%.5f', \
				delimiter=p['delimiter'])
	return result


def read_segment_decisions(audio_dirs, segment_dirs):
	"""Returns the same data as <segment_files>"""
	result = {}
	for audio_dir, segment_dir in zip(audio_dirs, segment_dirs):
		audio_fns = [os.path.join(audio_dir, i) for i in os.listdir(audio_dir) \
			if is_audio_file(i)]
		for audio_fn in audio_fns:
			segment_fn = audio_fn[:-4] + '.txt'
			segment_fn = os.path.join(segment_dir, segment_fn)
			segments = np.loadtxt(segment_fn)
			result[audio_fn] = segments
	return result


def segment_file(filename, template, p):
	"""
	Match linear audio features, extract times where features align.

	Parameters
	----------

	Returns
	-------

	Notes
	-----

	"""
	fs, audio = wavfile.read(filename)
	assert fs == p['fs'], "Found samplerate="+str(fs)+", expected "+str(p['fs'])
	big_spec, dt = get_spec(fs, audio, p)
	spec_len = template.shape[1]
	template = template.flatten()
	# Compute normalized cross-correlation
	result = np.zeros(big_spec.shape[1] - spec_len)
	for i in range(len(result)):
		temp = big_spec[:, i:i+spec_len].flatten()
		temp -= np.mean(temp)
		temp /= np.sum(np.power(temp, 2)) + EPSILON
		result[i] = np.dot(template, temp)
	median = np.median(result)
	abs_devs = np.abs(result - median)
	mad_sigma = np.median(abs_devs) + EPSILON

	# Get maxima.
	times = dt * np.arange(len(result))
	indices = np.argwhere(result>median + NUM_MAD).flatten()[1:-1]
	max_indices = []
	for i in range(2,len(indices)-1):
		if max(result[indices[i]-1], result[indices[i]+1]) < result[indices[i]]:
			max_indices.append(indices[i])
	max_indices = np.array(max_indices, dtype='int')
	max_indices = _clean_max_indices(max_indices, times, result)
	# Define onsets/offsets.
	segments = np.zeros((len(max_indices), 2))
	segments[:,0] = dt * max_indices # onsets
	segments[:,1] = segments[:,0] + spec_len * dt
	return segments


def clean_collected_data(result, audio_dirs, segment_dirs, template_length, p, \
	n=10**4):
	"""
	Take a look at the collected data and discard false positives.

	Parameters
	----------
	result : ...
	...
	audio_dirs : ...
	...
	segment_dirs : ...
	...

	Notes
	-----

	"""
	# Collect spectrograms.
	specs = []
	if template_length is not None:
		delta_i = int(round(template_length * p['fs']))
	for filename in result.keys():
		fs, audio = wavfile.read(filename)
		assert fs == p['fs']
		for segment in result[filename]:
			i1 = int(round(segment[0] * fs))
			if template_length is None:
				i2 = int(round(segment[1] * fs))
			else:
				i2 = i1 + delta_i
			spec, dt = get_spec(fs, audio[i1:i2], p)
			specs.append(spec)
	if template_length is None:
		max_t = max(spec.shape[1] for spec in specs)
		temp_specs = np.zeros((len(specs), specs[0].shape[0], max_t))
		for i, spec in enumerate(specs):
			temp_specs[i,:,:spec.shape[1]] = spec
		specs = temp_specs
	else:
		specs = np.array(specs)
	np.random.seed(42)
	specs = specs[np.random.permutation(len(specs))]
	np.random.seed(None)
	# UMAP the spectrograms.
	transform = umap.UMAP(random_state=42)
	embedding = transform.fit_transform(specs.reshape(len(specs), -1))
	# Plot and ask for user input.
	bounds = {
		'x1s':[],
		'x2s':[],
		'y1s':[],
		'y2s':[],
	}
	X, Y = embedding[:,0], embedding[:,1]
	i = 0
	while True:
		colors = ['b' if in_region(embed, bounds) else 'r' for embed in embedding]
		print("Selected ", len([c for c in colors if c=='b']), "out of", len(colors))
		plt.scatter(X, Y, c=colors, s=0.9, alpha=0.5)
		for x_tick in np.arange(np.floor(np.min(X)), np.ceil(np.max(X))):
			plt.axvline(x=x_tick, c='k', alpha=0.1, lw=0.5)
		for y_tick in np.arange(np.floor(np.min(Y)), np.ceil(np.max(Y))):
			plt.axhline(y=y_tick, c='k', alpha=0.1, lw=0.5)
		title = "Select relevant song:"
		plt.title(title)
		plt.savefig('temp.pdf')
		plt.close('all')
		if i == 0:
			tooltip_plot(embedding, specs, num_imgs=10**3, title=title)
		bounds['x1s'].append(float(input('x1: ')))
		bounds['x2s'].append(float(input('x2: ')))
		bounds['y1s'].append(float(input('y1: ')))
		bounds['y2s'].append(float(input('y2: ')))
		temp = input('(<c> to continue) ')
		if temp == 'c':
			break
		i += 1
	np.save('bounds.npy', bounds)
	# Save only the good segments.
	num_deleted, num_total = 0, 0
	for audio_dir, seg_dir in zip(audio_dirs, segment_dirs):
		audio_fns = [os.path.join(audio_dir, i) for i in os.listdir(audio_dir) \
			if is_audio_file(i)]
		for audio_fn in audio_fns:
			fs, audio = wavfile.read(audio_fn)
			assert fs == p['fs']
			segment_fn = audio_fn[:-4] + '.txt'
			segment_fn = os.path.join(seg_dir, segment_fn)
			segments = np.loadtxt(segment_fn, \
				delimiter=p['delimiter']).reshape(-1,2)
			if len(segments) == 0:
				continue
			new_segments = np.zeros(segments.shape)
			i = 0
			for segment in segments:
				i1 = int(round(segment[0] * fs))
				if template_length is None:
					i2 = int(round(segment[1] * fs))
				else:
					i2 = i1 + delta_i
				spec, dt = get_spec(fs, audio[i1:i2], p)
				if template_length is None:
					temp_spec = np.zeros((spec.shape[0], max_t))
					temp_spec[:,:spec.shape[1]] = spec
					spec = temp_spec
				embed = transform.transform(spec.reshape(1,-1)).reshape(2)
				if in_region(embed, bounds):
					new_segments[i] = segment[:]
					i += 1
					num_total += 1
				else:
					num_deleted += 1
			new_segments = new_segments[:i]
			np.savetxt(segment_fn, new_segments, fmt='%.5f', \
				delimiter=p['delimiter'])
	print("deleted", num_deleted, "total", num_total)


def _clean_max_indices(old_indices, old_times, values, min_dt=0.1):
	"""Remove maxima that are too close together."""
	if len(old_indices) <= 1:
		return old_indices
	old_indices = old_indices[np.argsort(values[old_indices])]
	indices = [old_indices[0]]
	times = [old_times[old_indices[0]]]
	i = 1
	while i < len(old_indices):
		time = old_times[old_indices[i]]
		flag = True
		for j in range(len(indices)):
			if abs(old_times[indices[j]] - time) < min_dt:
				flag = False
				break
		if flag:
			indices.append(old_indices[i])
			times.append(old_times[old_indices[i]])
		i += 1
	indices = np.array(indices)
	indices.sort()
	return indices



def in_region(point, bounds):
	"""Is the point in the union of the given rectangles?"""
	for i in range(len(bounds['x1s'])):
		if point[0] > bounds['x1s'][i] and point[0] < bounds['x2s'][i] and \
				point[1] > bounds['y1s'][i] and point[1] < bounds['y2s'][i]:
			return True
	return False


def is_audio_file(filename):
	"""Does this filename have a recognized audio extension?"""
	return len(filename) > 4 and filename[-4:] in ['.wav']



if __name__ == '__main__':
	# # Take 1.
	# root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/'
	# feature_dir = root + 'song_templates/'
	# audio_dirs = [root + i for i in ['OPTO_CHUNK']]
	# seg_dirs = audio_dirs
	# p = {
	# 	'fs':32000,
	# 	'min_freq':400,
	# 	'max_freq':8e3,
	# 	'spec_min_val': 2.0,
	# 	'spec_max_val': 6.5,
	# 	'nperseg': 512,
	# 	'noverlap': 0,
	# 	'delimiter': '\t',
	# }
	#
	# template, dt = get_template(feature_dir, p)
	# template_length = template.shape[1] * dt
	#
	# plt.imshow(template, aspect='auto', origin='lower')
	# plt.colorbar()
	# plt.savefig('temp_spec.pdf')
	# plt.close('all')
	#
	# result = segment_files(audio_dirs, seg_dirs, template, p)
	#
	# clean_collected_data(result, audio_dirs, seg_dirs, template_length, p)

	# Take 2.
	p = {
		'fs':32000,
		'min_freq':400,
		'max_freq':8e3,
		'spec_min_val': 2.0,
		'spec_max_val': 6.5,
		'nperseg': 512,
		'noverlap': 0,
		'delimiter': ' ',
	}
	root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/'
	audio_dirs = [root + i for i in ['OPTO_SAP', 'DIR_SAP', 'UNDIR_SAP']]
	seg_dirs = audio_dirs
	result = read_segment_decisions(audio_dirs, seg_dirs)
	clean_collected_data(result, audio_dirs, seg_dirs, None, p)


###
