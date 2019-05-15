TEMPLATE_DURATION"""
Segment song bouts using linear acoustic feature templates.


TO DO:
	- get rid of NUM_SPLITS
	- generalize to single syllable use.
"""
__author__ = "Jack Goffinet"
__date__ = "April-May 2019"


import h5py
import os
from time import strptime, mktime
from tqdm import tqdm

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, resample
from scipy.ndimage.filters import gaussian_filter, convolve
from skimage.transform import resize
import umap

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from .interactive_segmentation import make_html_plot


MIN_FREQ, MAX_FREQ = 300, 8e3
FS = 44100
EPSILON = 1e-9
SPEC_THRESH = -4.0

NUM_SPLITS = 1
NUM_SIGMA = 5.0

TEMPLATE_PREFIX = 0.05
TEMPLATE_DURATION = 0.9



def get_spec(audio, f_ind, fs=FS):
	f, t, Zxx = stft(audio, fs=fs)
	if f_ind is None:
		f_ind = np.searchsorted(f, [MIN_FREQ, MAX_FREQ])
	Zxx = Zxx[f_ind[0]:f_ind[1]]
	Zxx = np.log(np.abs(Zxx) + EPSILON)
	Zxx -= SPEC_THRESH
	Zxx[Zxx < 0.0] = 0.0
	Zxx /= np.max(Zxx) + EPSILON
	return Zxx, f_ind, t[1] - t[0]

def get_spec_norm(audio, f_ind, fs=FS):
	f, t, Zxx = stft(audio, fs=fs)
	if f_ind is None:
		f_ind = np.searchsorted(f, [MIN_FREQ, MAX_FREQ])
	Zxx = Zxx[f_ind[0]:f_ind[1]]
	Zxx = np.log(np.abs(Zxx) + EPSILON)
	Zxx -= SPEC_THRESH
	Zxx[Zxx < 0.0] = 0.0
	for j in range(Zxx.shape[1]):
		Zxx[:,j] /= np.sum(Zxx[:,j]) + EPSILON
	return Zxx, f_ind, t[1] - t[0]


def process_sylls(load_dir, save_dir, feature_dir, p):
	"""
	Main method: process files in <load_dir> and save to <save_dir>.

	Parameters
	----------

	Returns
	-------

	Notes
	-----
	"""
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	filenames = [os.path.join(load_dir, i) for i in os.listdir(load_dir) if is_audio_file(i)]
	np.random.shuffle(filenames)
	write_file_num = 0
	song_data = {
		'audio':[],
		'time':[],
		'file_time':[],
		'filename':[],
		'fs': [],
	}
	songs_per_file = p['songs_per_file']
	print("Processing audio files in", load_dir)
	features = get_features(feature_dir)
	for i, filename in enumerate(filenames):
		result, features = segment_file(filename, features, feature_dir)
		for key in result:
			song_data[key] += result[key]
		while len(song_data['time']) >= songs_per_file:
			save_filename = "songs_" + str(write_file_num).zfill(3) + '.hdf5'
			save_filename = os.path.join(save_dir, save_filename)
			song_data = save_data(save_filename, song_data, songs_per_file)
			write_file_num += 1


def save_data(save_filename, song_data, songs_per_file):
	with h5py.File(save_filename, "w") as f:
		for key in song_data.keys():
			if key in ['filename', 'audio']:
				continue
			f.create_dataset(key, data=np.array(song_data[key][:songs_per_file]))
		f.create_dataset('filename', data=np.array(song_data['filename'][:songs_per_file]).astype('S'))
		max_num_samples = max([len(i) for i in song_data['audio'][:songs_per_file]])
		audio = np.zeros((songs_per_file, max_num_samples))
		for i in range(songs_per_file):
			audio[i,:len(song_data['audio'][i])] = song_data['audio'][i]
		f.create_dataset('audio', data=audio)
	for k in song_data:
		song_data[k] = song_data[k][songs_per_file:]
	return song_data


def get_features(feature_dir):
	"""
	Create linear features given exemplar spectrograms.

	Parameters
	----------

	Returns
	-------

	"""
	samplerates = []
	filenames = [os.path.join(feature_dir, i) for i in os.listdir(feature_dir) if is_audio_file(i)]
	for i, filename in enumerate(filenames):
		fs, _ = wavfile.read(filename)
		if fs not in samplerates:
			samplerates.append(fs)
	d = {}
	for fs in samplerates:
		specs = []
		min_time_bins = 10**10
		for i, filename in enumerate(filenames):
			temp_fs, audio = wavfile.read(filename)
			if temp_fs != fs:
				continue
			spec, _, dt = get_spec_norm(audio, None, fs=fs)
			spec = gaussian_filter(spec, (1,1))
			specs.append(spec)
			if spec.shape[1] < min_time_bins:
				min_time_bins = spec.shape[1]
		specs = np.array([i[:,:min_time_bins] for i in specs])
		spec = np.mean(specs, axis=0)
		remainder = spec.shape[1] % NUM_SPLITS
		if remainder != 0:
			spec = spec[:,:-NUM_SPLITS+remainder]
		spec_len = spec.shape[1] // NUM_SPLITS
		features = []
		for i in range(NUM_SPLITS):
			feature = spec[:,i*spec_len:(i+1)*spec_len]
			feature -= np.mean(feature)
			feature /= np.std(feature) + EPSILON
			features.append(feature)
		d[fs] = features
	return d


def segment_file(filename, features, feature_dir):
	"""
	Match audio features, extract times where features align.

	Parameters
	----------

	Returns
	-------

	Notes
	-----

	"""
	fs, audio = wavfile.read(filename)
	if fs not in features:
		print("found fs", fs, filename)
		print(list(features.keys()))
		quit()
	big_spec, _, dt = get_spec_norm(audio, None, fs=fs)
	for i in range(big_spec.shape[1]):
		big_spec[:,i] /= np.sum(big_spec[:,i]) + EPSILON
	spec_len = features[fs][0].shape[1]
	# Compute normalized cross-correlation
	result = np.zeros(big_spec.shape[1]-NUM_SPLITS*spec_len)
	for i in range(len(result)):
		for j in range(NUM_SPLITS):
			temp = big_spec[:,i+j*spec_len:i+(j+1)*spec_len]
			if j == 0:
				result[i] = np.sum(np.power(features[fs][j] - temp,2))
			else:
				result[i] *= np.sum(np.power(features[fs][j] - temp,2))
	median = np.median(result)
	abs_devs = np.abs(result - median)
	mad_sigma = 1.4826 * np.median(abs_devs) + EPSILON # magic number is for gaussians
	# Get maxima.
	times = dt * np.arange(len(abs_devs))
	indices = np.argwhere(abs_devs/mad_sigma>NUM_SIGMA).flatten()[1:-1]
	max_indices = []
	for i in range(2,len(indices)-1):
		if max(abs_devs[indices[i]-1], abs_devs[indices[i]+1]) < abs_devs[indices[i]]:
			max_indices.append(indices[i])
	max_indices = np.array(max_indices, dtype='int')
	max_indices = clean_max_indices(max_indices, times, abs_devs)
	# Collect data for each maximum.
	file_times = [times[index]-TEMPLATE_PREFIX for index in max_indices]
	song_frames = int(fs * SONG_DURATION)
	start_frames = [int(fs * (time - TEMPLATE_PREFIX)) for time in file_times]
	audio_segs = [audio[start:start+song_frames] for start in start_frames]
	file_start_time = time_from_filename(filename)
	times = [file_time + file_start_time for file_time in file_times]
	samplerates = [fs] * len(times)
	d = {
		'filename': [filename]*len(file_times),
		'file_time': file_times,
		'audio': audio_segs,
		'time': times,
		'fs': samplerates,
	}
	return d, features


def clean_collected_data(load_dirs, save_dirs, p, n=10**4):
	"""
	Take a look at the collected data and discard false positives.

	Parameters
	----------

	Notes
	-----

	"""
	# Collect spectrograms.
	filenames = []
	for load_dir in load_dirs:
		filenames += [os.path.join(load_dir, i) for i in os.listdir(load_dir) if is_hdf5_file(i)]
	songs_per_file = p['songs_per_file']
	total_n = len(filenames)*songs_per_file
	n = min(n, total_n)
	if n < total_n:
		indices = np.random.permutation(total_n)[:n].sort()
	else:
		indices = np.arange(total_n)
	specs = []
	prev_filename, f_ind = None, None
	for index in indices:
		filename = filenames[index // songs_per_file]
		file_index = index % songs_per_file
		if filename != prev_filename:
			temp = h5py.File(filename, 'r')
			audio_segs = temp['audio']
			samplerates = temp['fs']
			f_ind = None
		spec, f_ind, _ = get_spec(audio_segs[file_index], f_ind, fs=samplerates[file_index])
		spec = resize(spec, (p['num_freq_bins'], p['num_time_bins']), anti_aliasing=True, mode='reflect')
		specs.append(spec)
	assert len(specs) > 0, "No spectrograms in: "+str(load_dirs)
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
		for x_tick in np.arange(np.ceil(np.min(X)), np.floor(np.max(X))):
			plt.axvline(x=x_tick, c='k', alpha=0.1)
		for y_tick in np.arange(np.ceil(np.min(Y)), np.floor(np.max(Y))):
			plt.axhline(y=y_tick, c='k', alpha=0.1)
		plt.savefig('temp.pdf')
		if i == 0:
			make_html_plot(embedding, specs, num_imgs=10**3)
		bounds['x1s'].append(float(input('x1: ')))
		bounds['x2s'].append(float(input('x2: ')))
		bounds['y1s'].append(float(input('y1: ')))
		bounds['y2s'].append(float(input('y2: ')))
		temp = input('(<c> to continue) ')
		if temp == 'c':
			break
		i += 1
	# Save the good spectrograms.
	for load_dir, save_dir in zip(load_dirs, save_dirs):
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		write_file_num = 0
		filenames = [os.path.join(load_dir, i) for i in os.listdir(load_dir) if is_hdf5_file(i)]
		song_data = {
			'audio': [],
			'time': [],
			'file_time': [],
			'filename': [],
			'fs': [],
		}
		print("Saving song: ")
		for i, filename in enumerate(filenames):
			f = h5py.File(filename, 'r')
			specs = []
			audio_segs = f['audio']
			samplerates = f['fs']
			for i in range(songs_per_file):
				spec, f_ind, _ = get_spec(audio_segs[i], f_ind, fs=samplerates[i])
				spec = resize(spec, (p['num_freq_bins'], p['num_time_bins']), anti_aliasing=True, mode='reflect')
				specs.append(spec)
			specs = np.array(specs).reshape(len(specs), -1)
			embedding = transform.transform(specs)
			bool_list = [in_region(embed, bounds) for embed in embedding]
			print("Songs in "+filename+":", len([j for j in bool_list if j]))
			for j in range(songs_per_file):
				if not bool_list[j]:
					continue
				for key in song_data.keys():
					song_data[key].append(f[key][j])
			while len(song_data['time']) > songs_per_file:
				save_filename = "songs_" + str(write_file_num).zfill(3) + '.hdf5'
				save_filename = os.path.join(save_dir, save_filename)
				song_data = save_data(save_filename, song_data, songs_per_file)
				write_file_num += 1


def clean_max_indices(old_indices, old_times, values, min_dt=0.1):
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


def time_from_filename(filename):
	"""Return time in seconds."""
	try:
		anchor = mktime(strptime("1899 12 29 19", "%Y %m %d %H")) #SAP anchor time
		temp = filename.split('/')[-1].split('_')[1].split('.')
		day = float(temp[0])
		millisecond = float(temp[1])
		time = anchor + 24*60*60*day + 1e-3*millisecond
		return time
	except:
		return 0.0


def in_region(point, bounds):
	"""Is the point in the union of the given rectangles?"""
	for i in range(len(bounds['x1s'])):
		if point[0] > bounds['x1s'][i] and point[0] < bounds['x2s'][i] and \
				point[1] > bounds['y1s'][i] and point[1] < bounds['y2s'][i]:
			return True
	return False


def is_hdf5_file(filename):
	return len(filename) > 5 and filename[-5:] == '.hdf5'


def is_audio_file(filename):
	return len(filename) > 4 and filename[-4:] in ['.wav', '.mat']



if __name__ == '__main__':
	feature_dir = 'data/features/blk215'
	features = get_features(feature_dir)
	result, features = segment_file('temp2.wav', features, feature_dir)
	print(result['time'])
	quit()
	for j in range(1,2):
		result, features = segment_file('temp'+str(j)+'.wav', features, feature_dir)
		print("time", result['time'])



###
