"""
Segment song bouts using linear acoustic feature templates.

"""
__author__ = "Jack Goffinet"
__date__ = "April 2019"

import os

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from scipy.ndimage.filters import gaussian_filter, convolve

import umap
from tqdm import tqdm

import matplotlib.pyplot as plt
# plt.switch_backend('agg')

from interactive_segmentation import make_html_plot


MIN_FREQ, MAX_FREQ = 300, 8e3
FS = 44100
EPSILON = 1e-9
SPEC_THRESH = -5.5

# MIN_SEP = 0.35
NUM_SPLITS = 1

SONG_PREFIX = 0.05
SONG_DURATION = 0.65



def get_spec(audio, f_ind):
	f, t, Zxx = stft(audio, fs=FS)
	if f_ind is None:
		f_ind = np.searchsorted(f, [MIN_FREQ, MAX_FREQ])
	Zxx = Zxx[f_ind[0]:f_ind[1]]
	Zxx = np.log(np.abs(Zxx) + EPSILON)
	Zxx -= SPEC_THRESH
	Zxx[Zxx < 0.0] = 0.0
	Zxx /= np.max(Zxx) + EPSILON
	return Zxx, f_ind, t[1] - t[0]


def process_sylls(load_dir, save_dir, p):
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
	}
	songs_per_file = p['songs_per_file']
	print("Processing audio files in", load_dir)
	features = get_features()
	f_ind = None
	for i, filename in enumerate(filenames):
		result = segment_file(filename, features, f_ind)
		for key in result:
			song_data[key].append(result[key])
		while len(song_data['time']) >= songs_per_file:
			save_filename = "songs_" + str(write_file_num).zfill(3) + '.hdf5'
			save_filename = os.path.join(save_dir, save_filename)
			with h5py.File(save_filename, "w") as f:
				for key in song_data.keys().remove('filename'):
					f.create_dataset(key, data=np.array(song_data[key][:songs_per_file]))
				f.create_dataset('filename', data=np.array(song_data['filename']).astype('S'))
				f['audio'].attrs['fs'] = FS
			for k in song_data:
				song_data[k] = song_data[k][songs_per_file:]
			write_file_num += 1



def get_features():
	"""
	Create linear features given exemplar spectrograms.

	TO DO: add a load directory, etc.
	"""
	f_ind = None
	specs = []
	min_time_bins = 10**10
	for i in range(7):
		fs, audio = wavfile.read(str(i).zfill(2)+'.wav')
		assert fs == FS
		spec, f_ind, dt = get_spec(audio, f_ind)
		spec = gaussian_filter(spec, (2,2))
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
		feature -= np.mean(feature, axis=0)
		feature /= np.std(feature, axis=0)
		# feature /= np.sqrt(np.sum(np.power(feature, 2)))
		features.append(feature)
	return features, f_ind


def segment_file(filename, features, f_ind):
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
	assert fs == FS
	big_spec, _, dt = get_spec(audio, f_ind)
	spec_len = features[0].shape[1]
	# Compute normalized cross-correlation
	result = np.zeros(big_spec.shape[1]-NUM_SPLITS*spec_len)
	for i in range(len(result)):
		for j in range(NUM_SPLITS):
			temp = big_spec[:,i+j*spec_len:i+(j+1)*spec_len]
			temp -= np.mean(temp, axis=0)
			# temp /= np.sqrt(np.sum(np.power(temp, 2)))
			temp /= np.std(temp, axis=0)
			if j == 0:
				result[i] = np.sum(np.power(features[j] - temp,2))
			else:
				result[i] *= np.sum(np.power(features[j] - temp,2))
	median = np.median(result)
	abs_devs = np.abs(result - median)
	mad_sigma = 1.4826 * np.median(abs_devs) # assumes normally-distributed data
	# Get maxima.
	times = dt * np.arange(len(abs_devs))
	indices = np.argwhere(abs_devs>8*mad_sigma).flatten()[1:-1]
	max_indices = []
	for i in range(2,len(indices)-1):
		if max(abs_devs[indices[i]-1], abs_devs[indices[i]+1]) < abs_devs[indices[i]]:
			max_indices.append(indices[i])
	max_indices = np.array(max_indices, dtype='int')
	# Collect data for each maximum.
	file_times = [times[index]-SONG_PREFIX for index in max_indices]
	song_frames = int(FS * SONG_DURATION)
	start_frames = [int(FS * (time - SONG_PREFIX)) for time in file_times]
	audio_segs = [audio[start:start+song_frames] for start in start_frames]
	file_start_time = time_from_filename(filename)
	times = [file_time + file_start_time for file_time in file_times]
	d = {
		'filename': [filename]*len(file_times),
		'file_time': file_times,
		'audio': audio_segs,
		'time': times,
	}
	return d


def clean_collected_data(save_dirs, p, n=10**4):
	"""
	Take a look at the collected data and discard false positives.
	"""
	# Collect spectrograms.
	filenames = []
	for save_dir in save_dirs:
		filenames += [os.path.join(save_dir, i) for i in os.listdir(save_dir) if is_hdf5_file(i)]
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
			audio_segs = h5py.File(filename, 'r')['audio']
			assert audio_segs.attrs['fs'] == FS
		spec, f_ind, _ = get_spec(audio_segs[file_index], f_ind)
		specs.append(spec)
	specs = np.array(specs)
	# UMAP the spectrograms.
	transform = umap.UMAP()
	embedding = transform.fit_transform(specs.reshape(len(specs), -1))
	# Plot and ask for user input.
	bounds = {
		'x1s':[],
		'x2s':[],
		'y1s':[],
		'y2s':[],
	}
	X, Y = embedding[:,0], embedding[:,1]
	while True:
		colors = ['b' if in_region(embed, bounds) else 'r' for embed in embedding]
		plt.scatter(X, Y, c='b', s=0.9, alpha=0.5)
		for x_tick in np.arange(np.ceil(np.min(X)), np.floor(np.max(X))):
			plt.axvline(x=x_tick, c='k', alpha=0.1)
		for y_tick in np.arange(np.ceil(np.min(Y)), np.floor(np.max(Y))):
			plt.axvline(y=y_tick, c='k', alpha=0.1)
		plt.savefig('temp.pdf')
		make_html_plot(embedding, specs, num_imgs=10**3)
		x1 = input('x1: (<q> to quit)')
		if x1 == 'q':
			break
		bounds['x1s'].append(float(x1))
		bounds['x2s'].append(float(input('x2: ')))
		bounds['y1s'].append(float(input('y1: ')))
		bounds['y2s'].append(float(input('y2: ')))
	# Save the good spectrograms.
	# NOTE: HERE



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
	features, f_ind = get_features()
	for j in range(1,8):
		segment_file('temp'+str(j)+'.wav', features, f_ind)










###
