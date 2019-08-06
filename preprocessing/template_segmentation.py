"""
Segment song bouts using linear acoustic feature templates.


TO DO:
	- Improve single syllable use.
	- Align the examplar spectrograms?
"""
__author__ = "Jack Goffinet"
__date__ = "April-May 2019"


import h5py
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy.signal import stft, resample
from scipy.ndimage.filters import gaussian_filter, convolve
from skimage.transform import resize
import os
from time import strptime, mktime
from tqdm import tqdm
import umap

from .interactive_segmentation import make_html_plot


MIN_FREQ, MAX_FREQ = 300, 8e3
FS = 44100
EPSILON = 1e-9
SPEC_THRESH = -4.0

NUM_SIGMA = 3.0




def get_spec(audio, p, fs=FS, norm=False):
	"""Get a spectrogram."""
	f, t, spec = stft(audio, fs=fs)
	spec = np.log(np.abs(spec) + EPSILON)
	spec -= p['spec_thresh']
	# spec -= np.quantile(spec, 0.8)
	spec[spec < 0.0] = 0.0
	# Switch to mel frequency spacing.
	if p['mel']:
		new_f = np.linspace(mel(p['min_freq']), mel(p['max_freq']), p['num_freq_bins'], endpoint=True)
		new_f = inv_mel(new_f)
		new_f[0] = f[0] # Correct for numerical errors.
		new_f[-1] = f[-1]
	else:
		new_f = np.linspace(f[0], f[-1], p['num_freq_bins'], endpoint=True)
	new_spec = np.zeros((p['num_freq_bins'], spec.shape[1]), dtype='float')
	for j in range(spec.shape[1]):
		interp = interp1d(f, spec[:,j], kind='linear')
		new_spec[:,j] = interp(new_f)
	# norm_factor = np.max(new_spec) + EPSILON
	# new_spec = resize(new_spec/norm_factor, (p['num_freq_bins'], p['num_time_bins']), anti_aliasing=True, mode='reflect')
	spec = new_spec
	if norm:
		for j in range(spec.shape[1]):
			spec[:,j] /= np.sum(spec[:,j]) + EPSILON
	return spec, t[1] - t[0]



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
	template = get_template(feature_dir, p)
	for i, filename in enumerate(filenames):
		result, template = segment_file(filename, template, feature_dir, p)
		for key in result:
			song_data[key] += result[key]
		while len(song_data['time']) >= songs_per_file:
			save_filename = "songs_" + str(write_file_num).zfill(3) + '.hdf5'
			save_filename = os.path.join(save_dir, save_filename)
			song_data = save_data(save_filename, song_data, songs_per_file)
			write_file_num += 1


def save_data(save_filename, song_data, songs_per_file):
	# Save things.
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
	# Then delete things that are saved.
	for k in song_data:
		song_data[k] = song_data[k][songs_per_file:]
	return song_data


# def get_templates(feature_dirs):
# 	"""Get multiple templates given multiple template directories."""
# 	return [get_template(i) for i in feature_dirs]


def get_template(feature_dir, p):
	"""
	Create a linear features/templates given exemplar spectrograms.

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
	templates = {}
	for fs in samplerates:
		specs = []
		for i, filename in enumerate(filenames):
			temp_fs, audio = wavfile.read(filename)
			if temp_fs != fs:
				continue
			spec, dt = get_spec(audio, p, fs=fs, norm=True)
			spec = gaussian_filter(spec, (1,1))
			specs.append(spec)
		min_time_bins = min(spec.shape[1] for spec in specs)
		specs = np.array([i[:,:min_time_bins] for i in specs])
		spec = np.mean(specs, axis=0) # Average over all the templates.
		spec -= np.mean(spec)
		spec /= np.std(spec) + EPSILON
		templates[fs] = spec
	return templates


def segment_file(filename, features, feature_dir, p):
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
	assert fs in features, "could not find fs="+str(fs)+" in <features>!"
	big_spec, dt = get_spec(audio, p, fs=fs, norm=True)
	spec_len = features[fs].shape[1]
	# Compute normalized cross-correlation
	result = np.zeros(big_spec.shape[1] - spec_len)
	for i in range(len(result)):
		temp = big_spec[:, i:i+spec_len]
		result[i] = np.sum(np.power(features[fs] - temp,2))
		# result[i] = np.sum(features[fs] * temp)
	median = np.median(result)
	devs = result - median
	abs_devs = np.abs(devs)
	mad_sigma = 1.4826 * np.median(abs_devs) + EPSILON # magic number is for gaussians
	plt.plot(dt * np.arange(len(devs)), devs)
	plt.title(filename)
	plt.savefig('temp.pdf')
	quit()
	# Get maxima.
	times = dt * np.arange(len(abs_devs))
	indices = np.argwhere(-devs/mad_sigma>NUM_SIGMA).flatten()[1:-1]
	max_indices = []
	for i in range(2,len(indices)-1):
		if max(abs_devs[indices[i]-1], abs_devs[indices[i]+1]) < abs_devs[indices[i]]:
			max_indices.append(indices[i])
	max_indices = np.array(max_indices, dtype='int')
	max_indices = clean_max_indices(max_indices, times, abs_devs)
	# Collect data for each maximum.
	file_times = [times[index] for index in max_indices]
	song_frames = int(fs * dt * (spec_len + 1))
	start_frames = [int(fs * (time - dt)) for time in file_times]
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
	if total_n == 0:
		print("No files found in "+str(load_dirs))
		return
	n = min(n, total_n)
	if n < total_n:
		indices = np.random.permutation(total_n)[:n]
		indices.sort()
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
		spec, _ = get_spec(audio_segs[file_index], p, fs=samplerates[file_index])
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
		for x_tick in np.arange(np.floor(np.min(X)), np.ceil(np.max(X))):
			plt.axvline(x=x_tick, c='k', alpha=0.1, lw=0.5)
		for y_tick in np.arange(np.floor(np.min(Y)), np.ceil(np.max(Y))):
			plt.axhline(y=y_tick, c='k', alpha=0.1, lw=0.5)
		title = "Select relevant song:"
		plt.title(title)
		plt.savefig('temp.pdf')
		plt.close('all')
		if i == 0:
			make_html_plot(embedding, specs, num_imgs=10**3, title=title)
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
				spec, _ = get_spec(audio_segs[i], p, fs=samplerates[i])
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


def mel(a):
	return 1127 * np.log(1 + a / 700)


def inv_mel(a):
	return 700 * (np.exp(a / 1127) - 1)


def is_hdf5_file(filename):
	"""Does this filename have an hdf5 extension?"""
	return len(filename) > 5 and filename[-5:] == '.hdf5'


def is_audio_file(filename):
	"""Does this filename have a recognized audio extension?"""
	return len(filename) > 4 and filename[-4:] in ['.wav', '.mat']



if __name__ == '__main__':
	p = {
		'songs_per_file': 20,
		'num_freq_bins': 128,
		'num_time_bins': 128,
		'min_freq': 350,
		'max_freq': 12e3,
		'mel': True,
		'spec_thresh': 1.5,
	}
	feature_dir = 'data/features/blk215'
	features = get_template(feature_dir, p)
	result, features = segment_file('temp2.wav', features, feature_dir, p)
	print(result['time'])
	quit()
	for j in range(1,2):
		result, features = segment_file('temp'+str(j)+'.wav', features, feature_dir, p)
		print("time", result['time'])



###
