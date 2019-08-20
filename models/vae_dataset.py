"""
Data stuff for animal vocalization syllables.

"""
__author__ = "Jack Goffinet"
__date__ = "November 2018 - August 2019"


import h5py
import numpy as np
import os
from scipy.interpolate import interp2d
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.signal import stft
import torch
from torch.utils.data import Dataset, DataLoader

EPSILON = 1e-12


def get_syllable_partition(dirs, split, shuffle=True):
	"""
	Partition the set of syllables into a random test/train split.

	Parameters
	----------
	dirs : list of strings
		List of directories containing saved syllable hdf5 files.

	split : float
		Portion of the hdf5 files to use for training, 0 < split <= 1.0

	shuffle : bool, optional
		Whether to shuffle the hdf5 files. Defaults to True.

	Returns
	-------
	partition : dictionary
		Contains two keys, 'test' and 'train', that map to lists of .hdf5 files.
		Defines the random test/train split.
	"""
	assert(split > 0.0 and split <= 1.0)
	# Collect filenames.
	filenames = []
	for dir in dirs:
		filenames += get_hdf5s_from_dir(dir)
	# Reproducibly shuffle.
	filenames = sorted(filenames)
	if shuffle:
		np.random.seed(42)
		np.random.shuffle(filenames)
		np.random.seed(None)
	# Split.
	index = int(round(split * len(filenames)))
	return {'train': filenames[:index], 'test': filenames[index:]}


def get_window_partition(audio_dirs, roi_dirs, split, roi_extension='.txt', \
	shuffle=True):
	"""

	"""
	assert(split > 0.0 and split <= 1.0)
	# Collect filenames.
	audio_filenames, roi_filenames = [], []
	for audio_dir, roi_dir in zip(audio_dirs, roi_dirs):
		temp = get_wavs_from_dir(audio_dir)
		audio_filenames += temp
		roi_filenames += \
			[os.path.join(roi_dir, os.path.split(i)[-1][:-4]+roi_extension) \
			for i in temp]
	# Reproducibly shuffle.
	audio_filenames = np.array(audio_filenames)
	roi_filenames = np.array(roi_filenames)
	perm = np.argsort(audio_filenames)
	audio_filenames, roi_filenames = audio_filenames[perm], roi_filenames[perm]
	if shuffle:
		np.random.seed(42)
		perm = np.random.permutation(len(audio_filenames))
		audio_filenames = audio_filenames[perm]
		roi_filenames = roi_filenames[perm]
		np.random.seed(None)
	# Split.
	i = int(round(split * len(audio_filenames)))
	return { \
		'train': { \
			'audio': audio_filenames[:i], 'rois': roi_filenames[:i]}, \
		'test': { \
			'audio': audio_filenames[i:], 'rois': roi_filenames[i:]} \
		}


def get_syllable_data_loaders(partition, batch_size=64, shuffle=(True, False), \
	num_workers=3):
	"""
	Return a pair of DataLoaders given a test/train split.

	Parameters
	----------
	partition : dictionary
		Test train split: a dictionary that maps the keys 'test' and 'train'
		to disjoint lists of .hdf5 filenames containing syllables.

	batch_size : int, optional
		Batch size of the returned Dataloaders. Defaults to 32.

	shuffle : tuple of bools, optional
		Whether to shuffle data for the train and test Dataloaders,
		respectively. Defaults to (True, False).

	num_workers : int, optional
		How many subprocesses to use for data loading. Defaults to 3.

	Returns
	-------
	dataloaders : dictionary
		Dictionary mapping two keys, 'test' and 'train', to respective
		torch.utils.data.Dataloader objects.
	"""
	sylls_per_file = get_sylls_per_file(partition)
	train_dataset = SyllableDataset(filenames=partition['train'], \
		transform=numpy_to_tensor, sylls_per_file=sylls_per_file)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, \
		shuffle=shuffle[0], num_workers=num_workers)
	if not partition['test']:
		return {'train':train_dataloader, 'test':None}
	test_dataset = SyllableDataset(filenames=partition['test'], \
		transform=numpy_to_tensor, sylls_per_file=sylls_per_file)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, \
		shuffle=shuffle[1], num_workers=num_workers)
	return {'train':train_dataloader, 'test':test_dataloader}


def get_window_data_loaders(partition, p, batch_size=64, shuffle=(True, False),\
	num_workers=3):
	"""

	"""
	train_dataset = WindowDataset(partition['train']['audio'], \
		partition['train']['rois'], p, transform=numpy_to_tensor)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, \
		shuffle=shuffle[0], num_workers=num_workers)
	if not partition['test']:
		return {'train':train_dataloader, 'test':None}
	test_dataset = WindowDataset(partition['test']['audio'], \
		partition['test']['rois'], p, transform=numpy_to_tensor)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, \
		shuffle=shuffle[1], num_workers=num_workers)
	return {'train':train_dataloader, 'test':test_dataloader}


def get_warped_window_data_loaders(audio_dirs, template_dir, p, batch_size=64, \
	num_workers=3):
	"""
	Audio files must all be the same duration!
	"""
	audio_fns = []
	for audio_dir in audio_dirs:
		audio_fns += [os.path.join(audio_dir, i) for i in \
			os.listdir(audio_dir) if i[-4:] == '.wav']
	dataset = WarpedWindowDataset(audio_fns, template_dir, p, \
		transform=numpy_to_tensor)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, \
		num_workers=num_workers)
	return {'train': dataloader, 'test': dataloader}


class SyllableDataset(Dataset):
	"""torch.utils.data.Dataset for animal vocalization syllables"""

	def __init__(self, filenames, sylls_per_file, transform=None):
		"""
		Create a torch.utils.data.Dataset for animal vocalization syllables.

		Parameters
		----------
		filenames : list of strings
			List of hdf5 files containing syllable spectrograms.

		sylls_per_file : int
			Number of syllables in each hdf5 file.

		transform : None or function, optional
			Transformation to apply to each item. Defaults to None (no
			transformation)
		"""
		self.filenames = filenames
		self.sylls_per_file = sylls_per_file
		self.transform = transform

	def __len__(self):
		return len(self.filenames) * self.sylls_per_file

	def __getitem__(self, index):
		result = []
		single_index = False
		try:
			iterator = iter(index)
		except TypeError:
			index = [index]
			single_index = True
		for i in index:
			# First find the file.
			load_filename = self.filenames[i // self.sylls_per_file]
			file_index = i % self.sylls_per_file
			# Then collect fields from the file.
			with h5py.File(load_filename, 'r') as f:
				try:
					spec = f['specs'][file_index]
				except:
					print(file_index, self.sylls_per_file)
					print(i // self.sylls_per_file, len(self.filenames))
					print(len(f['specs']))
					print(load_filename)
					quit()
			if self.transform:
				spec = self.transform(spec)
			result.append(spec)
		if single_index:
			return result[0]
		return result



class WindowDataset(Dataset):
	"""torch.utils.data.Dataset for chunks of animal vocalization"""

	def __init__(self, audio_filenames, roi_filenames, p, transform=None,
		dataset_length=2000):
		"""
		Create a torch.utils.data.Dataset for chunks of animal vocalization.

		Parameters
		----------
		audio_filenames : list of strings
			List of wav files.

		roi_filenames : list of strings
			List of files containing animal vocalization times. Format: ...

		transform : None or function, optional
			Transformation to apply to each item. Defaults to None (no
			transformation)
		"""
		self.audio = [wavfile.read(fn)[1] for fn in audio_filenames]
		self.fs = wavfile.read(audio_filenames[0])[0]
		self.roi_filenames = roi_filenames
		self.dataset_length = dataset_length
		self.p = p
		self.filenames = np.array(audio_filenames)
		self.rois = [np.loadtxt(i, ndmin=2) for i in roi_filenames]
		self.file_weights = np.array([np.sum(np.diff(i)) for i in self.rois])
		self.file_weights /= np.sum(self.file_weights)
		self.roi_weights = []
		for i in range(len(self.rois)):
			temp = np.diff(self.rois[i]).flatten()
			self.roi_weights.append(temp/np.sum(temp))
		self.transform = transform


	def __len__(self):
		return self.dataset_length


	def __getitem__(self, index, seed=None):
		result = []
		single_index = False
		try:
			iterator = iter(index)
		except TypeError:
			index = [index]
			single_index = True
		np.random.seed(seed)
		for i in index:
			while True:
				# First find the file, then the ROI.
				file_index = np.random.choice(np.arange(len(self.filenames)), \
					p=self.file_weights)
				load_filename = self.filenames[file_index]
				roi_index = \
					np.random.choice(np.arange(len(self.roi_weights[file_index])),
					p=self.roi_weights[file_index])
				roi = self.rois[file_index][roi_index]
				# Then choose a chunk of audio uniformly at random.
				start_t = roi[0] + (roi[1] - roi[0] - self.p['window_length']) \
					* np.random.rand()
				end_t = start_t + self.p['window_length']
				# Then make a spectrogram.
				spec, flag = self.p['get_spec'](start_t, end_t, \
					self.audio[file_index], self.p, fs=self.fs)
				if not flag:
					continue
				if self.transform:
					spec = self.transform(spec)
				result.append(spec)
				break
		np.random.seed(None)
		if single_index:
			return result[0]
		return result


	def write_hdf5_files(self, save_dir, num_files=300, sylls_per_file=100):
		"""
		Write hdf5 files containing spectrograms of random audio chunks.

		NOTE
		----
	 	This should be consistent with
		preprocessing.preprocessing.process_sylls.
		"""
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)
		for write_file_num in range(num_files):
			specs = self.__getitem__(np.arange(sylls_per_file),
				seed=write_file_num)
			specs = np.array([spec.detach().numpy() for spec in specs])
			fn = "syllables_" + str(write_file_num).zfill(4) + '.hdf5'
			fn = os.path.join(save_dir, fn)
			with h5py.File(fn, "w") as f:
				f.create_dataset('specs', data=specs)



class WarpedWindowDataset(Dataset):
	"""torch.utils.data.Dataset for chunks of animal vocalization"""

	def __init__(self, audio_filenames, template_dir, p, \
		transform=None, dataset_length=2000):
		"""
		Create a torch.utils.data.Dataset for chunks of animal vocalization.

		Parameters
		----------
		audio_filenames : list of strings
			List of wav files.

		roi_filenames : list of strings
			List of files containing animal vocalization times. Format: ...

		transform : None or function, optional
			Transformation to apply to each item. Defaults to None (no
			transformation)
		"""
		self.audio_filenames = audio_filenames
		self.audio = [wavfile.read(fn)[1] for fn in audio_filenames]
		self.fs = wavfile.read(audio_filenames[0])[0]
		self.dataset_length = dataset_length
		self.p = p
		self.transform = transform
		self.linear_warp(template_dir)


	def __len__(self):
		"""NOTE: length is arbitrary."""
		return self.dataset_length


	def write_hdf5_files(self, save_dir, num_files=400, sylls_per_file=100):
		"""
		Write hdf5 files containing spectrograms of random audio chunks.

		NOTE
		----
	 	This should be consistent with
		preprocessing.preprocessing.process_sylls.
		"""
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)
		for write_file_num in range(num_files):
			specs = self.__getitem__(np.arange(sylls_per_file),
				seed=write_file_num)
			specs = np.array([spec.detach().numpy() for spec in specs])
			fn = "sylls_" + str(write_file_num).zfill(4) + '.hdf5'
			fn = os.path.join(save_dir, fn)
			with h5py.File(fn, "w") as f:
				f.create_dataset('specs', data=specs)


	def get_template(self, feature_dir):
		"""Adapted from segmentation/template_segmentation_v2.py"""
		filenames = [os.path.join(feature_dir, i) for i in os.listdir(feature_dir) \
			if is_wav_file(i)]
		specs = []
		for i, filename in enumerate(filenames):
			fs, audio = wavfile.read(filename)
			assert fs == self.fs, "Found samplerate="+str(fs)+\
				", expected "+str(self.fs)
			spec, dt = self.get_spec(audio)
			spec = gaussian_filter(spec, (0.5,0.5))
			specs.append(spec)
		min_time_bins = min(spec.shape[1] for spec in specs)
		specs = np.array([i[:,:min_time_bins] for i in specs])
		template = np.mean(specs, axis=0) # Average over all the templates.
		self.template_dur = template.shape[1]*dt
		return template


	def get_spec(self, audio, target_ts=None):
		"""Not many options here."""
		f, t, spec = stft(audio, fs=self.fs, nperseg=self.p['nperseg'], \
			noverlap=self.p['noverlap'])
		i1 = np.searchsorted(f, self.p['min_freq'])
		i2 = np.searchsorted(f, self.p['max_freq'])
		spec = spec[i1:i2]
		f = f[i1:i2]
		spec = np.log(np.abs(spec) + EPSILON)
		if target_ts is not None:
			interp = interp2d(t, f, spec, copy=False, bounds_error=False, \
				fill_value=self.p['spec_min_val'])
			interp_spec = interp(target_ts, f, assume_sorted=True)
			spec = interp_spec
		spec -= self.p['spec_min_val']
		spec /= self.p['spec_max_val'] - self.p['spec_min_val'] + EPSILON
		spec = np.clip(spec, 0.0, 1.0)
		return spec, t[1]-t[0]


	def make_loss_func(self, template, audio):
		"""Make a time warping loss function."""
		def loss_func(coeffs):
			target_ts = coeffs[0] + coeffs[1] * \
				np.linspace(0,self.template_dur,template.shape[1],endpoint=True)
			try:
				predicted, _ = self.get_spec(audio, target_ts=target_ts)
			except:
				print(type(audio), len(audio))
				print(type(target_ts), len(target_ts))
				print(coeffs)
				print(target_ts)
				quit()
			return np.sum(np.power(template - predicted, 2))
		return loss_func


	def linear_warp(self, template_dir):
		"""Linearly warp each song rendition to the template."""
		template = self.get_template(template_dir)
		self.coeffs = np.zeros((len(self.audio), 2))
		# warps = {}
		# for i in range(len(self.audio)):
		# 	loss_func = self.make_loss_func(template, self.audio[i])
		# 	x0 = np.array([0.0,1.0])
		# 	res = minimize(loss_func, x0, bounds=[(-.1,.1),(.6,1.2)])
		# 	# print(res.x)
		# 	self.coeffs[i] = res.x[:]
		# 	warps[self.audio_filenames[i]] = self.coeffs[i]
		# np.save('warps.npy', warps)
		warps = np.load('warps.npy', allow_pickle=True).item()
		for i in range(len(self.audio)):
			self.coeffs[i] = warps[self.audio_filenames[i]]


	def __getitem__(self, index, seed=None):
		result = []
		single_index = False
		try:
			iterator = iter(index)
		except TypeError:
			index = [index]
			single_index = True
		np.random.seed(seed)
		for i in index:
			while True:
				# First find the file, then the ROI.
				file_index = np.random.randint(len(self.audio))
				# Then choose a chunk of audio uniformly at random.
				start_t = np.random.rand() * (self.template_dur - self.p['window_length'])
				stop_t = start_t + self.p['window_length']
				# Warp to the current audio chunk.
				start_t = self.coeffs[file_index,0] + self.coeffs[file_index,1] * start_t
				stop_t = self.coeffs[file_index,0] + self.coeffs[file_index,1] * stop_t
				# Then make a spectrogram.
				spec, flag = self.p['get_spec'](start_t, stop_t, self.audio[file_index], self.p, fs=self.fs)
				if not flag:
					continue
				if self.transform:
					spec = self.transform(spec)
				result.append(spec)
				break
		np.random.seed(None)
		if single_index:
			return result[0]
		return result


def get_sylls_per_file(partition):
	"""Open an hdf5 file and see how many syllables it has."""
	key = 'train' if len(partition['train']) > 0 else 'test'
	assert len(partition[key]) > 0
	filename = partition[key][0] # Just grab the first file.
	with h5py.File(filename, 'r') as f:
		sylls_per_file = len(f['specs'])
	return sylls_per_file


def numpy_to_tensor(x):
	"""Transform a numpy array into a torch.FloatTensor"""
	return torch.from_numpy(x).type(torch.FloatTensor)


def get_hdf5s_from_dir(dir):
	"""
	Return a sorted list of all hdf5s in a directory.

	Note
	----
	plotting.data_container relies on this.
	"""
	return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if \
		is_hdf5_file(f)]


def get_wavs_from_dir(dir):
	return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if \
		is_wav_file(f)]


def is_hdf5_file(filename):
	"""Is the given filename an hdf5 file?"""
	return len(filename) > 5 and filename[-5:] == '.hdf5'


def is_wav_file(filename):
	return len(filename) > 4 and filename[-4:] == '.wav'



if __name__ == '__main__':
	pass


###
