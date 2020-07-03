"""
Useful functions for feeding data to the shotgun VAE.

"""
__date__ = "August 2019 - July 2020"


from affinewarp import PiecewiseWarping
import h5py
import numpy as np
import os
from scipy.interpolate import interp1d, interp2d
from scipy.io import wavfile
from scipy.signal import stft
import torch
from torch.utils.data import Dataset, DataLoader
import warnings


DEFAULT_WARP_PARAMS = {
	'n_knots': 0,
	'warp_reg_scale':1e-6,
	'smoothness_reg_scale':20.0,
}
EPSILON = 1e-9



def get_window_partition(audio_dirs, roi_dirs, split=0.8, shuffle=True):
	"""
	Get a train/test split for fixed-duration shotgun VAE.

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	roi_dirs : list of str
		ROI (segment) directories.
	split : float, optional
		Train/test split. Defaults to ``0.8``, indicating an 80/20 train/test
		split.
	shuffle : bool, optional
		Whether to shuffle at the audio file level. Defaults to ``True``.

	Returns
	-------
	partition : dict
		Defines the test/train split. The keys ``'test'`` and ``'train'`` each
		map to a dictionary with keys ``'audio'`` and ``'rois'``, which both
		map to numpy arrays containing filenames.
	"""
	assert(split > 0.0 and split <= 1.0)
	# Collect filenames.
	audio_filenames, roi_filenames = [], []
	for audio_dir, roi_dir in zip(audio_dirs, roi_dirs):
		temp = _get_wavs_from_dir(audio_dir)
		audio_filenames += temp
		roi_filenames += \
			[os.path.join(roi_dir, os.path.split(i)[-1][:-4]+'.txt') \
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


def get_fixed_window_data_loaders(partition, p, batch_size=64, \
	shuffle=(True, False), num_workers=4):
	"""
	Get DataLoaders for training and testing: fixed-duration shotgun VAE

	Parameters
	----------
	partition : dict
		Output of ``ava.models.window_vae_dataset.get_window_partition``.
	p : dict
		Preprocessing parameters. Must contain keys: ...
	batch_size : int, optional
		Defaults to ``64``.
	shuffle : tuple of bool, optional
		Whether to shuffle train and test sets, respectively. Defaults to
		``(True, False)``.
	num_workers : int, optional
		Number of CPU workers to feed data to the network. Defaults to ``4``.

	Returns
	-------
	loaders : dict
		Maps the keys ``'train'`` and ``'test'`` to their respective
		DataLoaders.
	"""
	train_dataset = FixedWindowDataset(partition['train']['audio'], \
		partition['train']['rois'], p, transform=numpy_to_tensor)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, \
		shuffle=shuffle[0], num_workers=num_workers)
	if not partition['test']:
		return {'train':train_dataloader, 'test':None}
	test_dataset = FixedWindowDataset(partition['test']['audio'], \
		partition['test']['rois'], p, transform=numpy_to_tensor)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, \
		shuffle=shuffle[1], num_workers=num_workers)
	return {'train':train_dataloader, 'test':test_dataloader}



class FixedWindowDataset(Dataset):

	def __init__(self, audio_filenames, roi_filenames, p, transform=None,
		dataset_length=2048):
		"""
		Create a torch.utils.data.Dataset for chunks of animal vocalization.

		Parameters
		----------
		audio_filenames : list of str
			List of wav files.
		roi_filenames : list of str
			List of files containing animal vocalization times.
		transform : {``None``, function}, optional
			Transformation to apply to each item. Defaults to ``None`` (no
			transformation).
		dataset_length : int, optional
			Arbitrary number that determines batch size. Defaults to ``2048``.
		"""
		self.filenames = np.array(sorted(audio_filenames))
		self.audio = [wavfile.read(fn)[1] for fn in self.filenames]
		self.fs = wavfile.read(audio_filenames[0])[0]
		self.roi_filenames = roi_filenames
		self.dataset_length = dataset_length
		self.p = p
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


	def __getitem__(self, index, seed=None, shoulder=0.05):
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
				onset = roi[0] + (roi[1] - roi[0] - self.p['window_length']) \
					* np.random.rand()
				offset = onset + self.p['window_length']
				target_times = np.linspace(onset, offset, \
						self.p['num_time_bins'])
				# Then make a spectrogram.
				spec, flag = self.p['get_spec'](max(0.0, onset-shoulder), \
						offset+shoulder, self.audio[file_index], self.p, \
						fs=self.fs, target_times=target_times)
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


	def write_hdf5_files(self, save_dir, num_files=500, sylls_per_file=100):
		"""
		Write hdf5 files containing spectrograms of random audio chunks.

		TO DO
		-----
		* Write to multiple directories.

		Note
		----
	 	This should be consistent with
		ava.preprocessing.preprocess.process_sylls.

		Parameters
		----------
		save_dir : str
			Directory to save hdf5s in.
		num_files : int, optional
			Number of files to save. Defaults to ``500``.
		sylls_per_file : int, optional
			Number of syllables in each file. Defaults to ``100``.
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



def get_warped_window_data_loaders(audio_dirs, p, batch_size=64, num_workers=4,\
	load_warp=False, warp_fn=None, warp_params={}, warp_type='spectrogram'):
	"""
	Get DataLoaders for training and testing: warped shotgun VAE

	Warning
	-------
	- Audio files must all be the same duration! You can use
	  `segmenting.utils.write_segments_to_audio` to extract audio from song
	  segments, writing them as separate ``.wav`` files.

	TO DO
	-----
	* Add a train/test split!

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	p : dict
		Preprocessing parameters. Must contain keys: ``'window_length'``,
		``'nperseg'``, ``'noverlap'``, ``'min_freq'``, ``'max_freq'``,
		``'spec_min_val'``, and ``'spec_max_val'``.
	batch_size : int, optional
		DataLoader batch size. Defaults to ``64``.
	num_workers : int, optional
		Number of CPU workers to retrieve data for the model. Defaults to ``4``.
	load_warp : bool, optional
		Whether to load a previously saved time warping result. Defaults to
		``False``.
	warp_fn : {str, None}, optional
		Where the x-knots and y-knots should be saved and loaded. Defaults to
		``None``.
	warp_params : dict, optional
		Parameters passed to affinewarp. Defaults to ``{}``.
	warp_type : {``'amplitude'``, ``'spectrogram'``}, optional
		Whether to time-warp using ampltidue traces or full spectrograms.
		Defaults to ``'spectrogram'``.

	Returns
	-------
	loaders : dict
		Maps the keys ``'train'`` and ``'test'`` to their respective
		DataLoaders.
	"""
	assert type(p) == type({})
	assert warp_type in ['amplitude', 'spectrogram']
	# Collect audio filenames.
	audio_fns = []
	for audio_dir in audio_dirs:
		audio_fns += [os.path.join(audio_dir, i) for i in os.listdir(audio_dir)\
				if _is_wav_file(i)]
	# Make the Dataset and DataLoader.
	dataset = WarpedWindowDataset(audio_fns, p, \
		transform=numpy_to_tensor, load_warp=load_warp, warp_fn=warp_fn, \
		warp_params=warp_params, warp_type=warp_type)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, \
		num_workers=num_workers)
	return {'train': dataloader, 'test': dataloader}



class WarpedWindowDataset(Dataset):

	def __init__(self, audio_filenames, p, transform=None, dataset_length=2048,\
		load_warp=False, save_warp=True, start_q=-0.1, stop_q=1.1, \
		warp_fn=None, warp_params={}, warp_type='spectrogram'):
		"""
		Dataset for time-warped chunks of animal vocalization

		TO DO
		-----
		* Use affinewarp functions instead of direct references to knots.

		Parameters
		----------
		audio_filenames : list of strings
			List of .wav files.
		p : dict
			Preprocessing parameters. Must contain keys: ``'window_length'``,
			``'nperseg'``, ``'noverlap'``, ``'min_freq'``, ``'max_freq'``,
			``'spec_min_val'``, and ``'spec_max_val'``.
		transform : {None, function}, optional
			Transformation to apply to each item. Defaults to ``None`` (no
			transformation).
		dataset_length : int, optional
			Defaults to ``2048``. This is an arbitrary number that determines
			how many batches make up an epoch.
		load_warp : bool, optional
			Whether to load the results of a previous warp. Defaults to
			``False``.
		save_warp : bool, optional
			Whether to save the results of the warp. Defaults to ``True``.
		start_q : float, optional
			Start quantile. Defaults to ``-0.1``.
		stop_q : float, optional
			Stop quantile. Defaults to ``1.1``.
		warp_fn : {None, str}, optional
			Where to save the x knots and y knots of the warp. If ``None``, then
			nothing will be saved or loaded. Defaults to ``None``.
		warp_params : dict, optional
			Parameters passed to affinewarp. Defaults to ``{}``.
		warp_type : {``'amplitude'``, ``'spectrogram'``}, optional
			Whether to time-warp using ampltidue traces or full spectrograms.
			Defaults to ``'spectrogram'``.
		"""
		assert type(p) == type({})
		assert warp_type in ['amplitude', 'spectrogram']
		self.audio_filenames = sorted(audio_filenames)
		self.audio = [wavfile.read(fn)[1] for fn in self.audio_filenames]
		self.fs = wavfile.read(self.audio_filenames[0])[0]
		self.dataset_length = dataset_length
		self.p = p
		self.transform = transform
		self.start_q = start_q
		self.stop_q = stop_q
		self.warp_fn = warp_fn
		self.warp_params = {**DEFAULT_WARP_PARAMS, **warp_params}
		self._compute_warp(load_warp=load_warp, save_warp=save_warp, \
				warp_type=warp_type)
		self.window_frac = self.p['window_length'] / self.template_dur


	def __len__(self):
		"""NOTE: length is arbitrary."""
		return self.dataset_length


	def write_hdf5_files(self, save_dir, num_files=400, sylls_per_file=100):
		"""
		Write hdf5 files containing spectrograms of random audio chunks.

		Note
		----
	 	This should be consistent with
		``ava.preprocessing.preprocess.process_sylls``.

		Parameters
		----------
		save_dir : str
			Where to write.
		num_files : int, optional
			Number of files to write. Defaults to `400`.
		sylls_per_file : int, optional
			Number of spectrograms to write per file. Defaults to `100`.
		"""
		if save_dir != '' and not os.path.exists(save_dir):
			os.mkdir(save_dir)
		for write_file_num in range(num_files):
			specs = self.__getitem__(np.arange(sylls_per_file),
				seed=write_file_num)
			specs = np.array([spec.detach().numpy() for spec in specs])
			fn = "sylls_" + str(write_file_num).zfill(4) + '.hdf5'
			fn = os.path.join(save_dir, fn)
			with h5py.File(fn, "w") as f:
				f.create_dataset('specs', data=specs)


	def _get_spec(self, audio, target_ts=None):
		"""Make a basic spectrogram."""
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


	def _get_unwarped_times(self, y_vals, index):
		"""
		Convert warped quantile times in [0,1] to real quantile times.

		Assumes y_vals is sorted.
		"""
		x_knots, y_knots = self.x_knots[index], self.y_knots[index]
		interp = interp1d(y_knots, x_knots, bounds_error=False, \
				fill_value='extrapolate', assume_sorted=True)
		x_vals = interp(y_vals)
		return x_vals


	def _compute_warp(self, load_warp=False, save_warp=True, \
		warp_type='spectrogram'):
		"""
		Jointly warp all the song renditions.

		Warping is performed on spectrograms if ``warp_type == 'spectrogram'``.
		Otherwise, if ``warp_type == 'amplitude'``, warping is performed on
		spectrograms summed over the frequency dimension.
		"""
		# Load warps if we can.
		if load_warp:
			if self.warp_fn is None:
				warnings.warn(
					"Tried to load warps, but ``warp_fns`` is None.",
					UserWarning
				)
			else:
				try:
					data = np.load(self.warp_fn, allow_pickle=True).item()
					self.x_knots = data['x_knots']
					self.y_knots = data['y_knots']
					self.template_dur = data['template_dur']
					temp_fns = data['audio_filenames']
					assert np.all(temp_fns[:-1] <= temp_fns[1:]), "Filenames "+\
							"in " + self.warp_fn + " are not sorted!"
					assert len(temp_fns) >= len(self.audio_filenames)
					if len(temp_fns) == len(self.audio_filenames):
						# If the saved filenames and the passed filenames have
						# the same length, make sure they match.
						assert np.array_equal(temp_fns, self.audio_filenames), \
								"Input filenames do not match saved filenames!"
					else:
						# Otherwise, make sure the passed filenames are a subset
						# of the saved filenames and keep track of the correct
						# indices.
						unique_fns = np.unique(self.audio_filenames)
						assert len(self.audio_filenames) == len(unique_fns)
						perm = np.zeros(len(self.audio_filenames), dtype='int')
						for i in range(len(self.audio_filenames)):
							assert self.audio_filenames[i] in temp_fns, \
									"Could not find filename " + \
									self.audio_filenames[i] + " in saved warps!"
							index = temp_fns.index(self.audio_filenames[i])
							perm[i] = index
						self.x_knots = self.x_knots[perm]
						self.y_knots = self.y_knots[perm]
					if type(self.audio_filenames) == type(np.array([])):
						self.audio_filenames = self.audio_filenames.tolist()
					self.warp_params = data['warp_params']
					return
				except IOError:
					warnings.warn(
						"Can't load warps from: "+str(self.warp_fn),
						UserWarning
					)
		if save_warp:
			assert self.warp_fn is not None, "``warp_fn`` must be specified " +\
					"to save warps!"
		# Otherwise, make the warps.
		specs = []
		for i in range(len(self.audio)):
			spec, dt = self._get_spec(self.audio[i])
			specs.append(spec.T)
		# Check to make sure everything's the same shape.
		assert len(specs) > 0
		min_time_bins = min(spec.shape[0] for spec in specs)
		specs = [spec[:min_time_bins] for spec in specs]
		min_freq_bins = min(spec.shape[1] for spec in specs)
		specs = [spec[:,:min_freq_bins] for spec in specs]
		self.num_time_bins = specs[0].shape[0]
		assert self.num_time_bins == min_time_bins
		self.template_dur = self.num_time_bins * dt
		# Compute amplitude traces.
		amps = []
		for i in range(len(self.audio)):
			amp_trace = np.sum(specs[i], axis=-1, keepdims=True)
			amp_trace -= np.min(amp_trace)
			amp_trace /= np.max(amp_trace) + EPSILON
			amps.append(amp_trace)
		amps = np.stack(amps)
		specs = np.stack(specs)
		# Warp.
		model = PiecewiseWarping(**self.warp_params)
		if warp_type == 'amplitude':
			print("Computing amplitude warp:", amps.shape)
			model.fit(amps, iterations=50, warp_iterations=200)
		elif warp_type == 'spectrogram':
			print("Computing spectrogram warp:", specs.shape)
			model.fit(specs, iterations=50, warp_iterations=200)
		else:
			raise NotImplementedError
		# Save the warps.
		self.x_knots = model.x_knots
		self.y_knots = model.y_knots
		if save_warp:
			print("Saving warp to:", self.warp_fn)
			to_save = {
				'x_knots' : self.x_knots,
				'y_knots' : self.y_knots,
				'template_dur' : self.template_dur,
				'audio_filenames' : self.audio_filenames,
				'amplitude_traces': amps,
				'warp_params': self.warp_params,
			}
			np.save(self.warp_fn, to_save)


	def __getitem__(self, index, seed=None):
		"""
		Return a random window of birdsong.

		Parameters
		----------
		index : {int, list of int}
			Determines the number of spectrograms to return. If an int is
			passed, a single spectrogram is returned. If a list is passed,
			``len(index)`` spectrograms are returned. Elements (ints)
			themselves are ignored.
		seed : {None, int}, optional
			Random seed

		Returns
		-------
		spec : {numpy.ndarray, list of numpy.ndarray}
			Spectrograms
		"""
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
				start_t = self.start_q + np.random.rand() * \
						(self.stop_q - self.start_q - self.window_frac)
				stop_t = start_t + self.window_frac
				t_vals = np.linspace(start_t, stop_t, self.p['num_time_bins'])
				# Inverse warp.
				target_ts = self._get_unwarped_times(t_vals, file_index)
				target_ts *= self.template_dur
				# Then make a spectrogram.
				spec, flag = self.p['get_spec'](0.0, self.template_dur, \
					self.audio[file_index], self.p, fs=self.fs, \
					max_dur=None, target_times=target_ts)
				assert flag
				if self.transform:
					spec = self.transform(spec)
				result.append(spec)
				break
		np.random.seed(None)
		if single_index:
			return result[0]
		return result


	def get_specific_item(self, query_filename, quantile):
		"""
		Return a specific window of birdsong as a numpy array.

		Parameters
		----------
		query_filename : str
			Audio filename.
		quantile : float
			0 <= ``quantile`` <= 1

		Returns
		-------
		spec : numpy.ndarray
			Spectrogram.
		"""
		file_index = self.audio_filenames.index(query_filename)
		start_t = self.start_q + quantile * \
				(self.stop_q - self.start_q - self.window_frac)
		stop_t = start_t + self.window_frac
		t_vals = np.linspace(start_t, stop_t, self.p['num_time_bins'])
		# Inverse warp.
		target_ts = self._get_unwarped_times(t_vals, file_index)
		target_ts *= self.template_dur
		# Then make a spectrogram.
		spec, flag = self.p['get_spec'](0.0, self.template_dur, \
			self.audio[file_index], self.p, fs=self.fs, \
			max_dur=None, target_times=target_ts)
		assert flag
		return spec


	def get_whole_warped_spectrogram(self, query_filename, time_bins=128):
		"""
		Get an entire warped song motif.

		Parameters
		----------
		query_filename : str
			Which audio file to use.
		time_bins : int, optional
			Number of time bins.

		Returns
		-------
		spec : numpy.ndarray
			Spectrogram.
		"""
		file_index = self.audio_filenames.index(query_filename)
		t_vals = np.linspace(self.start_q, self.stop_q, time_bins)
		# Inverse warp.
		target_ts = self._get_unwarped_times(t_vals, file_index)
		target_ts *= self.template_dur
		# Then make a spectrogram.
		spec, flag = self.p['get_spec'](0.0, self.template_dur, \
			self.audio[file_index], self.p, fs=self.fs, \
			max_dur=None, target_times=target_ts)
		assert flag
		return spec



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
	``ava.data.data_container`` relies on this.
	"""
	return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if \
		_is_hdf5_file(f)]


def _get_wavs_from_dir(dir):
	"""Return a sorted list of wave files from a directory."""
	return [os.path.join(dir, f) for f in sorted(os.listdir(dir)) if \
		_is_wav_file(f)]


def _is_hdf5_file(filename):
	"""Is the given filename an hdf5 file?"""
	return len(filename) > 5 and filename[-5:] == '.hdf5'


def _is_wav_file(filename):
	"""Is the given filename a wave file?"""
	return len(filename) > 4 and filename[-4:] == '.wav'



if __name__ == '__main__':
	pass


###
