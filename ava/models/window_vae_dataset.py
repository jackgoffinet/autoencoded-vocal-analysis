"""
Data stuff for animal vocalization syllables.

Contains
--------
- function get_window_partition
- function get_window_data_loaders
- function get_warped_window_data_loaders
- class FixedWindowDataset
- class WarpedWindowDataset

"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"


from affinewarp import PiecewiseWarping
from affinewarp.piecewisewarp import densewarp
import h5py
import joblib
import numpy as np
import os
from scipy.interpolate import interp1d, interp2d
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter
from scipy.signal import stft
import torch
from torch.utils.data import Dataset, DataLoader

EPSILON = 1e-12



def get_window_partition(audio_dirs, roi_dirs, split, roi_extension='.txt', \
	shuffle=True):
	"""
	Get a train/test split.

	Parameters
	----------
	audio_dirs : ...
		...

	Returns
	-------
	partition : dict
		...

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


def get_fixed_window_data_loaders(partition, p, batch_size=64, \
	shuffle=(True, False), num_workers=4):
	"""

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


def get_warped_window_data_loaders(audio_dirs, template_dir, p, batch_size=64, \
	num_workers=3, load_warp=False):
	"""
	Get DataLoaders for training and testing.

	Right now, these are the same data loaders...

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	template_dir : str
		Directory where templates are saved.
	p : dict
		Parameters. ADD REFERENCE!
	batch_size : int, optional
		DataLoader batch size. Defaults to ``64``.
	num_workers : int, optional
		Number of CPU workers to retrieve data for the model. Defaults to ``3``.
	load_warp : bool, optional
		Whether to load a previously saved time warping result. Defaults to
		``False``.

	Returns
	-------
	loaders : dict
		A dictionary ...

	Warning
	-------
	- Audio files must all be the same duration!

	Note
	----
	- TO DO: add train/test split.

	"""
	audio_fns = []
	for audio_dir in audio_dirs:
		audio_fns += [os.path.join(audio_dir, i) for i in \
			sorted(os.listdir(audio_dir)) if i[-4:] == '.wav']
	dataset = WarpedWindowDataset(audio_fns, template_dir, p, \
		transform=numpy_to_tensor, load_warp=load_warp)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, \
		num_workers=num_workers)
	return {'train': dataloader, 'test': dataloader}



class FixedWindowDataset(Dataset):
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

	def __init__(self, audio_filenames, template_dir, p, transform=None, \
		dataset_length=2000, load_warp=False, start_q=-0.1, stop_q=1.1):
		"""
		Create a torch.utils.data.Dataset for chunks of animal vocalization.

		Parameters
		----------
		audio_filenames : list of strings
			List of wav files.
		template_dir : str
			...
		transform : {None, function}, optional
			Transformation to apply to each item. Defaults to ``None`` (no
			transformation)
		"""
		self.audio_filenames = audio_filenames
		self.audio = [wavfile.read(fn)[1] for fn in audio_filenames]
		self.fs = wavfile.read(audio_filenames[0])[0]
		self.dataset_length = dataset_length
		self.p = p
		self.transform = transform
		self.start_q = start_q
		self.stop_q = stop_q
		self.compute_warp(template_dir, load_warp=load_warp)
		self.window_frac = self.p['window_length'] / self.template_dur


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
		try:
			f, t, spec = stft(audio, fs=self.fs, nperseg=self.p['nperseg'], \
				noverlap=self.p['noverlap'])
		except:
			print("caught in get spec")
			print(type(audio))
			print(self.fs)
			print(type(self.p))
			quit()
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


	def get_unwarped_times(self, y_vals, k):
		"""
		Convert quantile times in [0,1] to real times in [0,1].

		Assumes y_vals is sorted.
		"""
		x_knots, y_knots = self.x_knots[k], self.y_knots[k]
		interp = interp1d(y_knots, x_knots, bounds_error=False, \
				fill_value='extrapolate', assume_sorted=True)
		x_vals = interp(y_vals)
		return x_vals


	def compute_warp(self, template_dir, load_warp=False):
		"""
		Warp each song rendition to the template.
		"""
		template = self.get_template(template_dir)
		if load_warp:
			try:
				self.x_knots = np.load('temp_data/x_knots.npy')
				self.y_knots = np.load('temp_data/y_knots.npy')
				return
			except IOError:
				pass
		amp_traces = []
		specs = []
		for i in range(len(self.audio)):
			specs.append(self.get_spec(self.audio[i])[0].T)
			amp_trace = np.sum(specs[-1], axis=1)
			amp_trace -= np.min(amp_trace)
			amp_trace /= np.max(amp_trace)
			amp_traces.append(amp_trace.reshape(-1,1))
		specs = np.stack(specs)
		amp_traces = np.stack(amp_traces)
		# print("specs", specs.shape)
		self.num_time_bins = specs.shape[1]
		# print("amp_traces", amp_traces.shape) # 2413, 84, 1
		model = PiecewiseWarping(n_knots=self.p['n_knots'], \
			warp_reg_scale=1e-6, smoothness_reg_scale=20.0)
		model.fit(specs, iterations=50, warp_iterations=200)
		np.save('temp_data/x_knots.npy', model.x_knots)
		np.save('temp_data/y_knots.npy', model.y_knots)
		self.x_knots = model.x_knots
		self.y_knots = model.y_knots
		# x_knots = np.load('temp_data/x_knots.npy')
		# y_knots = np.load('temp_data/y_knots.npy')
		# self.x_knots = x_knots
		# self.y_knots = y_knots

		# import matplotlib.pyplot as plt
		# plt.switch_backend('agg')
		# for i, amp_trace in enumerate(amp_traces):
		# 	plt.plot(amp_trace.flatten(), lw=0.5, c='b', alpha=0.1)
		# plt.savefig('temp1.pdf')
		# plt.close('all')
		# K = x_knots.shape[0]
		# T, N = 84, 1
		# unwarped = densewarp(y_knots, x_knots, amp_traces, np.empty((K, T, N)))
		# for amp_trace in unwarped:
		# 	plt.plot(amp_trace.flatten(), lw=0.5, c='b', alpha=0.1)
		# plt.savefig('temp2.pdf')
		# plt.close('all')
		#
		# q1, q2 = -0.1, 1.1
		# ts =  np.linspace(q1, q2, 200)
		# for k in range(specs.shape[0]):
		# 	unwarped_ts = self.get_unwarped_times(ts, k)
		# 	unwarped_ts = unwarped_ts * self.template_dur
		# 	spec = self.get_spec(self.audio[k], target_ts=unwarped_ts)[0].T
		# 	amp_trace = np.sum(spec, axis=1)
		# 	plt.plot(amp_trace.flatten(), lw=0.5, c='b', alpha=0.1)
		# plt.savefig('temp3.pdf')



	def __getitem__(self, index, seed=None):
		"""
		Return a random window of birdsong.
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
				target_ts = self.get_unwarped_times(t_vals, file_index)
				target_ts *= self.template_dur
				# Then make a spectrogram.
				spec, flag = self.p['get_spec'](0.0, self.template_dur, \
					self.audio[file_index], self.p, fs=self.fs, \
					max_dur=None, target_times=target_ts)
				if not flag: # NOTE: HERE
					print("flag")
					print(start_t, stop_t)
					print(start_wt, stop_wt)
					print(len(self.audio[file_index])/self.fs)
					t1, t2 = start_wt, stop_wt
					fs = self.fs
					s1, s2 = int(round(t1*fs)), int(round(t2*fs))
					print("s: ", s1, s2, len(self.audio[file_index]))
					print(t2 - t1)
					print(self.p['max_dur'] + 1e-4)
					import matplotlib.pyplot as plt
					plt.switch_backend('agg')
					plt.imshow(spec)
					plt.savefig('temp.pdf')
					print(start_wt, stop_wt)
					print(start_t, stop_t)
					print(y_knot)
					print(flag)
					quit()
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
		"""
		file_index = self.audio_filenames.index(query_filename)
		start_t = self.start_q + quantile * \
				(self.stop_q - self.start_q - self.window_frac)
		stop_t = start_t + self.window_frac
		t_vals = np.linspace(start_t, stop_t, self.p['num_time_bins'])
		# Inverse warp.
		target_ts = self.get_unwarped_times(t_vals, file_index)
		target_ts *= self.template_dur
		# Then make a spectrogram.
		spec, flag = self.p['get_spec'](0.0, self.template_dur, \
			self.audio[file_index], self.p, fs=self.fs, \
			max_dur=None, target_times=target_ts)

		if not flag: # NOTE: HERE
			print("flag")
			print(quantile)
			print(start_wt, stop_wt)
			print(len(self.audio[file_index])/self.fs)
			t1, t2 = start_wt, stop_wt
			fs = self.fs
			s1, s2 = int(round(t1*fs)), int(round(t2*fs))
			print("s: ", s1, s2, len(self.audio[file_index]))
			print(t2 - t1)
			print(self.p['max_dur'] + 1e-4)
			import matplotlib.pyplot as plt
			plt.switch_backend('agg')
			plt.imshow(spec)
			plt.savefig('temp.pdf')
			print(start_wt, stop_wt)
			print(start_t, stop_t)
			print(y_knot)
			print(flag)
			quit()
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
