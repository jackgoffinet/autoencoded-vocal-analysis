"""
Useful functions related to the `ava.models` subpackage.

"""
__date__ = "July 2020"


from affinewarp.crossval import paramsearch
import h5py
import numpy as np
import os
from scipy.io import wavfile
from scipy.signal import stft
import torch


DEFAULT_SEARCH_PARAMS = {
	'samples_per_knot': 10,
	'n_valid_samples': 5,
	'n_train_folds': 3,
	'n_valid_folds': 1,
	'n_test_folds': 1,
	'knot_range': (-1, 2),
	'smoothness_range': (1e-2, 1e2),
	'warpreg_range': (1e-2, 1e2),
	'iter_range': (50, 51),
	'warp_iter_range': (50, 101),
	'outfile': None,
}
"""Default parameters sent to `affinewarp.crossval.paramsearch`"""
EPSILON = 1e-9


def cross_validation_warp_parameter_search(audio_dirs, spec_params, \
	search_params={}, warp_type='spectrogram', verbose=True):
	"""
	Perform a parameter search over timewarping parameters.

	This is a wrapper around `affinewarp.crossval.paramsearch`.

	Note
	----
	* All `.wav` files should be the same duration!

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	spec_params : dict
		Preprocessing parameters. Must contain keys: ``'window_length'``,
		``'nperseg'``, ``'noverlap'``, ``'min_freq'``, ``'max_freq'``,
		``'spec_min_val'``, and ``'spec_max_val'``.
	search_params : dict, optional
		Parameters sent to `affinewarp.crossval.paramsearch`. Defaults to
		`DEFAULT_SEARCH_PARAMS`.
	warp_type : {``'amplitude'``, ``'spectrogram'``}, optional
		Whether to time-warp using ampltidue traces or full spectrograms.
		Defaults to ``'spectrogram'``.
	verbose : bool, optional
		Defaults to `True`.

	Returns
	-------
	res : dict
		Complete `affinewarp.crossval.paramsearch` result. See
		github.com/ahwillia/affinewarp/blob/master/affinewarp/crossval.py
	"""
	assert type(spec_params) == type({})
	assert warp_type in ['amplitude', 'spectrogram']
	search_params = {**DEFAULT_SEARCH_PARAMS, **search_params}
	# Collect audio filenames.
	if verbose:
		print("Collecting spectrograms...")
	audio_fns = []
	for audio_dir in audio_dirs:
		audio_fns += _get_wavs_from_dir(audio_dir)
	# Make spectrograms.
	all_audio = [wavfile.read(audio_fn)[1] for audio_fn in audio_fns]
	fs = wavfile.read(audio_fns[0])[0]
	specs, amps, _  = _get_specs_and_amplitude_traces(all_audio, fs, \
			spec_params)
	if verbose:
		print("\tDone.")
		print("Running parameter search...")
	# Run the parameter search and return.
	if warp_type == 'amplitude':
		to_warp = amps
	else:
		to_warp = specs
	res = paramsearch(to_warp, **search_params)
	if verbose:
		print("\tDone.")
	return res


def anchor_point_warp_parameter_search(audio_dirs, spec_params, search_params):
	"""
	Find the time-warping parameters that best align hand-labeled anchor points.

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	spec_params : dict
		Preprocessing parameters. Must contain keys: ``'window_length'``,
		``'nperseg'``, ``'noverlap'``, ``'min_freq'``, ``'max_freq'``,
		``'spec_min_val'``, and ``'spec_max_val'``.
	search_params : dict, optional
		Parameters sent to `affinewarp.crossval.paramsearch`. Defaults to
		`DEFAULT_SEARCH_PARAMS`.
	warp_type : {``'amplitude'``, ``'spectrogram'``}, optional
		Whether to time-warp using ampltidue traces or full spectrograms.
		Defaults to ``'spectrogram'``.
	verbose : bool, optional
		Defaults to `True`.

	Returns
	-------

	"""
	raise NotImplementedError
	assert type(spec_params) == type({})
	assert warp_type in ['amplitude', 'spectrogram']
	search_params = {**DEFAULT_SEARCH_PARAMS, **search_params}
	# Collect audio filenames.
	if verbose:
		print("Collecting spectrograms...")
	audio_fns = []
	for audio_dir in audio_dirs:
		audio_fns += _get_wavs_from_dir(audio_dir)
	# Make spectrograms.
	all_audio = [wavfile.read(audio_fn)[1] for audio_fn in audio_fns]
	fs = wavfile.read(audio_fns[0])[0]
	specs, amps, _  = _get_specs_and_amplitude_traces(all_audio, fs, \
			spec_params)
	if verbose:
		print("\tDone.")
		print("Running parameter search...")


def _get_sylls_per_file(partition):
	"""
	Open an hdf5 file and see how many syllables it has.

	Assumes all hdf5 file referenced by `partition` have the same number of
	syllables.

	Parameters
	----------
	partition : dict
		Contains two keys, ``'test'`` and ``'train'``, that map to lists of hdf5
		files. Defines the random test/train split.

	Returns
	-------
	sylls_per_file : int
		How many syllables are in each file.
	"""
	key = 'train' if len(partition['train']) > 0 else 'test'
	assert len(partition[key]) > 0
	filename = partition[key][0] # Just grab the first file.
	with h5py.File(filename, 'r') as f:
		sylls_per_file = len(f['specs'])
	return sylls_per_file


def _get_spec(audio, fs, p):
	"""
	Make a basic spectrogram.

	Parameters
	----------
	audio : numpy.ndarray
		Audio
	fs : int
		Samplerate
	p : dict
		Contains keys `'nperseg'`, `'noverlap'`, `'min_freq'`, `'max_freq'`,
		`'spec_min_val'`, and `'spec_max_val'`.

	Returns
	-------
	spec : numpy.ndarray
		Spectrogram, freq_bins x time_bins
	dt : float
		Spectrogram time step
	"""
	f, t, spec = stft(audio, fs=fs, nperseg=p['nperseg'], \
			noverlap=p['noverlap'])
	i1 = np.searchsorted(f, p['min_freq'])
	i2 = np.searchsorted(f, p['max_freq'])
	spec = spec[i1:i2]
	f = f[i1:i2]
	spec = np.log(np.abs(spec) + EPSILON)
	spec -= p['spec_min_val']
	spec /= p['spec_max_val'] - p['spec_min_val'] + EPSILON
	spec = np.clip(spec, 0.0, 1.0)
	return spec, t[1]-t[0]


def _get_specs_and_amplitude_traces(all_audio, fs, spec_params):
	"""
	Return spectrograms and amplitude traces given a list of audio.

	Parameters
	----------
	all_audio : list of numpy.ndarray
		List of audio.
	fs : int
		Audio samplerate
	spec_params : dict
		Contains keys `'nperseg'`, `'noverlap'`, `'min_freq'`, `'max_freq'`,
		`'spec_min_val'`, and `'spec_max_val'`.

	Returns
	-------
	specs : numpy.ndarray
		Spectrograms
	amps : numpy.ndarray
		Amplitude traces
	template_dur : float
		Template duration
	"""
	# Make spectrograms.
	specs = []
	for i in range(len(all_audio)):
		spec, dt = _get_spec(all_audio[i], fs, spec_params)
		specs.append(spec.T)
	# Check to make sure everything's the same shape.
	assert len(specs) > 0
	min_time_bins = min(spec.shape[0] for spec in specs)
	specs = [spec[:min_time_bins] for spec in specs]
	min_freq_bins = min(spec.shape[1] for spec in specs)
	specs = [spec[:,:min_freq_bins] for spec in specs]
	num_time_bins = specs[0].shape[0]
	assert num_time_bins == min_time_bins
	template_dur = num_time_bins * dt
	# Compute amplitude traces.
	amps = []
	for i in range(len(all_audio)):
		amp_trace = np.sum(specs[i], axis=-1, keepdims=True)
		amp_trace -= np.min(amp_trace)
		amp_trace /= np.max(amp_trace) + EPSILON
		amps.append(amp_trace)
	# Stack and return.
	amps = np.stack(amps)
	specs = np.stack(specs)
	return specs, amps, template_dur


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


def numpy_to_tensor(x):
	"""Transform a numpy array into a torch.FloatTensor."""
	return torch.from_numpy(x).type(torch.FloatTensor)


def _is_hdf5_file(filename):
	"""Is the given filename an hdf5 file?"""
	return len(filename) > 5 and filename[-5:] == '.hdf5'


def _is_wav_file(filename):
	"""Is the given filename a wave file?"""
	return len(filename) > 4 and filename[-4:] == '.wav'



if __name__ == '__main__':
	pass


###
