"""
Useful functions for segmenting.

"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"


import numpy as np
import os
from scipy.signal import stft



def get_spec(audio, p):
	"""
	Get a spectrogram.

	Much simpler than ava.preprocessing.utils.get_spec

	Parameters
	----------
	audio : numpy array of floats
		Audio

	p : dict
		Spectrogram parameters.

	Returns
	-------
	spec : numpy array of floats
		Spectrogram of shape [freq_bins x time_bins]

	dt : float
		Time step associated with time bins.

	f : Array of frequencies.
	"""
	f, t, spec = stft(audio, fs=p['fs'], nperseg=p['nperseg'], \
		noverlap=p['noverlap'])
	i1 = np.searchsorted(f, p['min_freq'])
	i2 = np.searchsorted(f, p['max_freq'])
	f, spec = f[i1:i2], spec[i1:i2]
	spec = np.log(np.abs(spec) + EPSILON)
	spec -= p['spec_min_val']
	spec /= p['spec_max_val'] - p['spec_min_val']
	spec = np.clip(spec, 0.0, 1.0)
	return spec, t[1]-t[0], f



def copy_segments_to_standard_format(orig_seg_dirs, new_seg_dirs):
	"""
	Copy onsets/offsets from SAP, MUPET, or Deepsqueak into their files.

	Parameters
	----------

	"""
	pass


def get_audio_seg_filenames(audio_dirs, seg_dirs, p):
	"""Return lists of sorted filenames."""
	audio_fns, seg_fns = [], []
	for audio_dir, seg_dir in zip(audio_dirs, seg_dirs):
		temp_fns = [i for i in sorted(os.listdir(audio_dir)) if \
				is_audio_file(i)]
		audio_fns += [os.path.join(audio_dir, i) for i in temp_fns]
		temp_fns = [i[:-4] + p['seg_extension'] for i in temp_fns]
		seg_fns += [os.path.join(segment_dir, i) for i in temp_fns]
	return audio_fns, seg_fns


def get_onsets_offsets_from_file(filename, p):
	"""
	A wrapper around numpy.loadtxt for reading onsets & offsets.

	TO DO: finish this!
	"""
	return np.loadtxt(filename, unpack=True)



if __name__ == '__main__':
	pass


###
