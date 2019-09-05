"""
Useful functions for segmenting.

"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"


import numpy as np
import os
from scipy.signal import stft


EPSILON = 1e-12



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


def copy_segments_to_standard_format(orig_seg_dirs, new_seg_dirs, seg_ext, \
	delimiter, usecols, skiprows):
	"""
	Copy onsets/offsets from SAP, MUPET, or Deepsqueak into their files.

	Note
	----
	- TO DO: rename

	Parameters
	----------
	orig_seg_dirs : list of str
		Directories containing original segments.
	new_seg_dirs : list of str
		Directories for new segments.
	p : dict
		...

	"""
	assert len(seg_ext) == 4
	for orig_seg_dir, new_seg_dir in zip(orig_seg_dirs, new_seg_dirs):
		if not os.path.exists(new_seg_dir):
			os.makedirs(new_seg_dir)
		seg_fns = [os.path.join(orig_seg_dir,i) for i in \
				os.listdir(orig_seg_dir) if len(i) > 4 and i[-4:] == seg_ext]
		for seg_fn in seg_fns:
			segs = np.loadtxt(seg_fn, delimiter=delimiter, skiprows=skiprows, \
					usecols=usecols).reshape(-1,2)
			new_seg_fn = os.path.join(new_seg_dir, os.path.split(seg_fn)[-1])
			new_seg_fn = new_seg_fn[:-4] + '.txt'
			header = "Onsets/offsets copied from "+seg_fn
			np.savetxt(new_seg_fn, segs, fmt='%.5f', header=header)


def get_audio_seg_filenames(audio_dirs, seg_dirs, p):
	"""Return lists of sorted filenames."""
	audio_fns, seg_fns = [], []
	for audio_dir, seg_dir in zip(audio_dirs, seg_dirs):
		temp_fns = [i for i in sorted(os.listdir(audio_dir)) if \
				_is_audio_file(i)]
		audio_fns += [os.path.join(audio_dir, i) for i in temp_fns]
		temp_fns = [i[:-4] + '.txt' for i in temp_fns]
		seg_fns += [os.path.join(seg_dir, i) for i in temp_fns]
	return audio_fns, seg_fns


def get_onsets_offsets_from_file(filename, p):
	"""
	A wrapper around numpy.loadtxt for reading onsets & offsets.

	TO DO: finish this!
	"""
	return np.loadtxt(filename, unpack=True)


def _is_audio_file(filename):
	return len(filename) > 4 and filename[-4:] == '.wav'


if __name__ == '__main__':
	pass


###
