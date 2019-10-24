"""
Useful functions for segmenting.

"""
__date__ = "August 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from scipy.signal import stft
from scipy.io import wavfile


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


def clean_segments_by_hand(audio_dirs, orig_seg_dirs, new_seg_dirs, p, \
	shoulder=0.1):
	"""
	Plot spectrograms and ask for accept/reject input.

	Parameters
	----------
	audio_dirs : ...
		...
	orig_seg_dirs : ...
		...
	new_seg_dirs : ...
		...
	p : ...
		...
	shoulder : ...
		...
	"""
	audio_fns, orig_seg_fns = get_audio_seg_filenames(audio_dirs, orig_seg_dirs)
	temp_dict = dict(zip(orig_seg_dirs, new_seg_dirs))
	new_seg_fns = []
	for orig_seg_fn in orig_seg_fns:
		a,b = os.path.split(orig_seg_fn)
		new_seg_fns.append(os.path.join(temp_dict[a], b))
	gen = zip(audio_fns, orig_seg_fns, new_seg_fns)
	for audio_fn, orig_seg_fn, new_seg_fn in gen:
		print("orig_seg_fn", orig_seg_fn)
		header = "Onsets/offsets cleaned by hand from "+orig_seg_fn
		# Get onsets and offsets.
		onsets, offsets = _read_onsets_offsets(orig_seg_fn)
		if len(onsets) == 0:
			np.savetxt(new_seg_fn, np.array([]), header=header)
			continue
		# Get spectrogram.
		fs, audio = wavfile.read(audio_fn)
		assert fs == p['fs'], "Found fs="+str(fs)+", expected fs="+str(p['fs'])
		spec, dt, f = get_spec(audio, p)
		# Collect user input.
		good_indices = []
		for i, (onset, offset) in enumerate(zip(onsets, offsets)):
			i1 = max(0, int((onset - shoulder) / dt))
			i2 = min(spec.shape[1], int((offset + shoulder) / dt))
			t1 = max(0, onset-shoulder)
			t2 = min(len(audio)/fs, offset+shoulder)
			print("t1,t2", t1,t2)
			plt.imshow(spec[:,i1:i2], origin='lower', aspect='auto', \
					extent=[t1, t2, f[0]/1e3, f[-1]/1e3])
			plt.ylabel('Frequency (kHz)')
			plt.xlabel('Time (s)')
			plt.axvline(x=onset, c='r')
			plt.axvline(x=offset, c='r')
			plt.savefig('temp.pdf')
			plt.close('all')

			response = input("[Good]? or 'x': ")
			if response != 'x':
				good_indices.append(i)
		good_indices = np.array(good_indices, dtype='int')
		onsets, offsets = onsets[good_indices], offsets[good_indices]
		combined = np.stack([onsets, offsets]).T
		np.savetxt(new_seg_fn, combined, fmt='%.5f', header=header)



def copy_segments_to_standard_format(orig_seg_dirs, new_seg_dirs, seg_ext, \
	delimiter, usecols, skiprows, max_duration=None):
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
	seg_ext : ..
		....
	delimiter : ...
		...
	usecols : ...
		...
	skiprows : ...
		...
	max_duration : {None, float}, optional
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

			if max_duration is not None:
				new_segs = []
				for seg in segs:
					if seg[1]-seg[0] < max_duration:
						new_segs.append(seg)
				if len(new_segs) > 0:
					segs = np.stack(new_segs)
				else:
					segs = np.array([])
			new_seg_fn = os.path.join(new_seg_dir, os.path.split(seg_fn)[-1])
			new_seg_fn = new_seg_fn[:-4] + '.txt'
			header = "Onsets/offsets copied from "+seg_fn
			np.savetxt(new_seg_fn, segs, fmt='%.5f', header=header)


def get_audio_seg_filenames(audio_dirs, seg_dirs):
	"""Return lists of sorted filenames."""
	audio_fns, seg_fns = [], []
	for audio_dir, seg_dir in zip(audio_dirs, seg_dirs):
		temp_fns = [i for i in sorted(os.listdir(audio_dir)) if \
				_is_audio_file(i)]
		audio_fns += [os.path.join(audio_dir, i) for i in temp_fns]
		temp_fns = [i[:-4] + '.txt' for i in temp_fns]
		seg_fns += [os.path.join(seg_dir, i) for i in temp_fns]
	return audio_fns, seg_fns


def _read_onsets_offsets(filename):
	"""
	A wrapper around numpy.loadtxt for reading onsets & offsets.

	Parameters
	----------
	filename : str
		Filename of a text file containing one header line and two columns.
	"""
	arr = np.loadtxt(filename, skiprows=1)
	if len(arr) == 0:
		return [], []
	assert arr.shape[1] == 2, "Found invalid shape: "+str(arr.shape)
	return arr[:,0], arr[:,1]


def _is_audio_file(filename):
	return len(filename) > 4 and filename[-4:] == '.wav'


if __name__ == '__main__':
	pass


###
