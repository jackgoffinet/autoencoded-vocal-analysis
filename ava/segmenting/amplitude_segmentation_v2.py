"""
Amplitude-based syllable segmentation.

"""
__author__ ="Jack Goffinet"
__date__ = "December 2018 - August 2019"


import numpy as np
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.signal import convolve2d


EPSILON = 1e-12

# Segmenting parameters. # KEEP THIS?
default_seg_params = {
	'th_1':0.1,
	'th_2':0.2,
	'th_3':0.3,
	'min_dur':0.1,
	'max_dur':2.0,
	'freq_smoothing': 3.0,
	'smoothing_timescale': 0.02,
	'num_time_bins': 128,
	'num_freq_bins': 128,
	'softmax': False,
	'temperature':0.5,
}


def get_onsets_offsets(audio, seg_params, return_traces=False):
	"""
	Segment the spectrogram using thresholds on its ampltiude.

	Parameters
	----------
	spec :

	dt :

	seg_params :

	return_traces : bool, optional

	Returns
	-------
	onsets : numpy array
		Onset times, in seconds

	offsets : numpy array
		Offset times, in seconds

	traces : list of a single numpy array
		The amplitude trace used in segmenting decisions. Returned if
		return_traces is True.
	"""
	p = {**default_seg_params, **seg_params}
	spec, dt = get_spec(audio, p)
	min_syll_len = int(np.floor(p['min_dur'] / dt))
	max_syll_len = min(p['num_time_bins'], int(np.ceil(p['max_dur'] / dt)))
	th_1, th_2, th_3 = p['th_1'], p['th_2'], p['th_3'] # treshholds
	smoothing_time = p['smoothing_timescale'] / dt
	onsets, offsets = [], []
	too_short, too_long = 0, 0

	# # Automated thresholding.
	# median = np.median(spec)
	# mad = np.median(np.abs(spec - median)) # Median Absolute Deviation
	# spec -= median
	# spec[spec<0.0] = 0.0
	# spec /= mad

	# smoothing_time = p['smoothing_timescale'] / dt
	# smooth = gaussian_filter(spec, [p['freq_smoothing'], smoothing_time])

	if p['softmax']:
		amps = softmax(spec, t=p['temperature'])
	else:
		amps = np.sum(spec, axis=0)
	# Smooth.
	# if p['smoothing'] ??
	amps = gaussian_filter(amps, smoothing_time)
	# Scale by MAD.
	median = np.median(amps)
	mad = np.median(np.abs(amps - median)) # Median Absolute Deviation
	amps -= median
	amps /= mad

	# Find local maxima greater than th_3.
	local_maxima = []
	for i in range(1,len(amps)-1,1):
		if amps[i] > th_3 and amps[i] == np.max(amps[i-1:i+2]):
			local_maxima.append(i)

	# Then search to the left and right for onsets and offsets.
	for local_max in local_maxima:
		if len(offsets) > 1 and local_max < offsets[-1]:
			continue
		i = local_max - 1
		while i > 0:
			if amps[i] < th_1:
				onsets.append(i)
				break
			elif amps[i] < th_2 and amps[i] == np.min(amps[i-1:i+2]):
				onsets.append(i)
				break
			i -= 1
		if len(onsets) != len(offsets) + 1:
			onsets = onsets[:len(offsets)]
			continue
		i = local_max + 1
		while i < len(amps):
			if amps[i] < th_1:
				offsets.append(i)
				break
			elif amps[i] < th_2 and amps[i] == np.min(amps[i-1:i+2]):
				offsets.append(i)
				break
			i += 1
		if len(onsets) != len(offsets):
			onsets = onsets[:len(offsets)]
			continue

	# Throw away syllables that are too long or too short.
	new_onsets = []
	new_offsets = []
	for i in range(len(offsets)):
		t1, t2 = onsets[i], offsets[i]
		if t2 - t1 + 1 <= max_syll_len and t2 - t1 + 1 >= min_syll_len:
			new_onsets.append(t1)
			new_offsets.append(t2)
		elif t2 - t1 + 1 > max_syll_len:
			too_long += 1
		else:
			too_short += 1

	# Return decisions.
	if return_traces:
		return new_onsets, new_offsets, [amps]
	return new_onsets, new_offsets


def get_spec(audio, p):
	"""
	Get a spectrogram.

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
	fs, audio = wavfile.read(audio)
	assert fs == p['fs']
	f, t, spec = stft(audio, fs=fs, nperseg=p['nperseg'], \
		noverlap=p['noverlap'])
	i1 = np.searchsorted(f, p['min_freq'])
	i2 = np.searchsorted(f, p['max_freq'])
	f, spec = f[i1:i2], spec[i1:i2]
	spec = np.log(np.abs(spec) + EPSILON)
	spec -= p['spec_min_val']
	spec /= p['spec_max_val'] - p['spec_min_val']
	spec = np.clip(spec, 0.0, 1.0)
	return spec, t[1]-t[0], f


def softmax(arr, t=0.5):
	temp = np.exp(arr/t)
	temp /= np.sum(temp, axis=0)
	return np.sum(np.multiply(arr, temp), axis=0)



if __name__ == '__main__':
	pass


###
