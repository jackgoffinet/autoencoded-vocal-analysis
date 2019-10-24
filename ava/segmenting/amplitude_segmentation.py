"""
Amplitude-based syllable segmentation.

"""
__date__ = "December 2018 - October 2019"


import numpy as np
from scipy.io import wavfile
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d

from ava.segmenting.utils import get_spec


EPSILON = 1e-9



def get_onsets_offsets(audio, p, return_traces=False):
	"""
	Segment the spectrogram using thresholds on its ampltiude.

	Parameters
	----------
	audio : numpy.ndarray
		Raw audio samples.
	p : dict
		Parameters.
	return_traces : bool, optional
		Whether to return traces. Defaults to `False`.
	Returns
	-------
	onsets : numpy array
		Onset times, in seconds
	offsets : numpy array
		Offset times, in seconds
	traces : list of a single numpy array
		The amplitude trace used in segmenting decisions. Returned if
		return_traces is `True`.
	"""
	spec, dt, _ = get_spec(audio, p)
	min_syll_len = int(np.floor(p['min_dur'] / dt))
	max_syll_len = int(np.ceil(p['max_dur'] / dt))
	th_1, th_2, th_3 = p['th_1'], p['th_2'], p['th_3'] # treshholds
	smoothing_time = p['smoothing_timescale'] / dt
	onsets, offsets = [], []
	too_short, too_long = 0, 0

	if p['softmax']:
		amps = _softmax(spec, t=p['temperature'])
	else:
		amps = np.sum(spec, axis=0)
	# Smooth.
	amps = gaussian_filter(amps, smoothing_time)

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
			new_onsets.append(t1 * dt)
			new_offsets.append(t2 * dt)
		elif t2 - t1 + 1 > max_syll_len:
			too_long += 1
		else:
			too_short += 1

	# Return decisions.
	if return_traces:
		return new_onsets, new_offsets, [amps]
	return new_onsets, new_offsets


def _softmax(arr, t=0.5):
	temp = np.exp(arr/t)
	temp /= np.sum(temp, axis=0) + EPSILON
	return np.sum(np.multiply(arr, temp), axis=0)



if __name__ == '__main__':
	pass


###
