"""
Amplitude-based syllable segmentation.

"""
__author__ ="Jack Goffinet"
__date__ = "December 2018 - July 2019"


import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d


# Segmenting parameters.
default_seg_params = {
	'th_1':0.1,
	'th_2':0.2,
	'th_3':0.3,
	'min_var':0.2,
	'min_dur':0.1,
	'max_dur':2.0,
	'freq_smoothing': 3.0,
	'smoothing_timescale': 0.02,
	'num_time_bins': 128,
	'num_freq_bins': 128,
	'softmax': False,
	'temperature':0.5,
}


def get_onsets_offsets(spec, dt, seg_params, return_traces=False):
	"""
	Segment the spectrogram using thresholds on its ampltiude.

	"""
	p = {**default_seg_params, **seg_params}
	min_syll_len = int(np.floor(p['min_dur'] / dt))
	max_syll_len = min(p['num_time_bins'], int(np.ceil(p['max_dur'] / dt)))
	onsets, offsets = [], []
	too_short = 0
	too_long = 0
	# Automated thresholding.
	med = np.median(spec)
	mad_sigma = 1.4826 * np.median(np.abs(spec-med)) # Assumes normal distribution
	spec -= med
	spec[spec<0.0] = 0.0
	spec /= mad_sigma
	# Get amplitude data and its derivative.
	smoothing_time = p['smoothing_timescale'] / dt
	smooth = gaussian_filter(spec, [p['freq_smoothing'], smoothing_time])
	if p['softmax']:
		amps = softmax(smooth, t=p['temperature'])
	else:
		amps = np.sum(smooth, axis=0)

	th_1, th_2, th_3 = p['th_1'], p['th_2'], p['th_3']
	local_maxima = []
	# Find local maxima greater than th_1
	for i in range(1,len(amps)-1,1):
		if amps[i] > th_3 and amps[i] == np.max(amps[i-1:i+2]):
			local_maxima.append(i)
	# Then search left and right for onsets and offsets.
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

	# Return everything else.
	if return_traces:
		return new_onsets, new_offsets, [amps]
	return new_onsets, new_offsets


def softmax(arr, t=0.5):
	temp = np.exp(arr/t)
	temp /= np.sum(temp, axis=0)
	return np.sum(np.multiply(arr, temp), axis=0)



if __name__ == '__main__':
	pass


###
