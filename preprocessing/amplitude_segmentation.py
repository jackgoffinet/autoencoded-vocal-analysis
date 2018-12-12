from __future__ import print_function, division
"""
Amplitude based syllable segmentation.

TO DO: Add lambda filtering
"""
__author__ ="Jack Goffinet"
__date__ = "December 2018"


import numpy as np
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.signal import convolve2d



# Segmenting parameters.
default_seg_params = {
	'a_onset':0.1,
	'a_offset':0.05,
	'a_dot_onset':0.0,
	'a_dot_offset':0.0,
	'min_var':0.2,
	'min_dur':0.1,
	'max_dur':2.0,
	'freq_smoothing': 3.0,
	'smoothing_timescale': 0.02,
	'num_time_bins': 128,
	'num_freq_bins': 128,
	'freq_response': np.ones(128),
}


def get_onsets_offsets(spec, dt, seg_params, return_traces=False):
	"""
	Segment the spectrogram using hard thresholds on its ampltiude & derivative.

	TO DO: pass filtering functions
	"""
	p = {**default_seg_params, **seg_params}
	min_syll_len = int(np.floor(p['min_dur'] / dt))
	max_syll_len = min(p['num_time_bins'], int(np.ceil(p['max_dur'] / dt)))
	onsets, offsets = [], []
	# Get amplitude data and its derivative.
	time_smoothing = p['smoothing_timescale'] / dt
	smooth = gaussian_filter(spec, [p['freq_smoothing'], time_smoothing])
	amps = np.einsum('ij,i->j', smooth, p['freq_response'])
	# amps = np.mean(smooth, axis=0)
	filter = np.array([[1,0,-1]]) # Sobel filter
	amps_dot = convolve2d(smooth, filter, mode='same', boundary='symm')

	# amps_dot_dot = convolve2d(amps_dot, filter, mode='same', boundary='symm')
	# amps_dot_dot = np.mean(amps_dot_dot, axis=0)
	# amps_dot_dot = gaussian_filter1d(amps_dot_dot, 8.0)

	amps_dot = np.mean(amps_dot, axis=0)
	amps_dot = gaussian_filter1d(amps_dot, time_smoothing)

	# Normalize.
	amps -= np.min(amps)
	amps /= np.max(np.abs(amps))
	amps = gaussian_filter1d(amps, time_smoothing)
	amps_dot /= np.max(np.abs(amps_dot))
	# amps_dot_dot /= np.max(np.abs(amps_dot_dot))

	# Collect syllable times using hard thresholds for detecting onsets/offsets.
	a_onset = seg_params['a_onset']
	a_offset = seg_params['a_offset']
	# a_dot_onset = seg_params['a_dot_onset']
	# a_dot_offset = seg_params['a_dot_offset']
	min_var = seg_params['min_var']
	var_trace = np.zeros(len(amps))
	last = 'off'
	for i in range(1,len(amps)-1,1):
		if last == 'off':
			if amps[i] > a_onset:
				onsets.append(i)
				last = 'on'
		else:
			long_enough = i - onsets[-1] >= min_syll_len
			# amp_local_min = amps[i] == min(amps[i-1:i+2])
			if long_enough and amps[i] < a_offset:
				var = np.mean(np.var(spec[:,onsets[-1]:i], axis=0))
				var_trace[onsets[-1]:i] = var
				if var < min_var:
					onsets = onsets[:-1]
				else:
					offsets.append(i)
				last = 'off'
			elif i - onsets[-1] >= max_syll_len:
				last = 'off'
				onsets = onsets[:-1]
	onsets = onsets[:len(offsets)] # We may have picked up an unmatched onset.

	# Throw away syllables that are too long or too short.
	new_onsets = []
	new_offsets = []
	for i in range(len(offsets)):
		t1, t2 = onsets[i], offsets[i]
		if t2 - t1 + 1 <= max_syll_len and t2 - t1 + 1 >= min_syll_len:
			new_onsets.append(t1)
			new_offsets.append(t2)

	# Return everything else.
	if return_traces:
		return new_onsets, new_offsets, amps, amps_dot, var_trace
	return new_onsets, new_offsets


if __name__ == '__main__':
	pass

###
