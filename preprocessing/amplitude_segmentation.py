from __future__ import print_function, division
"""
Amplitude based syllable segmentation.

TO DO: Add lambda filtering
TO DO: kwargs
TO DO: check imports
TO DO: polish this
"""
__author__ ="Jack Goffinet"
__date__ = "November 2018"

import matplotlib.pyplot as plt # TEMP!
plt.switch_backend('agg')

import numpy as np
from os import listdir
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.signal import convolve2d, stft, medfilt
from scipy.stats import linregress



# thresholds for syllable onset & offset
seg_params = {
	'th_1':0.12,
	'th_2':0.12,
	'min_var':0.2,
}
max_syll_len = 128
min_syll_len = 20


def get_onsets_offsets(spec, seg_params=seg_params, return_traces=False):
	"""
	Segment the spectrogram using hard thresholds on its ampltiude & derivative.

	TO DO: pass filtering functions
	"""
	onsets, offsets = [], []
	# Get amplitude data and its derivative.
	smooth = gaussian_filter(spec, [3,3])
	amps = np.mean(smooth, axis=0)
	filter = np.array([[1,0,-1]]) # Sobel filter
	amps_dot = convolve2d(smooth, filter, mode='same', boundary='symm')

	amps_dot_dot = convolve2d(amps_dot, filter, mode='same', boundary='symm')
	amps_dot_dot = np.mean(amps_dot_dot, axis=0)
	amps_dot_dot = gaussian_filter1d(amps_dot_dot, 8.0)

	amps_dot = np.mean(amps_dot, axis=0)
	amps_dot = gaussian_filter1d(amps_dot, 8.0)

	# Normalize.
	amps -= np.min(amps)
	amps /= np.max(np.abs(amps))
	amps = gaussian_filter1d(amps, 2.0)
	amps_dot /= np.max(np.abs(amps_dot))
	amps_dot_dot /= np.max(np.abs(amps_dot_dot))

	# Collect syllable times using hard thresholds for detecting onsets and
	# offsets.
	th_1 = seg_params['th_1']
	th_2 = seg_params['th_2']
	min_var = seg_params['min_var']
	var_trace = np.zeros(len(amps))
	last = 'off'
	for i in range(1,len(amps)-1,1):
		if last == 'off':
			if amps_dot[i] > th_1:
				onsets.append(i)
				last = 'on'
		else:
			long_enough = i - onsets[-1] >= min_syll_len
			quiet = amps[i] < th_2
			amp_local_min = amps[i] == min(amps[i-1:i+2])
			# if  long_enough and (quiet or (big_accel and amp_accel_local_max)):
			if long_enough and quiet and amp_local_min:
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
	if return_traces:
		return new_onsets, new_offsets, amps, amps_dot, var_trace
	return new_onsets, new_offsets




###
