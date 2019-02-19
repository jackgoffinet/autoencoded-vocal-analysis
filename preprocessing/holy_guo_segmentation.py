"""
Implementation of algorithm described in Holy & Guo (2005)


"""
__author__ = "Jack Goffinet"
__date__ = "August-September 2018"

import numpy as np
from scipy.signal import medfilt, istft
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.stats import zscore



def get_onsets_offsets(spec, dt, seg_params, return_traces=False):
	"""
	Segment the spectrogram using thresholds on three acoustic features.

	"""
	bit_vector = get_syllables(spec, seg_params['f'], dt)
	onsets, offsets = get_syllable_times(bit_vector)
	if return_traces:
		tr_1 = medfilt(get_spectral_purities(spec), 7)
		tr_2 = medfilt(np.concatenate((get_spectral_discontinuities(spec), [0.0])), 7)
		return onsets, offsets, [tr_1, tr_2]
	return onsets, offsets



def get_spectral_purities(spectrogram):
	temp =  np.sum(spectrogram, axis=0)
	result = np.zeros(spectrogram.shape[1])
	max_vals = np.max(spectrogram, axis=0)
	for i in range(len(temp)):
		if temp[i] > 0.0:
			result[i] =  max_vals[i] / temp[i]
	return result

def get_mean_frequencies(spec, f):
	sums = np.sum(spec, axis=0)
	temp = np.einsum('ij,i->j', spec[:], f)
	for i in range(len(sums)):
		if sums[i] > 0.0:
			temp[i] = temp[i] / sums[i]
	return temp


def get_mean_frequencies2(spectrogram, f):
	return f[np.argmax(spectrogram[:-1], axis=1)]



def get_spectral_discontinuities(spec):
	sums = np.sum(spec, axis=0)
	spectrogram = np.zeros(spec.shape)
	for i in range(len(sums)):
		if sums[i] > 0.0:
			spectrogram[:,i] = spec[:,i] / sums[i]
	result = np.zeros(spectrogram.shape[1]-1)
	for t in range(len(result)):
		sums = []
		for delta_j in range(-6,7,1):
			i1 = max(-delta_j, 0)
			i2 = spectrogram.shape[0] - delta_j # could be > array length
			i3 = max(delta_j, 0)
			i4 = spectrogram.shape[0] + delta_j # could be >= array length
			summation = np.sum(np.abs(spectrogram[i1:i2, t+1] - spectrogram[i3:i4, t]))
			sums.append(summation)
		result[t] = min(sums)
	return result




def get_syllables(spectrogram, f, dt, filter_size=7):
	"""
	Return a bit vector representing the syllable structure

	...
	"""
	sp = medfilt(get_spectral_purities(spectrogram), filter_size)
	mf = medfilt(get_mean_frequencies(spectrogram, f=f), filter_size)
	sd = medfilt(get_spectral_discontinuities(spectrogram), filter_size)
	sp, mf = sp[:-1], mf[:-1]
	# See where conditions are met.
	result = np.where(sp > 0.1, 1, 0)      # paper: 0.25    katie: 0.3  ()was 0.05
	result *= np.where(mf > 35e3, 1, 0)     # paper: 35e3   katie: 45e3
	result *= np.where(sd < 1.3, 1, 0)      # paper: 1.0   katie: 0.85 (was commented out)
	# Weed out candidates less than 5ms.
	mask_len = int(round(0.007 / dt))       # paper: 0.005
	i = 0
	while i < len(result):
		if not result[i]:
			i += 1
		elif np.all(result[i:i+mask_len]):
			while i < len(result) and result[i]:
				i += 1
		else:
			result[i] = 0 # could be more aggressive
			i += 1
	result[-mask_len:] = np.all(result[-mask_len:])
	# Merge candidates closer than 30ms.
	mask_len = int(round(0.025 / dt))        # paper 0.03
	i = 0
	while i < len(result) - 1:
		if (not result[i]) or result[i+1]:
			i += 1
			continue
		for j in range(2, mask_len+1, 1):
			if i+j < len(result) and result[i+j]:
				result[i:i+j] = 1
				break
		i += mask_len
	return result


def get_syllable_times(bit_vector):
	onsets, offsets = [], []
	i = 0
	while i < len(bit_vector) and bit_vector[i]:
		i+= 1

	start = i
	state = 0
	while i < len(bit_vector):
		if state and not bit_vector[i]:
			onsets.append(start)
			offsets.append(i)
			state = 0
		elif not state and bit_vector[i]:
			start = i
			state = 1
		i += 1
	onsets = [max(i-2, 0) for i in onsets]
	offsets = [min(i+2, len(bit_vector)-1) for i in offsets]
	return onsets, offsets






if __name__ == '__main__':
	pass

###
