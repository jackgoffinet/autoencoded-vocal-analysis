"""
Useful functions for preprocessing.

"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"


import numpy as np
from scipy.signal import stft
from scipy.interpolate import interp2d

EPSILON = 1e-12



def get_spec(t1, t2, audio, p, fs=32000, target_freqs=None, target_times=None, \
	fill_value=-1/EPSILON, max_dur=None):
	"""
	Norm, scale, threshold, strech, and resize a Short Time Fourier Transform.

	NOTE: <fill_value> necessary?

	Parameters
	----------
	t1 : float or None
		...

	t2 : float or None
		....

	audio : ...

	p : ...

	fs : ...

	target_freqs : ....

	fill_value : ....
	"""
	if max_dur is None:
		max_dur = p['max_dur']
	if t2 - t1 > max_dur + 1e-4:
		print("caught in spec")
		print(t1, t2, t2-t1)
		print(max_dur)
		return None, False
	s1, s2 = int(round(t1*fs)), int(round(t2*fs))
	# # Keep things divisible by reasonably large powers of 2.
	# remainder = (s2 - s1) % (p['nperseg'] - p['noverlap'])
	# if remainder != 0:
	# 	s2 += (p['nperseg'] - p['noverlap']) - remainder
	assert s1 < s2, "s1: " + str(s1) + " s2: " + str(s2) + " t1: " + str(t1) + \
			" t2: " + str(t2)
	# Get a spectrogram and define the interpolation object.
	temp = min(len(audio),s2) - max(0,s1)
	if temp < p['nperseg'] or s2 <= 0 or s1 >= len(audio):
		return np.zeros((p['num_freq_bins'], p['num_time_bins'])), True
	else:
		f, t, spec = stft(audio[max(0,s1):min(len(audio),s2)], fs=fs, \
			nperseg=p['nperseg'], noverlap=p['noverlap'])
	t += max(0,t1)
	spec = np.log(np.abs(spec) + EPSILON)
	interp = interp2d(t, f, spec, copy=False, bounds_error=False, \
		fill_value=fill_value)
	# Define target frequencies.
	if target_freqs is None:
		if p['mel']:
			target_freqs = np.linspace(mel(p['min_freq']), mel(p['max_freq']), \
				p['num_freq_bins'], endpoint=True)
			target_freqs = inv_mel(target_freqs)
		else:
			target_freqs = np.linspace(p['min_freq'], p['max_freq'], \
				p['num_freq_bins'], endpoint=True)
	# Define target times.
	if target_times is None:
		duration = t2 - t1
		if p['time_stretch']:
			duration = np.sqrt(duration * max_dur) # stretched duration
		shoulder = 0.5 * (max_dur - duration)
		target_times = np.linspace(t1-shoulder, t2+shoulder, p['num_time_bins'])
	# Then interpolate.
	interp_spec = interp(target_times, target_freqs, assume_sorted=True)
	spec = interp_spec
	# Normalize.
	spec -= p['spec_min_val']
	spec /= (p['spec_max_val'] - p['spec_min_val'])
	spec = np.clip(spec, 0.0, 1.0)
	# Within-syllable normalize.
	if p['within_syll_normalize']:
		spec -= np.percentile(spec, 10.0)
		spec[spec<0.0] = 0.0
		spec /= np.max(spec) + EPSILON
	return spec, True



if __name__ == '__main__':
	pass


###
