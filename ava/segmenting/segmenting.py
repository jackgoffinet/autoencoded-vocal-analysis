"""
Segment audio files and write segmenting decisions.


"""
__author__ = "Jack Goffinet"
__date__ = "December 2018 - August 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from scipy.io import wavfile, loadmat
from scipy.signal import stft
import warnings


# Constants
EPSILON = 1e-12



def segment(audio_dirs, seg_dirs, p):
	"""Segment audio files in audio_dirs and write decisions to seg_dirs."""
	pass


def tune_segmenting_params(load_dirs, p, window_dur=None):
	"""
	Tune segementing parameters by visualizing segmenting decisions.

	Parameters
	----------
	load_dirs : list of str
		Directories containing audio files.

	p : dict
		Segmenting parameters.

	window_dur : float or None, optional
		The length of audio plotted. If None, this is set to 2*p['max_dur'].
		Defaults to None.

	Returns
	-------
	p : dict
		Adjusted segmenting parameters
	"""
	# Collect filenames.
	filenames = []
	for load_dir in load_dirs:
		filenames += [os.path.join(load_dir, i) for i in os.listdir(load_dir) \
				if is_audio_file(i)]
	if len(filenames) == 0:
		warnings.warn("Found no audio files in directories: "+str(load_dirs))
		return
	# Set the amount of audio to display.
	if window_dur is None:
		window_dur = 2.0 * p['max_dur']
	window_samples = int(window_dur * p['fs'])

	# Main loop: keep tuning parameters...
	while True:

		# Tune the parameters.
		for key in p:
			# Skip non-tunable parameters.
			if key in ['num_time_bins', 'num_freq_bins'] or not is_number(p[key]):
				continue
			temp = 'not number and not empty'
			while not is_number_or_empty(temp):
				temp = input('Set value for '+key+': ['+str(p[key])+ '] ')
			if temp != '':
				p[key] = float(temp)

		# Visualize segmenting decisions.
		temp = 'not (q or r)'
		iteration = 0
		while temp != 'q' and temp != 'r':

			# Get a random audio file.
			file_index = np.random.randint(len(filenames))
			filename = filenames[file_index]

			# Get spectrogram.
			_, audio = wavfile.read(filename)
			if len(audio) < 3*window_samples:
				temp = len(audio) / p['fs']
				warnings.warn( \
						"Skipping short file: "+filename+" ("+str(temp)+"s)")
				continue
			start_index = np.random.randint(len(audio) - 3*window_samples)
			stop_index = start_index + 3*window_samples
			audio = audio[start_index:stop_index]
			spec, f, dt = get_spec(audio, p)

			# Get onsets and offsets.
			onsets, offsets, traces = \
					p['algorithm'](audio, p, return_traces=True)

			# Plot.
			i1 = int(window_dur / dt)
			i2 = 2 * i1
			t1, t2 = i1 * dt, i2 * dt
			_, axarr = plt.subplots(2,1, sharex=True)
			axarr[0].set_title(filename)
			axarr[0].imshow(spec[:,i1:i2], origin='lower', \
					aspect='auto', \
					extent=[t1, t2, f[0], f[-1]])
			for j in range(len(onsets)):
				if onsets[j] >= i1 and onsets[j] < i2:
					time = onsets[j] * dt
					for k in [0,1]:
						axarr[k].axvline(x=time, c='b', lw=0.5)
				if offsets[j] >= i1 and offsets[j] < i2:
					time = offsets[j] * dt
					for k in [0,1]:
						axarr[k].axvline(x=time, c='r', lw=0.5)
			for key in ['th_1', 'th_2', 'th_3']: # NOTE: clean this
				if key in p:
					axarr[1].axhline(y=p[key], lw=0.5, c='b')
			xvals = np.linspace(t1, t2, i2-i1)
			for trace in traces:
				axarr[1].plot(xvals, trace[i1:i2])
			plt.savefig('temp.pdf')
			plt.close('all')

			# Continue.
			all_events = [j for j in onsets if j>i1 and j<i2] + \
					[j for j in offsets if j>i1 and j<i2]
			if len(all_events) > 0 or (iteration+1) % 5 == 0:
				temp = input('Continue? [y] or [q]uit or [r]etune params: ')
			else:
				iteration += 1
				print("searching")
				temp = 'not (q or r)'
			if temp == 'q':
				return p


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


def is_audio_file(fn):
	return len(fn) >= 4 and fn[-4:] == '.wav'


def is_number_or_empty(s):
	if s == '':
		return True
	try:
		float(s)
		return True
	except:
		return False


def is_number(s):
	return type(s) == type(4) or type(s) == type(4.0)



if __name__ == '__main__':
	pass


###
