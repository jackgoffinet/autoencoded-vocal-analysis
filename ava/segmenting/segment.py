"""
Segment audio files and write segmenting decisions.

TO DO:
	- Toggle booleans
	- Plot in kHz, with labels
	- tune window size
	- segment could be sped up if it operated file by file.

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

from ava.segmenting.utils import get_spec

# Constants
EPSILON = 1e-12



def segment(audio_dir, seg_dir, p, verbose=True):
	"""
	Segment audio files in audio_dir and write decisions to seg_dir.

	Parameters
	----------
	audio_dir : str
		Directory containing audio files.
	seg_dir : str
		Directory containing segmenting decisions.
	p : dict
		Segmenting parameters. TO DO: ADD REFERENCE!
	verbose : bool, optional
		Defaults to ``False``.

	"""
	if verbose:
		print("\nSegmenting audio in", audio_dir+"\n"+'-'*(20+len(audio_dir)))
	if not os.path.exists(seg_dir):
		os.makedirs(seg_dir)
	num_sylls = 0
	audio_fns, seg_fns = get_audio_seg_filenames(audio_dir, seg_dir, p)
	for audio_fn, seg_fn in zip(audio_fns, seg_fns):
		# Collect audio.
		fs, audio = wavfile.read(audio_fn)
		# Segment.
		onsets, offsets = p['algorithm'](audio, p)
		combined = np.stack([onsets, offsets]).T
		num_sylls += len(combined)
		# Write.
		header = "Onsets/offsets for " + audio_fn
		np.savetxt(seg_fn, combined, fmt='%.5f', header=header)
	if verbose:
		print("Found", num_sylls, "segments in", audio_dir)


def tune_segmenting_params(load_dirs, p):
	"""
	Tune segementing parameters by visualizing segmenting decisions.

	Parameters
	----------
	load_dirs : list of str
		Directories containing audio files.
	p : dict
		Segmenting parameters. TO DO: ADD REFERENCE!

	Returns
	-------
	p : dict
		Adjusted segmenting parameters

	"""
	print("Tune segmenting parameters\n---------------------------")
	# Collect filenames.
	filenames = []
	for load_dir in load_dirs:
		filenames += [os.path.join(load_dir, i) for i in os.listdir(load_dir) \
				if is_audio_file(i)]
	if len(filenames) == 0:
		warnings.warn("Found no audio files in directories: "+str(load_dirs))
		return
	# Set the amount of audio to display.
	if 'window_dur' in p:
		window_dur = p['window_dur']
	else:
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
		temp = 'not (s or r)'
		iteration = 0
		while temp != 's' and temp != 'r':

			# Get a random audio file.
			file_index = np.random.randint(len(filenames))
			filename = filenames[file_index]

			# Get spectrogram.
			fs, audio = wavfile.read(filename)
			assert fs == p['fs'], 'Found fs='+str(fs)+', expected '+str(p['fs'])
			if len(audio) < 3*window_samples:
				temp = len(audio) / p['fs']
				warnings.warn( \
						"Skipping short file: "+filename+" ("+str(temp)+"s)")
				continue
			start_index = np.random.randint(len(audio) - 3*window_samples)
			stop_index = start_index + 3*window_samples
			audio = audio[start_index:stop_index]
			spec, dt, f = get_spec(audio, p)

			# Get onsets and offsets.
			onsets, offsets, traces = \
					p['algorithm'](audio, p, return_traces=True)
			onsets = [onset/dt for onset in onsets]
			offsets = [offset/dt for offset in offsets]

			# Plot.
			i1 = int(window_dur / dt)
			i2 = 2 * i1
			t1, t2 = i1 * dt, i2 * dt
			_, axarr = plt.subplots(2,1, sharex=True)
			axarr[0].set_title(filename)
			axarr[0].imshow(spec[:,i1:i2], origin='lower', \
					aspect='auto', \
					extent=[t1, t2, f[0]/1e3, f[-1]/1e3])
			axarr[0].set_ylabel('Frequency (kHz)')
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
			axarr[1].set_xlabel('Time (s)')
			plt.savefig('temp.pdf')
			plt.close('all')

			# Continue.
			all_events = [j for j in onsets if j>i1 and j<i2] + \
					[j for j in offsets if j>i1 and j<i2]
			if len(all_events) > 0 or (iteration+1) % 5 == 0:
				temp = input('Continue? [y] or [s]top tuning or [r]etune params: ')
			else:
				iteration += 1
				print("searching")
				temp = 'not (s or r)'
			if temp == 's':
				return p


def get_audio_seg_filenames(audio_dir, segment_dir, p):
	"""Return lists of sorted filenames."""
	temp_filenames = [i for i in sorted(os.listdir(audio_dir)) if \
			is_audio_file(i)]
	audio_filenames = [os.path.join(audio_dir, i) for i in temp_filenames]
	temp_filenames = [i[:-4] + '.txt' for i in temp_filenames]
	seg_filenames = [os.path.join(segment_dir, i) for i in temp_filenames]
	return audio_filenames, seg_filenames


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
