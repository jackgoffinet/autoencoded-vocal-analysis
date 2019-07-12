"""
Extract syllable spectrograms from audio files.


"""
__author__ = "Jack Goffinet"
__date__ = "December 2018 - July 2019"


import h5py
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.io import wavfile, loadmat
from scipy.signal import stft
from skimage.transform import resize
from time import strptime, mktime, localtime


# Constants
EPSILON = 1e-12



def process_sylls(audio_dir, segment_dir, save_dir, p):
	"""
	Extract syllables from <audio_dir> and save to <save_dir>.

	Parameters
	----------

	Returns
	-------

	Notes
	-----

	"""
	sylls_per_file = p['sylls_per_file']
	num_freq_bins = p['num_freq_bins']
	num_time_bins = p['num_time_bins']
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	audio_filenames = [audio_dir + i for i in os.listdir(audio_dir) if is_audio_file(i)]
	write_file_num = 0
	syll_data = {
		'specs':[],
		'times':[],
		'onsets':[],
		'offsets':[],
		'audio_filenames':[],
	}
	print("Processing audio files in", load_dir)
	for audio_filename in audio_filenames:
		# Get a start time. (for continual recording)
		start_time = time_from_filename(audio_filename)
		# Get onsets and offsets.
		onsets, offsets = read_onsets_offsets_from_file(load_filename, dt)
		durations = offset_ts - onset_ts
		# Retrieve a spectrogram for each detected syllable.
		specs = get_syll_specs(onsets, offsets, audio_filename, p)
		# Add the remaining syllables to <syll_data>.
		syll_data['specs'] += specs
		syll_data['times'] += (start_time + onset_ts).tolist()
		syll_data['onsets'] += onsets.tolist()
		syll_data['offsets'] += offsets.tolist()
		syll_data['audio_filenames'] += len(onsets)*[audio_filename.split('/')[-1]]
		# Write files until we don't have enough syllables.
		while len(syll_data['times']) >= sylls_per_file:
			save_filename = "syllables_" + str(write_file_num).zfill(3) + '.hdf5'
			save_filename = os.path.join(save_dir, save_filename)
			with h5py.File(save_filename, "w") as f:
				# Zero-pad the spectrograms and add them to the file.
				temp = np.zeros((sylls_per_file, num_freq_bins, num_time_bins),
						dtype='float')
				syll_specs = syll_data['specs']
				for i in range(sylls_per_file):
					gap = max(0, (num_time_bins - syll_specs[i].shape[1]) // 2)
					temp[i,:,gap:gap+syll_specs[i].shape[1]] = syll_specs[i][:,:num_time_bins]
				f.create_dataset('specs', data=temp)
				# Then add the rest.
				for k in ['times', 'onsets', 'offsets']:
					f.create_dataset(k, data=np.array(syll_data[k][:sylls_per_file]))
				temp = [os.path.join(save_dir, i) for i in syll_data['audio_filenames'][:sylls_per_file]]
				f.create_dataset('audio_filenames', data=np.array(temp).astype('S'))
			# Remove the written data from temporary storage.
			for k in syll_data:
				syll_data[k] = syll_data[k][sylls_per_file:]
			write_file_num += 1
			# Stop if we've written <max_num_syllables>.
			if p['max_num_syllables'] is not None and write_file_num*sylls_per_file >= p['max_num_syllables']:
				return


def get_syll_specs(onsets, offsets, filename, p):
	"""
	Return the spectrograms corresponding to <onsets> and <offsets>.
	"""
	audio, fs = get_audio(filename, p)
	assert p['nperseg'] % 2 == 0
	rfft_freqs = np.linspace(0, fs/2, p['nperseg']//2, endpoint=True)
	if p['mel']:
		target_freqs = np.linspace(mel(p['min_freq']), mel(p['max_freq']), p['num_freq_bins'], endpoint=True)
		target_freqs = inv_mel(target_freqs)
		target_freqs[0] = p['min_freq'] # Correct for numerical errors.
		target_freqs[1] = p['max_freq']
	else:
		target_freqs = np.linspace(p['min_freq'], p['max_freq'], p['num_freq_bins'], endpoint=True)
	specs = []
	# For each syllable...
	for t1, t2 in zip(onsets, offsets):
		# Figure out how many time bins to place the syllable into.
		assert t1 < t2
		ratio = (t2-t1) / p['max_dur']
		if p['time_stretch']:
			ratio = np.sqrt(ratio)
		num_bins = int(round(ratio * p['num_time_bins']))
		if num_bins < 1 or num_bins > p['num_time_bins']:
			continue
		start_bin = (p['num_time_bins'] - num_bins) // 2
		# Do an RFFT for each bin.
		spec = np.zeros((p['num_freq_bins'], p['num_time_bins']))
		ts = np.linspace(t1, t2, num_bins, endpoint=True)
		for i, t in enumerate(ts):
			# Define a slice of the audio.
			s1 = (t * fs) - p['nperseg'] // 2
			s2 = (t * fs) + p['nperseg'] // 2
			fourier = np.fft.rfft(audio[s1:s2])
			# Interpolate to the target frequencies.
			interp = interp1d(rfft_freqs, fourier, kind='linear', \
					assume_sorted=True, fill_value=0.0, bounds_error=False)
			spec[:,start_bin+i] = interp(target_freqs)
		# # Within-syllable normalization.
		# temp_spec -= np.percentile(temp_spec, 10.0)
		# temp_spec[temp_spec<0.0] = 0.0
		# temp_spec /= np.max(temp_spec)
		# Switch to square root duration.
		specs.append(spec)
	return specs


def get_audio(filename, p, start_index=None, stop_index=None):
	"""Get a waveform and samplerate given a filename."""
	# Make sure the samplerate is correct and the audio is mono.
	if filename[-4:] == '.wav':
		fs, audio = wavfile.read(filename)
	elif filename[-4:] == '.mat':
		d = loadmat(filename)
		audio = d['spike2Chunk'].reshape(-1)
		fs = d['fs'][0,0]
	if len(audio.shape) > 1:
		audio = audio[0,:]
	if start_index is not None and stop_index is not None:
		start_index = max(start_index, 0)
		audio = audio[start_index:stop_index]
	return fs, audio


def tune_preprocessing_params(audio_dirs, segment_dirs, p, window_dur=None):
	"""Flip through spectrograms and tune preprocessing parameters."""
	audio_filenames = []
	for audio_dir in audio_dirs:
		audio_filenames += [os.path.join(audio_dir, i) for i in os.listdir(audio_dir) if is_audio_file(i)]
	if window_dur is None:
		window_dur = 2 * p['max_dur']
	audio_filenames = np.array(audio_filenames)
	# NOTE: HERE!
	# Keep tuning params...
	while True:
		for key in seg_params:
			# Skip non-tunable parameters.
			if key in ['num_time_bins', 'num_freq_bins'] or not is_number(seg_params[key]):
				continue
			temp = 'not number and not empty'
			while not is_number_or_empty(temp):
				temp = input('Set value for '+key+': ['+str(seg_params[key])+ '] ')
			if temp != '':
				seg_params[key] = float(temp)
		# Visualize segmenting decisions.
		temp = 'not q or r'
		while temp != 'q' and temp != 'r':
			file_index = np.random.randint(len(filenames))
			filename = filenames[file_index]
			# Get spec & onsets/offsets.
			start_index = np.random.randint(file_lens[file_index] - 3*dur_samples)
			stop_index = start_index + 3*dur_samples
			spec, f, dt = get_spec(filename, p, start_index=start_index, stop_index=stop_index)
			# if 'f' not in seg_params:
			# 	seg_params['f'] = f
			if seg_params['algorithm'] == read_onsets_offsets_from_file:
				onsets, offsets = read_onsets_offsets_from_file(filename, dt, seg_params)
				traces = []
				temp = int(round(start_index/fs/dt))
				onsets = [i-temp for i in onsets]
				offsets = [i-temp for i in offsets]
			else:
				onsets, offsets, traces = seg_params['algorithm'](spec, dt, seg_params=seg_params, return_traces=True)
			dur_t_bins = int(dur_seconds / dt)
			# Plot.
			i1 = dur_t_bins
			i2 = 2 * dur_t_bins
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
			for key in ['th_1', 'th_2', 'th_3']:
				if key in seg_params:
					axarr[1].axhline(y=seg_params[key], lw=0.5, c='b')
			xvals = np.linspace(t1, t2, i2-i1)
			for trace in traces:
				axarr[1].plot(xvals, trace[i1:i2])
			plt.savefig('temp.pdf')
			plt.close('all')
			if len([j for j in onsets if j>i1 and j<i2]) > 0:
				temp = input('Continue? [y] or [q]uit or [r]etune params: ')
			else:
				print("searching")
				temp = 'not q or r'
			if temp == 'q':
				return seg_params



def read_onsets_offsets_from_file(txt_filename, p):
	"""Read a text file to collect onsets and offsets."""
	delimiter, skiprows, usecols = p['delimiter'], p['skiprows'], p['usecols']
	onsets, offsets = np.loadtxt(txt_filename, delimiter=delimiter, \
					skiprows=skiprows, usecols=usecols, unpack=True)
	return onsets, offsets


def mel(a):
	return 1127 * np.log(1 + a / 700)


def inv_mel(a):
	return 700 * (np.exp(a / 1127) - 1)


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


def is_audio_file(fn):
	return len(fn) >= 4 and fn[-4:] in ['.wav', '.mat']


def get_wav_len(filename):
	if filename[-4:] == '.wav':
		_, audio = wavfile.read(filename)
	elif filename[-4:] == '.mat':
		audio = loadmat(filename)['spike2Chunk'].reshape(-1)
	else:
		raise NotImplementedError
	return len(audio)


def time_from_filename(filename):
	"""Return time in seconds, following SAP conventions."""
	try:
		anchor = mktime(strptime("1899 12 29 19", "%Y %m %d %H")) #SAP anchor time
		temp = filename.split('/')[-1].split('_')[1].split('.')
		day = float(temp[0])
		millisecond = float(temp[1])
		time = anchor + 24*60*60*day + 1e-3*millisecond
		return time
	except:
		return 0.0



if __name__ == '__main__':
	pass


###
