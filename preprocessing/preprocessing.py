"""
Compute and process syllable spectrograms.

"""
__author__ = "Jack Goffinet"
__date__ = "December 2018 - July 2019"


import h5py
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from scipy.interpolate import interp1d, interp2d
from scipy.io import wavfile, loadmat
from scipy.signal import stft
from time import strptime, mktime, localtime # NOTE: move this later in the pipeline.
import warnings

# Silence numpy.loadtxt when reading empty files.
warnings.filterwarnings("ignore", category=UserWarning)


EPSILON = 1e-12



def process_sylls(audio_dir, segment_dir, save_dir, p):
	"""
	Extract syllables from <audio_dir> and save to <save_dir>.

	Parameters
	----------
	audio_dir : str

	segment_dir : str

	save_dir : str

	p : str

	Returns
	-------

	Notes
	-----

	"""
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	audio_filenames, seg_filenames = get_audio_seg_filenames(audio_dir, segment_dir, p)
	write_file_num = 0
	syll_data = {
		'specs':[],
		'times':[], # NOTE: move this later in the pipeline.
		'onsets':[],
		'offsets':[],
		'audio_filenames':[],
	}
	print("Processing audio files in", audio_dir)
	sylls_per_file = p['sylls_per_file']
	for audio_filename, seg_filename in zip(audio_filenames, seg_filenames):
		# Get a start time. (for chronic recordings)
		start_time = time_from_filename(audio_filename)
		# Get onsets and offsets.
		onsets, offsets = read_onsets_offsets_from_file(seg_filename, p)
		# Retrieve a spectrogram for each detected syllable.
		specs, good_sylls = get_syll_specs(onsets, offsets, audio_filename, p)
		onsets = [onsets[i] for i in good_sylls]
		offsets = [offsets[i] for i in good_sylls]
		# onsets, offsets = onsets[good_sylls], offsets[good_sylls]
		# Add the remaining syllables to <syll_data>.
		syll_data['specs'] += specs
		syll_data['times'] += [start_time + t for t in onsets]
		syll_data['onsets'] += onsets
		syll_data['offsets'] += offsets
		syll_data['audio_filenames'] += len(onsets)*[os.path.split(audio_filename)[-1]]
		# Write files until we don't have enough syllables.
		while len(syll_data['times']) >= sylls_per_file:
			save_filename = "syllables_" + str(write_file_num).zfill(4) + '.hdf5'
			save_filename = os.path.join(save_dir, save_filename)
			with h5py.File(save_filename, "w") as f:
				# Add all the fields.
				for k in ['times', 'onsets', 'offsets']:
					f.create_dataset(k, \
						data=np.array(syll_data[k][:sylls_per_file]))
				f.create_dataset('specs', \
					data=np.stack(syll_data['specs'][:sylls_per_file]))
				temp = [os.path.join(save_dir, i) for i in \
					syll_data['audio_filenames'][:sylls_per_file]]
				f.create_dataset('audio_filenames', \
					data=np.array(temp).astype('S')) # special string format
			write_file_num += 1
			# Remove the written data from temporary storage.
			for k in syll_data:
				syll_data[k] = syll_data[k][sylls_per_file:]
			# Stop if we've written <max_num_syllables>.
			if p['max_num_syllables'] is not None and \
					write_file_num*sylls_per_file >= p['max_num_syllables']:
				return


def get_spec(t1, t2, audio, p, fs=32000, target_freqs=None, \
	fill_value=-1/EPSILON):
	"""
	Norm, scale, threshold, strech, and resize a Short Time Fourier Transform.

	Parameters
	----------
	t1 : float or None
	"""
	s1, s2 = int(round(t1*fs)), int(round(t2*fs))
	assert s2 <= len(audio)
	if t2 - t1 > p['max_dur'] + 1e-4:
		return None, False
	# Keep things divisible by reasonably large powers of 2 when possible.
	remainder = (s2 - s1) % (p['nperseg'] - p['noverlap'])
	if remainder != 0:
		s2 += (p['nperseg'] - p['noverlap']) - remainder
	assert s1 < s2, "s1: " + str(s1) + " s2: " + str(s2)
	# Get a spectrogram and define the interpolation object.
	f, t, spec = stft(audio[s1:s2], fs=fs, nperseg=p['nperseg'], \
		noverlap=p['noverlap'])
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
	duration = t2 - t1
	if p['time_stretch']:
		duration = np.sqrt(duration * p['max_dur']) # fake duration
	shoulder = 0.5 * (p['max_dur'] - duration)
	target_times = np.linspace(-shoulder, t2-t1+shoulder, p['num_time_bins'], \
		endpoint=True)
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


def get_syll_specs(onsets, offsets, audio_filename, p):
	"""
	Return the spectrograms corresponding to <onsets> and <offsets>.

	Parameters
	----------
	onsets :

	offsets :

	audio_filename :

	p : dictionary
		A dictionary mapping preprocessing parameters to their values.
		NOTE: ADD REFERENCE HERE

	Returns
	-------
	specs :

	valid_syllables : numpy.ndarray

	"""
	fs, audio = get_audio(audio_filename, p)
	assert p['nperseg'] % 2 == 0 and p['nperseg'] > 2
	if p['mel']:
		target_freqs = np.linspace(mel(p['min_freq']), mel(p['max_freq']), p['num_freq_bins'], endpoint=True)
		target_freqs = inv_mel(target_freqs)
	else:
		target_freqs = np.linspace(p['min_freq'], p['max_freq'], p['num_freq_bins'], endpoint=True)
	specs, valid_syllables = [], []
	# For each syllable...
	for i, t1, t2 in zip(range(len(onsets)), onsets, offsets):
		spec, valid = get_spec(t1, t2, audio, p, fs, target_freqs=target_freqs)
		if valid:
			valid_syllables.append(i)
			specs.append(spec)
	return specs, valid_syllables


def get_audio(filename, p, start_index=None, stop_index=None):
	"""Get a waveform and samplerate given a filename."""
	# Make sure the samplerate is correct and the audio is mono.
	if filename[-4:] == '.wav':
		fs, audio = wavfile.read(filename)
	elif filename[-4:] == '.mat':
		d = loadmat(filename)
		audio = d['spike2Chunk'].reshape(-1)
		fs = d['fs'][0,0]
	else:
		raise NotImplementedError
	if len(audio.shape) > 1:
		audio = audio[0,:]
	if start_index is not None and stop_index is not None:
		start_index = max(start_index, 0)
		audio = audio[start_index:stop_index]
	return fs, audio


def tune_preprocessing_params(audio_dirs, seg_dirs, p, window_dur=None):
	"""Flip through spectrograms and tune preprocessing parameters."""
	# Collect all the relevant filenames.
	audio_filenames, seg_filenames = [], []
	for audio_dir, seg_dir in zip(audio_dirs, seg_dirs):
		temp_audio, temp_seg = get_audio_seg_filenames(audio_dir, seg_dir, p)
		audio_filenames += temp_audio
		seg_filenames += temp_seg
	audio_filenames = np.array(audio_filenames)
	seg_filenames = np.array(seg_filenames)
	# Set the window around the syllable to display.
	if window_dur is None:
		window_dur = 2 * p['max_dur']
	# Main loop: keep tuning parameters ...
	while True:
		for key in p['real_preprocess_params']:
			temp = 'not number and not empty'
			while not is_number_or_empty(temp):
				temp = input('Set value for '+key+': ['+str(p[key])+ '] ')
			if temp != '':
				p[key] = float(temp)
		for key in p['int_preprocess_params']:
			temp = 'not number and not empty'
			while not is_number_or_empty(temp):
				temp = input('Set value for '+key+': ['+str(p[key])+ '] ')
			if temp != '':
				p[key] = int(temp)
		for key in p['binary_preprocess_params']:
			temp = 'not t and f'
			while temp not in ['t', 'T', 'f', 'F', '']:
				current_value = 'T' if p[key] else 'F'
				temp = input('Set value for '+key+': ['+current_value+'] ')
			if temp != '':
				p[key] = temp in ['t', 'T']
		# Keep plotting example spectrograms.
		temp = 'not (q or r)'
		while temp != 'q' and temp != 'r':
			# Grab a random file.
			file_index = np.random.randint(len(audio_filenames))
			audio_filename = audio_filenames[file_index]
			seg_filename = seg_filenames[file_index]
			# Grab a random syllable from within the file.
			onsets, offsets = read_onsets_offsets_from_file(seg_filename, p)
			if len(onsets) == 0:
				continue
			syll_index = np.random.randint(len(onsets))
			onsets, offsets = [onsets[syll_index]], [offsets[syll_index]]
			# If this is a sliding window, get a random onset & offset.
			if p['sliding_window']:
				onsets = [onsets[0] + (offsets[0] - onsets[0]) * np.random.rand()]
				offsets = [onsets[0] + p['window_length']]
			# Get the preprocessed spectrogram.
			specs, good_sylls = get_syll_specs(onsets, offsets, audio_filename, p)
			specs = [specs[i] for i in good_sylls]
			if len(specs) == 0:
				continue
			spec = specs[0]
			# Plot.
			plt.imshow(spec, aspect='equal', origin='lower', vmin=0, vmax=1)
			plt.axis('off')
			plt.savefig('temp.pdf')
			plt.close('all')
			temp = input('Continue? [y] or [q]uit or [r]etune params: ')
			if temp == 'q':
				return p


def get_audio_seg_filenames(audio_dir, segment_dir, p):
	temp_filenames = [i for i in sorted(os.listdir(audio_dir)) if is_audio_file(i)]
	audio_filenames = [os.path.join(audio_dir, i) for i in temp_filenames]
	temp_filenames = [i[:-4] + p['seg_extension'] for i in temp_filenames]
	seg_filenames = [os.path.join(segment_dir, i) for i in temp_filenames]
	return audio_filenames, seg_filenames


def read_onsets_offsets_from_file(txt_filename, p):
	"""Read a text file to collect onsets and offsets."""
	delimiter, skiprows, usecols = p['delimiter'], p['skiprows'], p['usecols']
	segs = np.loadtxt(txt_filename, delimiter=delimiter, skiprows=skiprows, \
		usecols=usecols).reshape(-1,2)
	return segs[:,0], segs[:,1]


def mel(a):
	"""https://en.wikipedia.org/wiki/Mel-frequency_cepstrum"""
	return 1127 * np.log(1 + a / 700)


def inv_mel(a):
	"""https://en.wikipedia.org/wiki/Mel-frequency_cepstrum"""
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
