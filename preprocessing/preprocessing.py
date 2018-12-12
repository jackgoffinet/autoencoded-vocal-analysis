from __future__ import print_function, division
"""
Process syllables and save data.

"""
__author__ = "Jack Goffinet"
__date__ = "December 2018"


import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
import h5py
from tqdm import tqdm
from scipy.interpolate import interp1d
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from .amplitude_segmentation import get_onsets_offsets

# Constants
EPS = 1e-12

# Default parameters
default_params = {
	# Spectrogram parameters
	'fs': 44100,
	'min_freq': 50,
	'max_freq': 15e3,
	'nperseg': 512,
	'noverlap': 0,
	'spec_percentile': 90.0,
	'num_freq_bins': 128,
	'num_time_bins': 128,
	'spacing': 'mel',
	# Segmenting parameters
	'seg_params': {},
	# I/O parameters
	'max_num_files': 100,
	'sylls_per_file': 1000,
	'meta': {},
}



def process_sylls(load_dir, save_dir, params):
	"""
	Process files in <load_dir> and save to <save_dir>.
	"""
	p = {**default_params, **params}
	meta = p['meta']
	max_num_files = p['max_num_files']
	sylls_per_file = p['sylls_per_file']
	num_freq_bins = p['num_freq_bins']
	num_time_bins = p['num_time_bins']
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	if 'load_dir' not in meta:
		meta['load_dir'] = load_dir
	filenames = [load_dir + i for i in os.listdir(load_dir) if i[-4:] == '.wav']
	np.random.shuffle(filenames)
	filenames = filenames[:max_num_files]
	write_file_num = 0
	onsets, offsets, syll_specs, syll_lens, syll_times = [], [], [], [], []
	print("Processing .wav files in", load_dir)
	start_times = [] # TEMP
	for load_filename in tqdm(filenames):
		start_time = time_from_filename(load_filename)
		# Get a spectrogram.
		spec, f, dt, i1, i2 = get_spec(load_filename, p)
		try:
			t_onsets, t_offsets = get_onset_offsets_from_file(load_filename, dt) # TEMP
		except:
			print(load_filename)
			quit()
		# # Collect syllable onsets and offsets.
		# t_onsets, t_offsets = get_onsets_offsets(spec, dt, p['seg_params'])
		# Retrieve spectrograms for each detected syllable.
		t_syll_specs, t_syll_lens , t_syll_times = get_syll_specs(t_onsets,
				t_offsets, spec, start_time, dt, p)
		for arr, t_arr in zip(
				[onsets, offsets, syll_specs, syll_lens, syll_times],
				[t_onsets, t_offsets, t_syll_specs, t_syll_lens, t_syll_times]):
			arr += t_arr
		# Write a file when we have enough syllables.
		while len(onsets) >= sylls_per_file:
			save_filename = save_dir + "syllables_"
			save_filename += str(write_file_num).zfill(3) + '.hdf5'
			with h5py.File(save_filename, "w") as f:
				temp = np.zeros((sylls_per_file, num_freq_bins, num_time_bins),
						dtype='float')
				for i in range(sylls_per_file):
					gap = (num_time_bins - syll_specs[i].shape[1]) // 2
					temp[i,:,gap:gap+syll_specs[i].shape[1]] = syll_specs[i]
				f.create_dataset('syll_specs', data=temp)
				f.create_dataset('syll_lens',
						data=np.array(syll_lens[:sylls_per_file]))
				f.create_dataset('syll_times',
						data=np.array(syll_times[:sylls_per_file]))
				for key in meta:
					f.attrs[key] = meta[key]
			onsets = onsets[sylls_per_file:]
			offsets = offsets[sylls_per_file:]
			syll_specs = syll_specs[sylls_per_file:]
			syll_lens = syll_lens[sylls_per_file:]
			syll_times = syll_times[sylls_per_file:]
			write_file_num += 1


def get_syll_specs(onsets, offsets, spec, start_time, dt, params):
	"""
	Return a list of spectrograms, one for each syllable.
	"""
	p = {**default_params, **params}
	syll_specs, syll_lens, syll_times = [], [], []
	# For each syllable...
	for t1, t2 in zip(onsets, offsets):
		# Take a slice of the spectrogram.
		temp_spec = spec[:,t1:t2+1]
		# Within-syllable normalization.
		try:
			temp_spec -= np.percentile(temp_spec, 10.0)
		except:
			print("get_syll_specs")
			print(temp_spec.shape)
			quit()
		temp_spec[temp_spec<0.0] = 0.0
		temp_spec /= np.max(temp_spec)

		# Switch to sqrt duration.
		new_dur = int(round(temp_spec.shape[1]**0.5 * p['num_time_bins']**0.5))
		temp_spec = resize(temp_spec, (temp_spec.shape[0], new_dur), anti_aliasing=True)

		# Collect spectrogram, duration, & onset time.
		syll_specs.append(temp_spec)
		syll_lens.append(temp_spec.shape[1])
		syll_times.append(start_time + (t1 * dt)/(24*60*60)) # in days
	return syll_specs, syll_lens, syll_times


def get_spec(filename, params, start_index=None, end_index=None):
	"""Get a spectrogram given a filename."""
	p = {**default_params, **params}
	# Make sure the samplerate is correct and the audio is mono.
	temp_fs, audio = wavfile.read(filename)
	assert(temp_fs == p['fs'])
	if len(audio.shape) > 1:
		audio = audio[0,:]
	if start_index is not None and end_index is not None:
		start_index = max(start_index, 0)
		audio = audio[start_index:end_index]

	# Convert to a magnitude-only spectrogram.
	f, t, Zxx = stft(audio, fs=p['fs'], nperseg=p['nperseg'],
			noverlap=p['noverlap'])
	i1 = np.searchsorted(f, p['min_freq'])
	i2 = np.searchsorted(f, p['max_freq'])
	f = f[i1:i2]
	spec = np.log(np.abs(Zxx[i1:i2,:]) + EPS)

	# Denoise.
	spec_thresh = np.percentile(spec, p['spec_percentile'])
	spec -= spec_thresh
	spec[spec<0.0] = 0.0

	# Switch to mel frequency spacing.
	mel_f = np.linspace(mel(f[0]), mel(f[-1]), p['num_freq_bins'], endpoint=True)
	mel_f = inv_mel(mel_f)
	mel_f[0] = f[0] # Correct for numerical errors.
	mel_f[-1] = f[-1]
	mel_f_spec = np.zeros((p['num_freq_bins'], spec.shape[1]), dtype='float')
	for j in range(spec.shape[1]):
		interp = interp1d(f, spec[:,j], kind='cubic')
		mel_f_spec[:,j] = interp(mel_f)
	spec = mel_f_spec

	return spec, mel_f, t[1]-t[0], i1, i2


def tune_segmenting_params(load_dirs, params):
	"""Tune params by visualizing segmenting decisions."""
	params = {**default_params, **params}
	seg_params = params['seg_params']
	filenames = []
	for load_dir in load_dirs:
		filenames += [load_dir + i for i in os.listdir(load_dir) if i[-4:] == '.wav']
	file_num = 0
	file_len = get_wav_len(filenames[file_num])
	# Keep tuning params...
	while True:
		for key in seg_params:
			# Skip non-tunable parameters.
			if key in ['num_time_bins', 'num_freq_bins', 'freq_response']:
				continue
			temp = 'not a valid option'
			while not is_number_or_empty(temp):
				temp = input('Set value for '+key+': ['+str(seg_params[key])+ '] ')
			if temp != '':
				seg_params[key] = float(temp)

		# Visualize segmenting decisions.
		temp = ''
		i = 0
		dur_seconds = 2.0
		dur_samples = int(dur_seconds * params['fs'])
		while temp != 'q' and temp != 'r':
			if (i+1)*dur_samples < file_len:
				# Get spec & onsets/offsets.
				spec, f, dt, _, _ = get_spec(filenames[file_num], params, \
						start_index=(i-1)*dur_samples, end_index=(i+2)*dur_samples)
				onsets, offsets, tr_1, tr_2, tr_3 = get_onsets_offsets(spec, \
						dt, seg_params=seg_params, return_traces=True)
				dur = int(dur_seconds / dt)
				# Plot.
				_, axarr = plt.subplots(2,1, sharex=True)
				i1 = dur
				if i == 0:
					i1 = 0
				i2 = i1 + dur
				axarr[0].imshow(spec[:,i1:i2], origin='lower', \
						aspect='auto', \
						extent=[i*dur_seconds, (i+1)*dur_seconds, f[0], f[-1]])
				for j in range(len(onsets)):
					if onsets[j] >= i1 and onsets[j] < i2:
						for k in [0,1]:
							# time = onsets[j] * dt
							time = i*dur_seconds + (onsets[j] - i1) * dt
							axarr[k].axvline(x=time, c='b', lw=0.5)
					if offsets[j] >= i1 and offsets[j] < i2:
						for k in [0,1]:
							time = i*dur_seconds + (offsets[j] - i1) * dt
							axarr[k].axvline(x=time, c='r', lw=0.5)
				axarr[1].axhline(y=seg_params['a_onset'], lw=0.5, c='b')
				axarr[1].axhline(y=seg_params['a_offset'], lw=0.5, c='b')
				# axarr[1].axhline(y=seg_params['a_dot_onset'], lw=0.5, c='r')
				# axarr[1].axhline(y=seg_params['a_dot_offset'], lw=0.5, c='r')
				axarr[1].axhline(y=seg_params['min_var'], lw=0.5, c='g')
				axarr[1].axhline(y=0.0, lw=0.5, c='k')
				xvals = np.linspace(i*dur_seconds, (i+1)*dur_seconds, dur)
				axarr[1].plot(xvals, tr_1[i1:i2], c='b')
				axarr[1].plot(xvals, tr_2[i1:i2], c='r')
				axarr[1].plot(xvals, tr_3[i1:i2], c='g', alpha=0.2)
				plt.savefig('temp.pdf')
				plt.close('all')
				i += 1
			elif file_num != len(filenames)-1:
				i = 0
				file_num += 1
				file_len = get_wav_len(filenames[file_num])
			else:
				print('Reached the end of the files...')
				return
			temp = input('Continue? [y] or [q]uit or [r]etune params: ')
		if temp == 'q':
			return seg_params


def get_onset_offsets_from_file(audio_filename, dt):
	filename = audio_filename.split('.')[0] + '.txt'
	d = np.loadtxt(filename)
	onsets = []
	offsets = []
	for i in range(len(d)):
		onsets.append(int(np.floor(d[i,1]/dt)))
		offsets.append(int(np.ceil(d[i,2]/dt)))
		if offsets[-1] - onsets[-1] >= 128:
			onsets = onsets[:-1]
			offsets = offsets[:-1]
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


def get_wav_len(filename):
	_, audio = wavfile.read(filename)
	return len(audio)


def time_from_filename(filename):
	"""Return time in seconds, from 8/15 0h0m0s"""
	try:
		mon, day, hr, min, sec = filename.split('/')[-1].split('_')[2:]
		sec = sec.split('.')[0] # remove .wav
		time = 0.0
		for unit, in_secs in zip([day, hr, min, sec], [1., 1./24, 1./1440, 1./86400]):
			time += float(unit) * in_secs
		return time
	except:
		return 0.0


if __name__ == '__main__':
	pass


###
