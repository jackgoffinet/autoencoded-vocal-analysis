from __future__ import print_function, division
"""
Process syllables and save data.

TO DO: denoise?

TO DO: change variable scopes, with optinoal passing
"""
__author__ = "Jack Goffinet"
__date__ = "November 2018"


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

# Parameters
# NOTE: These are only default values that are rendered out of scope in the
# methods below if keyword arguments are used.
EPS = 1e-12
nperseg, noverlap = 512, 512-128-64
fs = 44100
spec_percentile = 80.0
min_freq, max_freq = 50, 15e3
num_freq_bins = 128
sylls_per_file = 1000
seg_params = {
	'th_1':0.12,
	'th_2':0.12,
	'th_3':0.0,
	'min_var':0.2,
}



def time_from_filename(filename):
	"""Return time in seconds, from 8/15 0h0m0s"""
	mon, day, hr, min, sec = filename.split('/')[-1].split('_')[2:]
	sec = sec.split('.')[0] # remove .wav
	time = 0.0
	for unit, in_secs in zip([day, hr, min, sec], [1., 1./24, 1./1440, 1./86400]):
		time += float(unit) * in_secs
	return time


def process_sylls(load_dir, save_dir, meta={}, min_freq=300, max_freq=12e3,
				num_freq_bins=64, min_dur=6e-3, max_dur=0.2,
				num_time_bins=128, verbose=True, fs=44100.0,
				log_spacing=True, max_num_files=100, seg_params=seg_params):
	"""
	Process files in <load_dir> and save to <save_dir>.

	Notes
	-----
		-   Assumes the .wav files are contiguous recordings with alphabetically
			ordered filenames.


	Arguments
	---------
	- load_dir :

	- save_dir :

	...

	"""
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	if 'load_dir' not in meta:
		meta['load_dir'] = load_dir
	filenames = [load_dir + i for i in os.listdir(load_dir) if i[-4:] == '.wav']
	np.random.shuffle(filenames)
	filenames = filenames[:max_num_files]
	write_file_num = 0
	onsets, offsets, syll_specs, syll_lens, syll_times = [], [], [], [], []
	if verbose:
		print("Processing .wav files in", load_dir)
	start_times = [] # TEMP
	for load_filename in tqdm(filenames):
		start_time = time_from_filename(load_filename)
		# Get a spectrogram.
		spec, f, dt, i1, i2 = get_spec(load_filename, fs, min_freq, max_freq, num_freq_bins)
		# Collect syllable onsets and offsets.
		t_onsets, t_offsets = get_onsets_offsets(spec, seg_params)
		spectrogram_params = {
				'num_freq_bins':num_freq_bins,
				'num_time_bins':num_time_bins,
				# 'fs':fs,
				'f':f,
				'dt':dt,
				# 'i1':i1,
				# 'i2':i2,
		}
		# Retrieve spectrograms for each detected syllable.
		t_syll_specs, t_syll_lens , t_syll_times = get_syll_specs(t_onsets, \
				t_offsets, spec, start_time, **spectrogram_params)
		for arr, t_arr in zip( \
				[onsets, offsets, syll_specs, syll_lens, syll_times],\
				[t_onsets, t_offsets, t_syll_specs, t_syll_lens, t_syll_times]):
			arr += t_arr
		# Write a file when we have enough syllables.
		while len(onsets) >= sylls_per_file:
			save_filename = save_dir + "syllables_"
			save_filename += str(write_file_num).zfill(3) + '.hdf5'
			with h5py.File(save_filename, "w") as f:
				temp = np.zeros((sylls_per_file, num_freq_bins, num_time_bins), dtype='float')
				for i in range(sylls_per_file):
					temp[i,:,:syll_lens[i]] = syll_specs[i]
				f.create_dataset('syll_specs', data=temp)
				f.create_dataset('syll_lens', data=np.array(syll_lens[:sylls_per_file]))
				f.create_dataset('syll_times', data=np.array(syll_times[:sylls_per_file]))
				for key in meta:
					f.attrs[key] = meta[key]
			onsets = onsets[sylls_per_file:]
			offsets = offsets[sylls_per_file:]
			syll_specs = syll_specs[sylls_per_file:]
			syll_lens = syll_lens[sylls_per_file:]
			syll_times = syll_times[sylls_per_file:]
			write_file_num += 1



def get_syll_specs(onsets, offsets, spec, start_time, log_spacing=True, num_freq_bins=128,
				num_time_bins=128, f=None, dt=None):
	"""
	Return a list of spectrograms, one for each syllable.

	Arguments
	---------
		- onsets: ...
	"""
	assert(f is not None)
	assert(dt is not None)
	syll_specs, syll_lens, syll_times = [], [], []
	# For each syllable...
	for t1, t2 in zip(onsets, offsets):
		# Take a slice of the spectrogram.
		temp_spec = spec[:,t1:t2+1]
		# Within-syllable normalization.
		temp_spec -= np.percentile(temp_spec, 10.0)
		temp_spec[temp_spec<0.0] = 0.0
		temp_spec /= np.max(temp_spec)
		# Collect spectrogram, duration, & onset time.
		syll_specs.append(temp_spec)
		syll_lens.append(temp_spec.shape[1])
		syll_times.append(start_time + (t1 * dt)/(24*60*60)) # in days
	return syll_specs, syll_lens, syll_times



def get_spec(filename, fs, min_freq, max_freq, num_freq_bins):
	"""Get a spectrogram given a filename."""
	# Make sure the samplerate is correct and the audio is mono.
	temp_fs, audio = wavfile.read(filename)
	assert(temp_fs == fs)
	if len(audio.shape) > 1:
		audio = audio[0,:]

	# Convert to a magnitude-only spectrogram.
	f, t, Zxx = stft(audio, fs=fs, nperseg=nperseg, noverlap=noverlap)
	i1 = np.searchsorted(f, min_freq)
	i2 = np.searchsorted(f, max_freq)
	spec = np.log(np.abs(Zxx[i1:i2,:]) + EPS)

	# Denoise.
	spec_thresh = np.percentile(spec, spec_percentile, axis=1)
	spec_thresh = gaussian_filter1d(spec_thresh, 2)
	spec -= np.tile(spec_thresh, (spec.shape[1],1)).T
	spec[spec<0.0] = 0.0
	f = f[i1:i2]

	# Switch to mel frequency spacing.
	mel_f = np.linspace(mel(f[0]), mel(f[-1]), num_freq_bins, endpoint=True)
	mel_f = inv_mel(mel_f)
	mel_f[0] = f[0] # Correct for numerical errors.
	mel_f[-1] = f[-1]

	mel_f_spec = np.zeros((num_freq_bins, spec.shape[1]), dtype='float')
	for j in range(spec.shape[1]):
		interp = interp1d(f, spec[:,j], kind='cubic')
		mel_f_spec[:,j] = interp(mel_f)
	spec = mel_f_spec

	return spec, mel_f, t[1]-t[0], i1, i2


def tune_segmenting_params(load_dirs, fs=fs, min_freq=min_freq,
		max_freq=max_freq, num_freq_bins=num_freq_bins, **kwargs):
	"""Tune params by visualizing segmenting decisions."""
	filenames = []
	for load_dir in load_dirs:
		filenames += [load_dir + i for i in os.listdir(load_dir) if i[-4:] == '.wav']
	np.random.shuffle(filenames)
	# Keep tuning params...
	while True:
		for key in seg_params:
			temp = ''
			while not is_number(temp):
				temp = input('Set value for '+key+': ')
			seg_params[key] = float(temp)

		# Visualize segmenting decisions.
		temp = ''
		file_num = 0
		i = 0
		spec, _, dt, _, _ = get_spec(filenames[file_num], fs, min_freq, max_freq, num_freq_bins)
		dur = int(round(2.0/dt))
		onsets, offsets, tr_1, tr_2, tr_3 = get_onsets_offsets(spec, \
				seg_params=seg_params, return_traces=True)
		while temp != 'q' and temp != 'r':
			if (i+1)*dur < spec.shape[1]:
				_, axarr = plt.subplots(2,1, sharex=True)
				axarr[0].imshow(spec[:,i*dur:(i+1)*dur], origin='lower', \
						aspect='auto')
				for j in range(len(onsets)):
					if onsets[j] >= i*dur and onsets[j] < (i+1)*dur:
						for k in [0,1]:
							axarr[k].axvline(x=onsets[j]-i*dur, c='b', lw=0.5)
					if offsets[j] >= i*dur and offsets[j] < (i+1)*dur:
						for k in [0,1]:
							axarr[k].axvline(x=offsets[j]-i*dur, c='r', lw=0.5)
				axarr[1].axhline(y=seg_params['th_1'], lw=0.5, c='r')
				axarr[1].axhline(y=seg_params['th_2'], lw=0.5, c='b')
				axarr[1].axhline(y=seg_params['min_var'], lw=0.5, c='g')
				axarr[1].axhline(y=0.0, lw=0.5, c='k')
				axarr[1].plot(range(dur), tr_1[i*dur:(i+1)*dur], c='b')
				axarr[1].plot(range(dur), tr_2[i*dur:(i+1)*dur], c='r')
				axarr[1].plot(range(dur), tr_3[i*dur:(i+1)*dur], c='g', alpha=0.2)
				plt.savefig('temp.pdf')
				plt.close('all')
				i += 1
			elif file_num != len(filenames)-1:
				i = 0
				file_num += 1
				spec,_,_,_,_ = get_spec(filenames[file_num], fs, min_freq, max_freq, num_freq_bins)
				onsets, offsets, tr_1, tr_2, tr_3 = \
						get_onsets_offsets(spec, seg_params, return_traces=True)
			else:
				print('Reached the end of the files...')
				return
			temp = input('Continue? [y] or [q]uit or [r]etune params: ')
		if temp == 'q':
			return seg_params

def mel(a):
	return 1127. * np.log(1 + a / 700.)

def inv_mel(a):
	return 700. * (np.exp(a / 1127.) - 1.0)


def is_number(s):
	try:
		float(s)
		return True
	except:
		return False


if __name__ == '__main__':
	pass
