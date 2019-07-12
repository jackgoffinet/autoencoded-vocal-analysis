"""
Process syllables and save data.


TO DO:
	- save segmentation parameters as attributes in each hdf5 file
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



def process_sylls(load_dir, save_dir, p, noise_detector=None):
	"""
	Main method: process files in <load_dir> and save to <save_dir>.

	Parameters
	----------
	load_dir : string
		Directory to load files from
	save_fir : string
		Directory to save files to
	p : dictionary
		Parameters
	noise_detector : noise_detection.NoiseDetector or None
		Throws away bad syllables. default: None

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
	filenames = [load_dir + i for i in os.listdir(load_dir) if i[-4:] in ['.wav', '.mat']]
	np.random.shuffle(filenames)
	# if p['seg_params']['algorithm'] == get_onsets_offsets_from_file:
		# filenames = [i for i in filenames if os.path.exists('.'.join(i.split('.')[:-1]) + '.txt')]
	write_file_num = 0
	syll_data = {
		'specs':[],
		'times':[],
		'file_times':[],
		'durations':[],
		'filenames':[],
	}
	print("Processing audio files in", load_dir)
	for load_filename in filenames:
		start_time = time_from_filename(load_filename)
		# Get a spectrogram.
		spec, _, dt = get_spec(load_filename, p)
		# if 'f' not in p['seg_params']:
		# 	p['seg_params']['f'] = f
		# Get onsets and offsets.
		if p['seg_params']['algorithm'] == get_onsets_offsets_from_file:
			t_onsets, t_offsets = get_onsets_offsets_from_file(load_filename, dt)
		else:
			t_onsets, t_offsets = p['seg_params']['algorithm'](spec, dt, p['seg_params'])
		t_durations = [(b-a+1)*dt for a,b in zip(t_onsets, t_offsets)]
		# Retrieve spectrograms and start times for each detected syllable.
		t_specs, t_times = get_syll_specs(t_onsets, t_offsets, spec, start_time, dt, p)
		# Find noise and expunge it.
		if noise_detector is not None:
			mask = noise_detector.batch_classify(t_specs, threshold=0.5)
			for i in range(len(mask))[::-1]:
				if not mask[i]:
					del t_durations[i]
					del t_specs[i]
					del t_times[i]
		# Add the remaining syllables to <syll_data>.
		syll_data['durations'] += t_durations
		syll_data['specs'] += t_specs
		syll_data['times'] += t_times
		syll_data['file_times'] += [i - start_time for i in t_times]
		syll_data['filenames'] += len(t_durations)*[load_filename.split('/')[-1]]
		# Write a file when we have enough syllables.
		while len(syll_data['durations']) >= sylls_per_file:
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
				for k in ['durations', 'times', 'file_times']:
					f.create_dataset(k, data=np.array(syll_data[k][:sylls_per_file]))
				temp = [os.path.join(save_dir, i) for i in syll_data['filenames'][:sylls_per_file]]
				f.create_dataset('filenames', data=np.array(temp).astype('S'))
			# Remove the written data from temporary storage.
			for k in syll_data:
				syll_data[k] = syll_data[k][sylls_per_file:]
			write_file_num += 1
			if p['max_num_syllables'] is not None and write_file_num*sylls_per_file >= p['max_num_syllables']:
				return


def get_syll_specs(onsets, offsets, spec, start_time, dt, p):
	"""
	Return a list of spectrograms, one for each syllable.
	"""
	syll_specs, syll_times = [], []
	# For each syllable...
	for t1, t2 in zip(onsets, offsets):
		# Take a slice of the spectrogram.
		temp_spec = spec[:,t1:t2+1]
		# Within-syllable normalization.
		temp_spec -= np.percentile(temp_spec, 10.0)
		temp_spec[temp_spec<0.0] = 0.0
		temp_spec /= np.max(temp_spec)
		# Switch to square root duration.
		if p['time_stretch']:
			new_dur = int(round(temp_spec.shape[1]**0.5 * p['num_time_bins']**0.5))
			temp_spec = resize(temp_spec, (temp_spec.shape[0], new_dur), anti_aliasing=True, mode='reflect')
		# Collect spectrogram, duration, & onset time.
		syll_specs.append(temp_spec)
		syll_times.append(start_time + t1*dt) # in seconds
	return syll_specs, syll_times


def get_audio(filename, p, start_index=None, stop_index=None):
	"""Get a waveform given a filename."""
	# Make sure the samplerate is correct and the audio is mono.
	if filename[-4:] == '.wav':
		temp_fs, audio = wavfile.read(filename)
	elif filename[-4:] == '.mat':
		d = loadmat(filename)
		audio = d['spike2Chunk'].reshape(-1)
		temp_fs = d['fs'][0,0]
	assert temp_fs == p['fs'], "found fs: "+str(temp_fs)+", expected: "+str(p['fs'])
	if len(audio.shape) > 1:
		audio = audio[0,:]
	if start_index is not None and stop_index is not None:
		start_index = max(start_index, 0)
		audio = audio[start_index:stop_index]
	return audio


def get_spec(filename, p, start_index=None, stop_index=None):
	"""Get a spectrogram."""
	audio = get_audio(filename, p, start_index=start_index, stop_index=stop_index)
	f, t, spec = stft(audio, fs=p['fs'], nperseg=p['nperseg'], noverlap=p['noverlap'])
	spec = np.log(np.abs(spec) + EPSILON)
	spec -= p['seg_params']['spec_thresh']
	spec[spec < 0.0] = 0.0
	# Switch to mel frequency spacing.
	if p['mel']:
		new_f = np.linspace(mel(p['min_freq']), mel(p['max_freq']), p['num_freq_bins'], endpoint=True)
		new_f = inv_mel(new_f)
		new_f[0] = f[0] # Correct for numerical errors.
		new_f[-1] = f[-1]
	else:
		f_1 = p['min_freq'] - p['freq_shift']
		f_2 = p['max_freq'] - p['freq_shift']
		new_f = np.linspace(f_1, f_2, p['num_freq_bins'], endpoint=True)
	new_spec = np.zeros((p['num_freq_bins'], spec.shape[1]))
	for j in range(spec.shape[1]):
		interp = interp1d(f, spec[:,j], kind='linear', assume_sorted=True, fill_value=0.0, bounds_error=False)
		new_spec[:,j] = interp(new_f)
	spec = new_spec
	f = np.linspace(p['min_freq'], p['max_freq'], p['num_freq_bins'], endpoint=True)
	return spec, f, t[1] - t[0]


def tune_segmenting_params(load_dirs, p):
	"""Tune segementing parameters by visualizing segmenting decisions."""
	fs = p['fs']
	seg_params = p['seg_params']
	filenames = []
	for load_dir in load_dirs:
		filenames += [load_dir + i for i in os.listdir(load_dir) if i[-4:] in ['.wav', '.mat']]
	if len(filenames) == 0:
		print("Found no audio files!")
		return
	if p['seg_params']['algorithm'] == get_onsets_offsets_from_file:
		filenames = [i for i in filenames if os.path.exists('.'.join(i.split('.')[:-1]) + '.txt')]
	filenames = np.array(filenames)
	filenames = np.random.choice(filenames, min(1000, len(filenames)), replace=False)
	file_lens = [get_wav_len(filename) for filename in filenames]
	dur_seconds = 2.0 * seg_params['max_dur']
	dur_samples = int(dur_seconds * fs)
	filenames, file_lens = np.array(filenames), np.array(file_lens, dtype='int')
	filenames = filenames[file_lens > 3 * dur_samples]
	file_lens = file_lens[file_lens > 3 * dur_samples]
	assert len(filenames) >= 1
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
			if seg_params['algorithm'] == get_onsets_offsets_from_file:
				onsets, offsets = get_onsets_offsets_from_file(filename, dt, seg_params)
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


def tune_preprocessing_params(load_dirs, p):
	"""Tune segementing parameters by visualizing segmenting decisions."""
	fs = p['fs']
	seg_params = p['seg_params']
	filenames = []
	for load_dir in load_dirs:
		filenames += [load_dir + i for i in os.listdir(load_dir) if i[-4:] in ['.wav', '.mat']]
	if len(filenames) == 0:
		print("Found no audio files!")
		return
	if p['seg_params']['algorithm'] == get_onsets_offsets_from_file:
		filenames = [i for i in filenames if os.path.exists('.'.join(i.split('.')[:-1]) + '.txt')]
	filenames = np.array(filenames)
	filenames = np.random.choice(filenames, min(1000, len(filenames)), replace=False)
	file_lens = [get_wav_len(filename) for filename in filenames]
	dur_seconds = 2.0 * seg_params['max_dur']
	dur_samples = int(dur_seconds * fs)
	filenames, file_lens = np.array(filenames), np.array(file_lens, dtype='int')
	filenames = filenames[file_lens > 3 * dur_samples]
	file_lens = file_lens[file_lens > 3 * dur_samples]
	assert len(filenames) >= 1
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
			if seg_params['algorithm'] == get_onsets_offsets_from_file:
				onsets, offsets = get_onsets_offsets_from_file(filename, dt, seg_params)
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



def get_onsets_offsets_from_file(audio_filename, dt, p, txt_filename=None):
	"""Read a text file to collect onsets and offsets."""
	delimiter, skiprows, usecols = p['delimiter'], p['skiprows'], p['usecols']
	onsets = []
	offsets = []
	if txt_filename is None:
		txt_filename = '.'.join(audio_filename.split('.')[:-1]) + '.txt'
	d = np.loadtxt(txt_filename, delimiter=delimiter, skiprows=skiprows, usecols=usecols)
	for i in range(len(d)):
		onsets.append(int(np.floor(d[i,1]/dt)))
		offsets.append(int(np.ceil(d[i,2]/dt))+1)
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


def is_number(s):
	return type(s) == type(4) or type(s) == type(4.0)


def get_wav_len(filename):
	if filename[-4:] == '.wav':
		_, audio = wavfile.read(filename)
	elif filename[-4:] == '.mat':
		audio = loadmat(filename)['spike2Chunk'].reshape(-1)
	else:
		raise NotImplementedError
	return len(audio)


def time_from_filename(filename):
	"""Return time in seconds."""
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
