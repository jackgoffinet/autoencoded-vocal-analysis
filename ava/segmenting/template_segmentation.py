"""
Segment song bouts using linear acoustic feature templates.

"""
__date__ = "April-October 2019"


from affinewarp import ShiftWarping
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from scipy.ndimage.filters import gaussian_filter
import os
import umap
import warnings

from ava.plotting.tooltip_plot import tooltip_plot
from ava.segmenting.utils import softmax

# Silence numpy.loadtxt when reading empty files.
warnings.filterwarnings("ignore", category=UserWarning)

EPSILON = 1e-9


def get_spec(fs, audio, p):
	"""
	Get a spetrogrma. Not many options here.

	Parameters
	----------
	fs : float
		Samplerate.
	audio : numpy.ndarray
		Raw audio.
	p : dict
		Parameters. Must contain keys: ``'nperseg'``, ``'noverlap'``,
		``'min_freq'``, ``'max_freq'``, ``'spec_min_val'``, and
		``'spec_max_val'``.

	Returns
	-------
	spec : numpy.ndarray
		Spectrogram.
	dt : float
		Timestep.
	"""
	f, t, spec = stft(audio, fs=fs, nperseg=p['nperseg'], \
			noverlap=p['noverlap'])
	i1 = np.searchsorted(f, p['min_freq'])
	i2 = np.searchsorted(f, p['max_freq'])
	spec = spec[i1:i2]
	spec = np.log(np.abs(spec) + EPSILON)
	spec -= p['spec_min_val']
	spec /= p['spec_max_val'] - p['spec_min_val']
	spec[spec<0.0] = 0.0
	spec[spec>1.0] = 1.0
	return spec, t[1]-t[0]


def get_template(feature_dir, p):
	"""
	Create a linear template given exemplar spectrograms.

	Parameters
	----------
	feature_dir : str
		Directory containing multiple audio files to average together.
	p : dict
		Parameters. Must contain keys: ...

	Returns
	-------
	template : np.ndarray
		Spectrogram template.
	dt : float
		Timestep.
	"""
	filenames = [os.path.join(feature_dir, i) for i in os.listdir(feature_dir) \
		if _is_audio_file(i)]
	specs = []
	for i, filename in enumerate(filenames):
		fs, audio = wavfile.read(filename)
		assert fs == p['fs'], "Found samplerate="+str(fs)+\
			", expected "+str(p['fs'])
		spec, dt = get_spec(fs, audio, p)
		spec = gaussian_filter(spec, (0.5,0.5))
		specs.append(spec)
	min_time_bins = min(spec.shape[1] for spec in specs)
	specs = np.array([i[:,:min_time_bins] for i in specs])
	template = np.mean(specs, axis=0) # Average over all the templates.
	template -= np.mean(template)
	template /= np.sum(np.power(template, 2)) + EPSILON
	return template, dt


def segment_files(audio_dirs, segment_dirs, template, p, num_mad=2.0):
	"""
	Write segments.

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	segment_dirs : list of str
		Corresponding directories containing segmenting decisions.
	template : numpy.ndarray
		Spectrogram template.
	p : dict
		Segmenting parameters.
	num_mad : float, optional
		Number of median absolute deviations for cross-correlation threshold.
		Defaults to ``2.0``.

	Returns
	-------
	result : dict
		Maps audio filenames to segments (numpy.ndarrays).
	"""
	result = {}
	for audio_dir, segment_dir in zip(audio_dirs, segment_dirs):
		if not os.path.exists(segment_dir):
			os.makedirs(segment_dir)
		audio_fns = [os.path.join(audio_dir, i) for i in os.listdir(audio_dir) \
			if _is_audio_file(i)]
		for audio_fn in audio_fns:
			segments = segment_file(audio_fn, template, p, num_mad=num_mad)
			result[audio_fn] = segments
			segment_fn = os.path.split(audio_fn)[-1][:-4] + '.txt'
			segment_fn = os.path.join(segment_dir, segment_fn)
			np.savetxt(segment_fn, segments, fmt='%.5f')
	return result


def read_segment_decisions(audio_dirs, segment_dirs):
	"""Returns the same data as ``segment_files``."""
	result = {}
	for audio_dir, segment_dir in zip(audio_dirs, segment_dirs):
		audio_fns = [os.path.join(audio_dir, i) for i in os.listdir(audio_dir) \
			if _is_audio_file(i)]
		for audio_fn in audio_fns:
			segment_fn = os.path.split(audio_fn)[-1][:-4] + '.txt'
			segment_fn = os.path.join(segment_dir, segment_fn)
			segments = np.loadtxt(segment_fn)
			result[audio_fn] = segments
	return result


def segment_file(filename, template, p, num_mad=2.0, min_dt=0.05):
	"""
	Match linear audio features, extract times where features align.

	Parameters
	----------
	filename : str
		Audio filename.
	template : numpy.ndarray
		Spectrogram template.
	p : dict
		Segmenting parameters.

	Returns
	-------

	Notes
	-----

	"""
	fs, audio = wavfile.read(filename)
	assert fs == p['fs'], "Found samplerate="+str(fs)+", expected "+str(p['fs'])
	big_spec, dt = get_spec(fs, audio, p)
	spec_len = template.shape[1]
	template = template.flatten()
	# Compute normalized cross-correlation.
	result = np.zeros(big_spec.shape[1] - spec_len)
	for i in range(len(result)):
		temp = big_spec[:,i:i+spec_len].flatten()
		temp -= np.mean(temp)
		temp /= np.sum(np.power(temp, 2)) + EPSILON
		result[i] = np.dot(template, temp)
	median = np.median(result)
	abs_devs = np.abs(result - median)
	mad = np.median(abs_devs) + EPSILON
	# Get maxima.
	times = dt * np.arange(len(result))
	indices = np.argwhere(result>median + num_mad*mad).flatten()[1:-1]
	max_indices = []
	for i in range(2,len(indices)-1):
		if max(result[indices[i]-1], result[indices[i]+1]) < result[indices[i]]:
			max_indices.append(indices[i])
	max_indices = np.array(max_indices, dtype='int')
	max_indices = _clean_max_indices(max_indices, times, result, min_dt=min_dt)
	# Define onsets/offsets.
	segments = np.zeros((len(max_indices), 2))
	segments[:,0] = dt * max_indices # onsets
	segments[:,1] = segments[:,0] + spec_len * dt
	return segments


def clean_collected_data(result, audio_dirs, segment_dirs, template_length, p, \
	n=10**4):
	"""
	Take a look at the collected data and discard false positives.

	Parameters
	----------
	result : ...
	...
	audio_dirs : ...
	...
	segment_dirs : ...
	...

	Notes
	-----

	"""
	# Collect spectrograms.
	specs = []
	if template_length is not None:
		delta_i = int(round(template_length * p['fs']))
	for filename in result.keys():
		fs, audio = wavfile.read(filename)
		assert fs == p['fs']
		for segment in result[filename]:
			i1 = int(round(segment[0] * fs))
			if template_length is None:
				i2 = int(round(segment[1] * fs))
			else:
				i2 = i1 + delta_i
			spec, dt = get_spec(fs, audio[i1:i2], p)
			specs.append(spec)
	if template_length is None:
		max_t = max(spec.shape[1] for spec in specs)
		temp_specs = np.zeros((len(specs), specs[0].shape[0], max_t))
		for i, spec in enumerate(specs):
			temp_specs[i,:,:spec.shape[1]] = spec
		specs = temp_specs
	else:
		specs = np.array(specs)
	np.random.seed(42)
	specs = specs[np.random.permutation(len(specs))]
	np.random.seed(None)
	# UMAP the spectrograms.
	transform = umap.UMAP(random_state=42)
	embedding = transform.fit_transform(specs.reshape(len(specs), -1))
	# Plot and ask for user input.
	bounds = {
		'x1s':[],
		'x2s':[],
		'y1s':[],
		'y2s':[],
	}
	X, Y = embedding[:,0], embedding[:,1]
	i = 0
	while True:
		colors = ['b' if _in_region(embed, bounds) else 'r' for embed in embedding]
		print("Selected ", len([c for c in colors if c=='b']), "out of", len(colors))
		plt.scatter(X, Y, c=colors, s=0.9, alpha=0.5)
		for x_tick in np.arange(np.floor(np.min(X)), np.ceil(np.max(X))):
			plt.axvline(x=x_tick, c='k', alpha=0.1, lw=0.5)
		for y_tick in np.arange(np.floor(np.min(Y)), np.ceil(np.max(Y))):
			plt.axhline(y=y_tick, c='k', alpha=0.1, lw=0.5)
		title = "Select relevant song:"
		plt.title(title)
		plt.savefig('temp.pdf')
		plt.close('all')
		if i == 0:
			tooltip_plot(embedding, specs, num_imgs=10**3, title=title)
		bounds['x1s'].append(float(input('x1: ')))
		bounds['x2s'].append(float(input('x2: ')))
		bounds['y1s'].append(float(input('y1: ')))
		bounds['y2s'].append(float(input('y2: ')))
		temp = input('(<c> to continue) ')
		if temp == 'c':
			break
		i += 1
	# Save only the good segments.
	num_deleted, num_total = 0, 0
	for audio_dir, seg_dir in zip(audio_dirs, segment_dirs):
		audio_fns = [os.path.join(audio_dir, i) for i in os.listdir(audio_dir) \
			if _is_audio_file(i)]
		for audio_fn in audio_fns:
			fs, audio = wavfile.read(audio_fn)
			assert fs == p['fs']
			segment_fn = os.path.split(audio_fn)[-1][:-4] + '.txt'
			segment_fn = os.path.join(seg_dir, segment_fn)
			segments = np.loadtxt(segment_fn).reshape(-1,2)
			if len(segments) == 0:
				continue
			new_segments = np.zeros(segments.shape)
			i = 0
			for segment in segments:
				i1 = int(round(segment[0] * fs))
				if template_length is None:
					i2 = int(round(segment[1] * fs))
				else:
					i2 = i1 + delta_i
				spec, dt = get_spec(fs, audio[i1:i2], p)
				if template_length is None:
					temp_spec = np.zeros((spec.shape[0], max_t))
					temp_spec[:, :spec.shape[1]] = spec
					spec = temp_spec
				embed = transform.transform(spec.reshape(1,-1)).reshape(2)
				if _in_region(embed, bounds):
					new_segments[i] = segment[:]
					i += 1
					num_total += 1
				else:
					num_deleted += 1
			new_segments = new_segments[:i]
			np.savetxt(segment_fn, new_segments, fmt='%.5f')
	print("deleted", num_deleted, "remaining", num_total)


def segment_sylls_from_songs(audio_dirs, song_seg_dirs, syll_seg_dirs, p, \
	shoulder=0.05, img_fn='temp.pdf'):
	"""
	Split song renditions into syllables.

	Parameters
	----------
	audio_dirs : list of str
		Audio directories.
	song_seg_dirs : list of str
		Directories containing song segments.
	syll_seg_dirs : list of str
		Directories where syllable segments are written.
	p : dict
		Segmenting parameters.
	shoulder : float, optional
		Duration of padding on either side of song segments.
	img_fn : str, optional
		Image filename. Defaults to ``'temp.pdf'``.
	"""
	# Read segments.
	song_segs = read_segment_decisions(audio_dirs, song_seg_dirs)
	# Collect spectrograms.
	empty_audio_files = []
	specs, fns, song_onsets = [], [], []
	for audio_fn in song_segs:
		fs, audio = wavfile.read(audio_fn)
		for seg in song_segs[audio_fn].reshape(-1,2):
			onset, offset = seg[0] - shoulder, seg[1] + shoulder
			i1, i2 = int(fs*onset), int(fs*offset)
			assert i1 >= 0, "Negative index! Decrease `shoulder`."
			assert i2 <= len(audio), "Index > len(audio)! Decrease `shoulder`."
			spec, dt = get_spec(fs, audio[i1:i2], p)
			specs.append(spec)
			fns.append(audio_fn)
			song_onsets.append(onset)
		if len(song_segs[audio_fn]) == 0:
			empty_audio_files.append(audio_fn)
	assert len(specs) > 0, "Found no spectrograms!"
	# Calculate and smooth amplitude traces.
	amp_traces = []
	for spec in specs:
		if p['softmax']:
			amps = softmax(spec, t=p['temperature'])
		else:
			amps = np.sum(spec, axis=0)
		# amps = gaussian_filter(amps, p['smoothing_timescale']/dt)
		amps -= np.mean(amps)
		amps /= np.std(amps) + EPSILON
		amp_traces.append(amps)
	amp_traces = np.array(amp_traces)
	max_t = amp_traces.shape[1]*dt*1e3
	num_time_bins = amp_traces.shape[1]
	model = ShiftWarping(maxlag=.3, smoothness_reg_scale=10.)
	model.fit(amp_traces[:,:,np.newaxis], iterations=100)
	aligned = model.predict().squeeze()
	max_raw_val = np.max(amp_traces)
	max_aligned_val = np.max(aligned)
	shifts = model.shifts
	quantiles = []
	break_flag = False
	while True:
		# Plot.
		_, axarr = plt.subplots(3,1, sharex=True)
		axarr[0].imshow(specs[0], origin='lower', aspect='auto', \
				extent=[0,max_t,p['min_freq']/1e3,p['max_freq']/1e3])
		temp = np.copy(amp_traces)
		for q in quantiles:
			for i in range(len(temp)):
				temp[i,int(round(q*num_time_bins))+shifts[i]] = max_raw_val
		axarr[1].imshow(temp, origin='lower', aspect='auto', \
				extent=[0,max_t,0,len(amp_traces)])
		temp = np.copy(aligned)
		for q in quantiles:
			for i in range(len(temp)):
				temp[i,int(round(q*num_time_bins))] = max_aligned_val
		axarr[2].imshow(temp, origin='lower', aspect='auto', \
				extent=[0,max_t,0,len(amp_traces)])
		axarr[0].set_ylabel("Frequency (kHz)")
		axarr[1].set_ylabel('Amplitude')
		axarr[2].set_ylabel('Shifted')
		axarr[0].set_title('Enter segmenting quantiles:')
		axarr[2].set_xlabel('Time (ms)')
		plt.savefig(img_fn)
		plt.close('all')
		# Ask for segmenting decisions.
		while True:
			temp = input("Add or delete quantile or [s]top: ")
			if temp == 's':
				break_flag = True
				break
			try:
				temp = float(temp)
				assert 0.0 < temp and temp < 1.0
				if temp in quantiles:
					quantiles.remove(temp)
				else:
					quantiles.append(temp)
				break
			except:
				print("Invalid input!")
				print("Must be \'s\' or a float between 0 and 1.")
				continue
		if break_flag:
			break
	# Write syllable segments.
	duration = num_time_bins * dt
	quantiles = np.array(quantiles)
	quantiles.sort()
	files_encountered = {}
	for i, (fn, song_onset) in enumerate(zip(fns, song_onsets)):
		# Unshifted onsets and offsets.
		onsets = song_onset + duration * quantiles[:-1]
		offsets = song_onset + duration * quantiles[1:]
		# Apply shifts.
		onsets += shifts[i] * dt
		offsets += shifts[i] * dt
		# Save.
		index = audio_dirs.index(os.path.split(fn)[0])
		write_fn = os.path.join(syll_seg_dirs[index], os.path.split(fn)[-1])
		write_fn = write_fn[:-4] + '.txt'
		if not os.path.exists(os.path.split(write_fn)[0]):
			os.makedirs(os.path.split(write_fn)[0])
		segs = np.stack([onsets, offsets]).reshape(2,-1).T
		header, mode = "", 'ab'
		if fn not in files_encountered:
			files_encountered[fn] = 1
			mode = 'wb'
			header += "Syllables from song: " + fn + "\n"
		header += "Song onset: "+str(song_onset)
		with open(write_fn, mode) as f:
			np.savetxt(f, segs, fmt='%.5f', header=header)
	# Write empty files corresponding to audio files without song.
	for fn in empty_audio_files:
		index = audio_dirs.index(os.path.split(fn)[0])
		write_fn = os.path.join(syll_seg_dirs[index], os.path.split(fn)[-1])
		write_fn = write_fn[:-4] + '.txt'
		if not os.path.exists(os.path.split(write_fn)[0]):
			os.makedirs(os.path.split(write_fn)[0])
		header = "Syllables from song: " + fn
		np.savetxt(write_fn, np.array([]), header=header)


def _clean_max_indices(old_indices, old_times, values, min_dt=0.05):
	"""Remove maxima that are too close together."""
	if len(old_indices) <= 1:
		return old_indices
	old_indices = old_indices[np.argsort(values[old_indices])]
	indices = [old_indices[0]]
	times = [old_times[old_indices[0]]]
	i = 1
	while i < len(old_indices):
		time = old_times[old_indices[i]]
		flag = True
		for j in range(len(indices)):
			if abs(old_times[indices[j]] - time) < min_dt:
				flag = False
				break
		if flag:
			indices.append(old_indices[i])
			times.append(old_times[old_indices[i]])
		i += 1
	indices = np.array(indices)
	indices.sort()
	return indices


def _in_region(point, bounds):
	"""Is the point in the union of the given rectangles?"""
	for i in range(len(bounds['x1s'])):
		if point[0] > bounds['x1s'][i] and point[0] < bounds['x2s'][i] and \
				point[1] > bounds['y1s'][i] and point[1] < bounds['y2s'][i]:
			return True
	return False


def _is_audio_file(filename):
	return len(filename) > 4 and filename[-4:] in ['.wav']



if __name__ == '__main__':
	pass


###
