"""
Process syllables and save data.

"""
__author__ = "Jack Goffinet"
__date__ = "December 2018 - January 2019"


import os
import numpy as np
from scipy.io import wavfile, loadmat
from scipy.signal import stft

import h5py
from tqdm import tqdm
from scipy.interpolate import interp1d
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import mode
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.switch_backend('agg')

import umap
import hdbscan

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
	'num_freq_bins': 128,
	'num_time_bins': 128,
	'spacing': 'mel',
	'time_stretch': False,
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
	filenames = [load_dir + i for i in os.listdir(load_dir) if i[-4:] in ['.wav', '.mat']]
	np.random.shuffle(filenames)
	filenames = filenames[:max_num_files]
	write_file_num = 0
	onsets, offsets, syll_specs, syll_lens, syll_times = [], [], [], [], []
	print("Processing .wav files in", load_dir)
	for load_filename in tqdm(filenames):
		start_time = time_from_filename(load_filename)
		# Get a spectrogram.
		spec, f, dt, i1, i2 = get_spec(load_filename, p)
		# t_onsets, t_offsets = get_onsets_offsets_from_file(load_filename, dt) # TEMP
		# Collect syllable onsets and offsets.
		t_onsets, t_offsets = get_onsets_offsets(spec, dt, p['seg_params'])
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
		temp_spec -= np.percentile(temp_spec, 10.0)
		temp_spec[temp_spec<0.0] = 0.0
		temp_spec /= np.max(temp_spec)

		# Switch to sqrt duration.
		if p['time_stretch']:
			new_dur = int(round(temp_spec.shape[1]**0.5 * p['num_time_bins']**0.5))
			temp_spec = resize(temp_spec, (temp_spec.shape[0], new_dur), anti_aliasing=True, mode='reflect')

		# Collect spectrogram, duration, & onset time.
		syll_specs.append(temp_spec)
		syll_lens.append(temp_spec.shape[1])
		syll_times.append(start_time + (t1 * dt)/(24*60*60)) # in days
	return syll_specs, syll_lens, syll_times


def get_spec(filename, params, start_index=None, end_index=None):
	"""Get a spectrogram given a filename."""
	p = {**default_params, **params}
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
	spec -= p['seg_params']['spec_thresh']
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
		filenames += [load_dir + i for i in os.listdir(load_dir) if i[-4:] in ['.wav', '.mat']]
	file_num = 0
	file_len = get_wav_len(filenames[file_num])
	# Keep tuning params...
	while True:
		for key in seg_params:
			# Skip non-tunable parameters.
			if key in ['num_time_bins', 'num_freq_bins', 'freq_response', 'is_noise']:
				continue
			temp = 'not a valid option'
			while not is_number_or_empty(temp):
				temp = input('Set value for '+key+': ['+str(seg_params[key])+ '] ')
			if temp != '':
				seg_params[key] = float(temp)

		# Visualize segmenting decisions.
		temp = ''
		i = 0
		dur_seconds = 2.0 * seg_params['max_dur']
		dur_samples = int(dur_seconds * params['fs'])
		while temp != 'q' and temp != 'r':
			if (i+1)*dur_samples < file_len:
				# Get spec & onsets/offsets.
				spec, f, dt, _, _ = get_spec(filenames[file_num], params, \
						start_index=(i-1)*dur_samples, end_index=(i+2)*dur_samples)
				onsets, offsets, tr_1 = get_onsets_offsets(spec, \
						dt, seg_params=seg_params, return_traces=True)
				dur = int(dur_seconds / dt)
				# Plot.
				i1 = dur
				if i == 0:
					i1 = 0
				i2 = i1 + dur
				_, axarr = plt.subplots(2,1, sharex=True)
				axarr[0].imshow(spec[:,i1:i2], origin='lower', \
						aspect='auto', \
						extent=[i*dur_seconds, (i+1)*dur_seconds, f[0], f[-1]])
				for j in range(len(onsets)):
					if onsets[j] >= i1 and onsets[j] < i2:
						time = i*dur_seconds + (onsets[j] - i1) * dt
						for k in [0,1]:
							axarr[k].axvline(x=time, c='b', lw=0.5)
						temp = np.max(np.mean(np.abs(np.diff(spec[:,onsets[j]:offsets[j]], axis=0)), axis=0))
						axarr[k].text(time, 20, str(temp)[:6], fontsize=8)
					if offsets[j] >= i1 and offsets[j] < i2:
						time = i*dur_seconds + (offsets[j] + 1 - i1) * dt
						for k in [0,1]:
							axarr[k].axvline(x=time, c='r', lw=0.5)
				axarr[1].axhline(y=seg_params['th_1'], lw=0.5, c='b')
				axarr[1].axhline(y=seg_params['th_2'], lw=0.5, c='b')
				axarr[1].axhline(y=seg_params['th_3'], lw=0.5, c='b')
				axarr[1].axhline(y=0.0, lw=0.5, c='k')
				xvals = np.linspace(i*dur_seconds, (i+1)*dur_seconds, dur)
				axarr[1].plot(xvals, tr_1[i1:i2], c='b')
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
			if len([j for j in onsets if j>i1 and j<i2]) > 0:
				temp = input('Continue? [y] or [q]uit or [r]etune params: ')
			else:
				temp = 'y'
		if temp == 'q':
			return seg_params


def get_onsets_offsets_from_file(audio_filename, dt):
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


def tune_noise_detection(load_dirs, params, save_filename, load_filename=None):
	"""Main loop for semisupervised noise detection"""
	# Collect labeled data.
	p = {**default_params, **params}
	labeled_data = {
			'specs':np.array([], dtype='object'),
			'filenames':np.array([], dtype=np.str_),
			'onsets':np.array([], dtype='int'),
			'offsets':np.array([], dtype='int'),
			'labels':np.array([], dtype='int')
	}
	try:
		labeled_data = np.load(load_filename).item()
	except AttributeError:
		pass
	labeled_set = {(i,j):0 for i,j in zip(labeled_data['filenames'], labeled_data['onsets'])}
	# Segment all audio and collect unlabeled data.
	print("Segmenting audio...")
	unlabeled_data = {'specs':[], 'filenames':[], 'onsets':[], 'offsets':[], 'labels':[]}
	filenames = []
	for load_dir in load_dirs:
		filenames += [load_dir + i for i in os.listdir(load_dir) if i[-4:] in ['.wav', '.mat']]
	for filename in tqdm(filenames[:100]): # TEMP!
		spec, f, dt, i1, i2 = get_spec(filename, p)
		onsets, offsets = get_onsets_offsets(spec, dt, p['seg_params'])
		# Remove the data that's already been labeled.
		mask = np.array([i for i in range(len(onsets)) if (filename, onsets[i]) not in labeled_set], dtype='int')
		onsets, offsets = np.array(onsets)[mask].tolist(), np.array(offsets)[mask].tolist()
		# Get spectrograms.
		syll_specs, _ , _ = get_syll_specs(onsets, offsets, spec, 0.0, dt, p)
		# Append stuff to <unlabeled_data>.
		unlabeled_data['specs'] += syll_specs
		unlabeled_data['filenames'] += [filename] * len(onsets)
		unlabeled_data['onsets'] += onsets
		unlabeled_data['offsets'] += offsets
		unlabeled_data['labels'] += [-1] * len(onsets)
	for field in unlabeled_data:
		if field == 'filenames':
			unlabeled_data[field] = np.array(unlabeled_data[field])
		else:
			temp = np.empty(len(unlabeled_data[field]), dtype=labeled_data[field].dtype)
			temp[:] = unlabeled_data[field]
			unlabeled_data[field] = temp
	p['dt'] = dt
	# Repeatedly cluster and ask for user input.
	est_acc, emp_acc = [], [] # Estimated accuracy, empirical accuracy
	while True:
		labeled_data, unlabeled_data, est_acc, emp_acc = \
				run_noise_detection_iteration(labeled_data, unlabeled_data, est_acc, emp_acc, p, save_filename)
		response = 'not valid'
		while response not in ['', 'y', 'n']:
			response = input("Continue? [y]/n ")
		if response == 'n':
			return


def run_noise_detection_iteration(labeled_data, unlabeled_data, est_acc, emp_acc, params, save_filename):
	"""Iterated semisupervised learning for noise detection"""
	# Perform semisupervised dimensionality reduction.
	print("Performing dimensionality reduction...")
	N_l, N_nl = len(labeled_data['specs']), len(unlabeled_data['specs'])
	all_specs = np.zeros((N_l+N_nl, params['num_freq_bins'], params['num_time_bins']))
	for i, spec in enumerate(labeled_data['specs']):
		all_specs[i,:,:spec.shape[1]] = spec
	for i, spec in enumerate(unlabeled_data['specs']):
		all_specs[N_l+i,:,:spec.shape[1]] = spec
	all_specs = all_specs.reshape(N_l + N_nl, -1)
	true_labels = np.concatenate((labeled_data['labels'], unlabeled_data['labels']))
	fitter = umap.UMAP().fit(all_specs, y=true_labels)
	embedding = fitter.embedding_
	# Then cluster.
	print("Clustering...")
	clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=15, prediction_data=True)
	clusterer.fit(embedding.astype(np.float64))
	labels = np.copy(clusterer.labels_)
	# Compute stuff for optimal experimental design.
	temp = hdbscan.all_points_membership_vectors(clusterer)
	soft_labels = np.zeros((temp.shape[0], temp.shape[1]+1))
	soft_labels[:,:-1] = temp
	soft_labels[:,-1] = 1.0 - np.sum(soft_labels, axis=1)
	n_c = soft_labels.shape[1]
	cluster_weights = np.sum(soft_labels, axis=0) # Expected number of points.
	cluster_counts = np.zeros(n_c)
	cluster_positive_counts = np.zeros(n_c)
	for i, label in enumerate(labeled_data['labels']):
		if label == 1:
			cluster_positive_counts[:] += soft_labels[i]
		cluster_counts[:] += soft_labels[i]
	# Jeffreys prior
	cluster_prob = (cluster_positive_counts + 0.5) / (cluster_counts + 1.0)
	cluster_prob_pos = (cluster_positive_counts + 1.5) / (cluster_counts + 2.0)
	cluster_prob_neg = (cluster_positive_counts + 0.5) / (cluster_counts + 2.0)
	cluster_g = 2.0*cluster_prob**2 - 2.0*cluster_prob + 1.0
	cluster_g_pos = 2.0*cluster_prob_pos**2 - 2.0*cluster_prob_pos + 1.0
	cluster_g_neg = 2.0*cluster_prob_neg**2 - 2.0*cluster_prob_neg + 1.0
	cluster_delta_g = cluster_prob * (cluster_g_pos - cluster_g) + \
					(1.0 - cluster_prob) * (cluster_g_neg - cluster_g)
	cluster_scores = cluster_weights * cluster_delta_g
	temp = 1e2 * np.dot(cluster_weights, cluster_g) / (N_l + N_nl)
	est_acc.append(temp)
	print("Estimated classification accuracy: %.2f%%" % (temp))
	# Figure out cluster to input label mapping.
	cluster_map = {-1:-1}
	for label in range(labels.max()+1):
		true_labels = labeled_data['labels'][labels[:N_l] == label]
		# If we don't have many labels or there's >10% disagreement...
		if len(true_labels) == 0 or np.where(true_labels != mode(true_labels)[0][0])[0].sum() / len(true_labels) > 0.1:
			labels[labels == label] = -1
		else:
			cluster_map[label] = mode(true_labels)[0][0]
			labels[labels == label] = cluster_map[label]
	# Plot the embedding.
	_, axarr = plt.subplots(3,1, sharex=True)
	axarr[0].set_title("HDBSCAN Clustering")
	axarr[0].scatter(embedding[:,0], embedding[:,1], c=clusterer.labels_, cmap='Spectral', alpha=0.3, s=0.5)
	axarr[2].set_title("Inferred Labels")
	predictions = np.einsum('ij,j->i', soft_labels, cluster_prob)
	plot1 = axarr[2].scatter(embedding[:,0], embedding[:,1], c=predictions, cmap='viridis', vmin=0, vmax=1, alpha=0.3, s=0.5)
	cbar = plt.colorbar(plot1,ax=axarr[2])
	cbar.solids.set_rasterized(True)
	axarr[1].set_title("True Labels")
	axarr[1].scatter(embedding[N_l:,0], embedding[N_l:,1], c='k', alpha=0.04, s=0.5)
	axarr[1].scatter(embedding[:N_l,0], embedding[:N_l,1], c=labeled_data['labels'], cmap='viridis', alpha=0.8, s=0.8)
	plt.savefig('temp.pdf')
	plt.close('all')
	# Plot accuracies.
	plt.title("Accuracies")
	plt.plot(est_acc, label='Estimated')
	plt.plot(np.linspace(0,len(est_acc)-1,len(emp_acc)), emp_acc, label='Empirical')
	plt.ylim(50,100)
	plt.legend(loc='best')
	plt.savefig('acc.pdf')
	plt.close('all')
	print("Found", clusterer.labels_.max()+1, "clusters.")
	num, denom = (clusterer.labels_ != -1).sum(), len(labels)
	print("Clustered", '%.2f%%' % (1e2*num/denom), "of syllables. (", num, "/", denom, ")")
	print("Clustered", '%.2f%%' % (1e2*(clusterer.probabilities_ > 0.9).sum()/denom), "with >90% confidence")
	num, denom = (labels != -1).sum(), len(labels)
	print("Labeled", '%.2f%%' % (1e2*num/denom), "of syllables. (",num,"/",denom,")")
	# Get user input.
	avail_indices, taken_indices = list(range(N_nl)), []
	n = min(5, len(avail_indices))
	temp = input("Sort unlabeled data ["+str(n)+"]: ")
	try:
		n = min(int(temp), len(avail_indices))
	except ValueError:
		pass
	for i in range(n):
		# Score the possible spectrograms to label.
		spec_scores = np.einsum('ij,j->i', soft_labels[N_l:], cluster_scores)
		big_number = np.max(spec_scores) - np.min(spec_scores) + 1.0
		for j in taken_indices:
			spec_scores[j] -= big_number
		# Select the best.
		index = np.argmax(spec_scores)
		avail_indices.remove(index)
		taken_indices.append(index)
		# Ask for a label.
		label = tune_noise_helper(index, unlabeled_data, params)
		temp = float(np.dot(soft_labels[N_l+index], cluster_prob) > 0.5 == bool(label))
		temp *= 100.0
		if len(emp_acc) == 0:
			emp_acc.append(temp)
		else:
			emp_acc.append(emp_acc[-1] + (temp - emp_acc[-1])/(len(emp_acc) + 1))

		# Update statistics.
		cluster_counts += soft_labels[index]
		if label == 1:
			cluster_positive_counts += soft_labels[index]
		# Jeffreys prior
		cluster_prob = (cluster_positive_counts + 0.5) / (cluster_counts + 1.0)
		cluster_prob_pos = (cluster_positive_counts + 1.5) / (cluster_counts + 2.0)
		cluster_prob_neg = (cluster_positive_counts + 0.5) / (cluster_counts + 2.0)
		cluster_g = 2.0*cluster_prob**2 - 2.0*cluster_prob + 1.0
		cluster_g_pos = 2.0*cluster_prob_pos**2 - 2.0*cluster_prob_pos + 1.0
		cluster_g_neg = 2.0*cluster_prob_neg**2 - 2.0*cluster_prob_neg + 1.0
		cluster_delta_g = cluster_prob * (cluster_g_pos - cluster_g) + \
						(1.0 - cluster_prob) * (cluster_g_neg - cluster_g)
		cluster_scores = cluster_weights * cluster_delta_g
		temp = 1e2 * np.dot(cluster_weights, cluster_g) / (N_l + N_nl)
		est_acc.append(temp)
		print("Estimated classification accuracy: %.2f%%" % (temp))
	# Append answers to the labeled dataset.
	avail_indices = np.array(avail_indices, dtype='int')
	taken_indices = np.array(taken_indices, dtype='int')
	for field in labeled_data:
		if field == 'filenames':
			temp = unlabeled_data[field][taken_indices]
		else:
			temp = np.empty(len(taken_indices), dtype=labeled_data[field].dtype)
			temp[:] = [unlabeled_data[field][i] for i in taken_indices]
		labeled_data[field] = np.concatenate((labeled_data[field], temp))
	# Then save it.
	print("Saving data...")
	np.save(save_filename, labeled_data)
	# And remove these from the unlabeled dataset.
	for field in unlabeled_data:
		if field == 'filename':
			temp = unlabeled_data[field][avail_indices]
		else:
			temp = np.empty(len(avail_indices), dtype=labeled_data[field].dtype)
			temp[:] = [unlabeled_data[field][i] for i in avail_indices]
		unlabeled_data[field] = temp
	return labeled_data, unlabeled_data, est_acc, emp_acc



def tune_noise_helper(index, data, p):
	"""Get true labels from user."""
	dt, fs = p['dt'], p['fs']
	dur_seconds = 1.0 # p['seg_params']['max_dur']
	dur_samples = int(round(dur_seconds * fs))
	filename, onset, offset = tuple(data[i][index] for i in ['filenames', 'onsets', 'offsets'])
	start = int(round(0.5 * dt * fs * (onset+offset))) - dur_samples // 2
	end = start + dur_samples
	start = max(0, start)
	end = min(get_wav_len(filename), end)
	spec, _, _, _, _ = get_spec(filename, p, start_index=start, end_index=end)
	plt.imshow(spec, origin='lower', aspect='auto', \
			extent=[start/fs,end/fs,0,128])
	plt.axvline(x=onset*dt, c='r', lw=0.5)
	plt.axvline(x=offset*dt, c='r', lw=0.5)
	plt.title("Set label:")
	plt.savefig('temp.pdf')
	plt.close('all')
	response = 'invalid'
	while type(response) != type(4):
		response = input("Set label: ")
		try:
			response = int(response)
			assert response in [0,1]
		except ValueError:
			print("Invalid response!")
			pass
	# Fill in the true label.
	data['labels'][index] = response
	return response


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
	if filename[-4:] == '.wav':
		_, audio = wavfile.read(filename)
	elif filename[-4:] == '.mat':
		audio = loadmat(filename)['spike2Chunk'].reshape(-1)
	return len(audio)



def time_from_filename(filename):
	"""Return time in seconds, from July 31st 0h0m0s"""
	try:
		mon, day, hr, min, sec = filename.split('/')[-1].split('_')[2:]
		sec = sec.split('.')[0] # remove .wav
		time = 0.0
		for unit, in_days in zip([mon, day, hr, min, sec], [31., 1., 1./24, 1./1440, 1./86400]):
			time += float(unit) * in_days
		return time
	except:
		return 0.0


if __name__ == '__main__':
	pass


###
