"""
DataContainer class for linking directories containing different sorts of data.

This is meant to make plotting and analysis easier.

TO DO:
- request random subsets.
- check for errors
- throw better errors
- Read feature files.
- Flexibly read segmenting decisions.
- make sure input directories are iterable
- add features to existing files.
"""
__author__ = "Jack Goffinet"
__date__ = "July-August 2019"


import h5py
import numpy as np
import os
from scipy.io import wavfile
from sklearn.decomposition import PCA
import torch
import umap

from ava.models.vae import VAE
from ava.models.vae_dataset import get_syllable_partition, \
	get_syllable_data_loaders, get_hdf5s_from_dir


AUDIO_FIELDS = ['audio']
SEGMENT_FIELDS = ['segments', 'segment_audio']
PROJECTION_FIELDS = ['latent_means', 'latent_mean_pca', 'latent_mean_umap']
SPEC_FIELDS = ['specs', 'onsets', 'offsets', 'audio_filenames']
MUPET_FIELDS = ['syllable_number', 'syllable_start_time', 'syllable_end_time',
	'inter-syllable_interval', 'syllable_duration', 'starting_frequency',
	'final_frequency', 'minimum_frequency', 'maximum_frequency',
	'mean_frequency', 'frequency_bandwidth', 'total_syllable_energy',
	'peak_syllable_amplitude', 'cluster']
DEEPSQUEAK_FIELDS = ['id', 'label', 'accepted', 'score', 'begin_time',
	'end_time', 'call_length', 'principal_frequency', 'low_freq', 'high_freq',
	'delta_freq', 'frequency_standard_deviation', 'slope', 'sinuosity',
	'mean_power', 'tonality']
SAP_FIELDS = ['syllable_duration_sap', 'syllable_start', 'mean_amplitude',
	'mean_pitch', 'mean_FM', 'mean_AM2', 'mean_entropy', 'mean_pitch_goodness',
	'mean_mean_freq', 'pitch_variance', 'FM_variance', 'entropy_variance',
	'pitch_goodness_variance', 'mean_freq_variance', 'AM_variance']
ALL_FIELDS = AUDIO_FIELDS + SEGMENT_FIELDS + PROJECTION_FIELDS + SPEC_FIELDS + \
	MUPET_FIELDS + DEEPSQUEAK_FIELDS + SAP_FIELDS
MUPET_ONSET_COL = MUPET_FIELDS.index('syllable_start_time')
DEEPSQUEAK_ONSET_COL = DEEPSQUEAK_FIELDS.index('begin_time')
SAP_ONSET_COL = SAP_FIELDS.index('syllable_start')
PRETTY_NAMES = {
	'audio': 'Audio',
	'segments': 'Segments',
	'segment_audio': 'Segment Audio',
	'latent_means': 'Latent Means',
	'latent_mean_pca': 'Latent Mean PCA Projection',
	'latent_mean_umap': 'Latent Mean UMAP Projection',
	'specs': 'Spectrograms',
	'onsets': 'Onsets (s)',
	'offsets': 'Offsets (s)',
	'aduio_filenames': 'Filenames',
	'syllable_number': 'Syllable Number',
	'syllable_start_time': 'Onsets (s)',
	'syllable_duration': 'Duration (ms)',
	'starting_frequency': 'Starting Frequency (kHz)',
	'final_frequency': 'Final Frequency (kHz)',
	'minimum_frequency': 'Minimum Frequency (kHz)',
	'maximum_frequency': 'Maximum Frequency (kHz)',
	'mean_frequency': 'Mean Frequency (kHz)',
	'frequency_bandwidth': 'Frequency Bandwidth (kHz)',
	'total_syllable_energy': 'Total Energy (dB)',
	'peak_syllable_amplitude': 'Peak Amplitude (dB)',
	'cluster': 'Cluster',
	'id': 'Syllabler Number',
	'label': 'Label',
	'accepted': 'Accepted',
	'score': 'DeepSqueak Detection Score',
	'begin_time': 'Onsets (s)',
	'end_time': 'Offsets (s)',
	'call_length': 'Duration (ms)',
	'principal_frequency': 'Principal Frequency (kHz)',
	'low_freq': 'Minimum Frequency (kHz)',
	'high_freq': 'Maximum Frequency (kHz)',
	'delta_freq': 'Frequency Bandwidth (kHz)',
	'frequency_standard_deviation': 'Frequency Standard Deviation (kHz)',
	'slope': 'Frequency Modulation (kHz/s)',
	'sinuosity': 'Sinuosity',
	'mean_power': 'Mean Power (dB/Hz)',
	'tonality': 'Tonality',
	'syllable_start': 'Onset (s)',
	'mean_amplitude': 'Mean Amplitude',
	'mean_pitch': 'Mean Pitch',
	'mean_FM': 'Mean Frequency Modulation',
	'mean_AM2': 'Mean Amplitude Modulation Squared',
	'mean_entropy': 'Mean Entropy',
	'mean_pitch_goodness': 'Mean Goodness of Pitch',
	'mean_mean_freq': 'Mean Frequency',
	'pitch_variance': 'Pitch Variance',
	'FM_variance': 'Frequency Modulation Variance',
	'entropy_variance': 'Entropy Variance',
	'pitch_goodness_variance': 'Goodness of Pitch Variance',
	'mean_freq_variance': 'Frequency Variance',
	'AM_variance': 'Amplitude Modulation Variance',
}
PRETTY_NAMES_NO_UNITS = {}
for k in PRETTY_NAMES:
	PRETTY_NAMES_NO_UNITS[k] = ' '.join(PRETTY_NAMES[k].split('(')[0].split(' '))
MUPET_PARAMS = {
	'kwargs': {},
	'extension': '.csv',
}
DEEPSQUEAK_PARAMS = {
	'kwargs': {},
	'extension': '.csv',
}
SAP_PARAMS = {
	'kwargs': {},
	'extension': '.csv',
}



class DataContainer():
	"""
	Link directories containing different data sources for easy plotting.

	The idea here is for plotting and analysis tools to accept a DataContainer,
	from which they can request different types of data. Those requests can then
	be handled here in a central location, which can cut down on redundant code
	and processing steps.

	Supported directory structure:

	├──	animal_1
	│	├──	audio						(raw audio)
	│	│	├── foo.wav
	│	│	├── bar.wav
	│	│	└── baz.wav
	│	├──	features 					(output of MUPET, DeepSqueak, SAP, ...)
	│	│	├── foo.csv
	│	│	├── bar.csv
	│	│	└── baz.csv
	│	├──	spectrograms 				(used to train models, written by
	│	│	├── syllables_000.hdf5		preprocessing.process_sylls)
	│	│	└──	syllables_001.hdf5
	│	└──	projections 				(latent means, UMAP, PCA, tSNE
	│		├── syllables_000.hdf5		projections, copies of features in
	│		└──	syllables_001.hdf5		experiment_1/features. These are written
	│									by a DataContainer object.)
	├──	animal_2
	│	├──	audio
	│	│	├── 1.wav
	│	│	└── 2.wav
	│	├──	features
	│	│	├── 1.csv
	│	│	└── 2.csv
	│	├──	spectrograms
	│	│	├── syllables_000.hdf5
	│	│	└── syllables_001.hdf5
	│	└──	projections
	│		├── syllables_000.hdf5
	│		└──	syllables_001.hdf5
	.
	.
	.

	There should be a 1-to-1 correspondence between, for example, the syllables
	in animal_1/audio/baz.wav and the features described in
	animal_1/features/baz.csv. Analogously, the fifth entry in
	animal_2/spectrograms/syllables_000.hdf5 should describe the same syllable
	as the fifth entry in animal_2/projections/syllables_000.hdf5. There is no
	strict relationship, however, between individual files in animal_1/audio and
	animal_1/spectrograms. The hdf5 files in the spectrograms and projections
	directories should contain a subset of the syllables in the audio and
	features directories.

	Then a DataContainer object can be initialized as:

	>>> audio_dirs = ['animal_1/audio', 'animal_2/audio']
	>>> spec_dirs = ['animal_1/spectrograms', 'animal_2/spectrograms']
	>>> model_filename = 'checkpoint.tar'
	>>> dc = DataContainer(audio_dirs=audio_dirs, spec_dirs=spec_dirs, \
	... 	model_filename=model_filename)
	>>> latent_means = dc.request('latent_means')

	It's fine to leave some of the initialization parameters unspecified. If the
	DataContainer object is asked to do something it can't, it will hopefully
	complain politely. Or at least informatively.
	"""

	def __init__(self, audio_dirs=None, segment_dirs=None, spec_dirs=None, \
		feature_dirs=None, projection_dirs=None, plots_dir='', \
		model_filename=None, template_dir=None, verbose=True):
		"""
		Parameters
		----------
		audio_dirs : list of str, or None, optional
			Directories containing audio. Defaults to None.

		segment_dirs : list of str, or None, optional
			Directories containing segmenting decisions.

		spec_dirs : list of str, or None, optional
			Directories containing hdf5 files of spectrograms. These should be
			files output by preprocessing.preprocessing (the files that are then
			read by the VAE in models.vae_dataset). Defaults to None.

		model_filename : str or None, optional
			The VAE checkpoint to load. Written by models.vae.save_state.
			Defaults to None.

		projection_dirs : list of str, or None, optional
			Directory containing different projections. This is where things
			like latent means, their projections, and handcrafted features
			found in feature_dirs are saved. Defaults to None.

		plots_dir : str, optional
			Directory to save plots. Defaults to '' (current working directory).

		feature_dirs : list of str, or None, optional
			Directory containing text files with different syllable features.
			For exmaple, this could contain exported MUPET, DeepSqueak or SAP
			syllable tables. Defaults to None.

		template_dir : list of str, or None, optional
			Directory continaing audio files of song templates. Defaults to
			None.
		"""
		self.audio_dirs = audio_dirs
		self.segment_dirs = segment_dirs
		self.spec_dirs = spec_dirs
		self.feature_dirs = feature_dirs
		self.projection_dirs = projection_dirs
		self.plots_dir = plots_dir
		self.model_filename = model_filename
		self.template_dir = template_dir
		self.verbose = verbose
		self.sylls_per_file = None # syllables in each hdf5 file in spec_dirs
		self.fields = self.check_for_fields()
		if self.plots_dir not in [None, ''] and not os.path.exists(self.plots_dir):
			os.makedirs(self.plots_dir)


	def request(self, field):
		"""
		Request some type of data.

		Besides __init__, this should be the only external-facing method.
		"""
		assert field in ALL_FIELDS, str(field) + " is not a valid field!"
		# If it's not here, make it and return it.
		if field not in self.fields:
			if self.verbose:
				print("Making field:", field)
			data = self.make_field(field)
		# Otherwise, read it and return it.
		else:
			if self.verbose:
				print("Reading field:", field)
			data = self.read_field(field)
		if self.verbose:
			print("\tDone with:", field)
		return data


	def make_field(self, field):
		"""Make a field."""
		if field == 'latent_means':
			data = self.make_latent_means()
		elif field == 'latent_mean_pca':
			data = self.make_latent_mean_pca_projection()
		elif field == 'latent_mean_umap':
			data = self.make_latent_mean_umap_projection()
		elif field in MUPET_FIELDS:
			data = self.make_feature_field(field, kind='mupet')
		elif field in DEEPSQUEAK_FIELDS:
			data = self.make_feature_field(field, kind='deepsqueak')
		elif field in SAP_FIELDS:
			data = self.make_feature_field(field, kind='sap')
		elif field == 'specs':
			raise NotImplementedError
		else:
			raise NotImplementedError
		# Add this field to the collection of fields that have been computed.
		self.fields[field] = 1
		if self.verbose:
			print("Making field:", field)
		return data


	def read_field(self, field):
		"""
		Read a field from memory.

		Paramters
		---------
		field : str
			Field name to read from file.
		"""
		if field in AUDIO_FIELDS:
			raise NotImplementedError
		elif field == 'segments':
			return self.read_segments()
		elif field == 'segment_audio':
			return self.read_segment_audio()
		elif field in PROJECTION_FIELDS:
			load_dirs = self.projection_dirs
		elif field in SPEC_FIELDS:
			load_dirs = self.spec_dirs
		elif field in MUPET_FIELDS:
			load_dirs = self.projection_dirs
		elif field in DEEPSQUEAK_FIELDS:
			load_dirs = self.projection_dirs
		elif field in SAP_FIELDS:
			load_dirs = self.projection_dirs
		else:
			raise Exception("Can\'t read field: "+field+"\n This should have \
				been caught in self.request!")
		to_return = []
		for i in range(len(self.spec_dirs)):
			spec_dir, load_dir = self.spec_dirs[i], load_dirs[i]
			hdf5s = get_hdf5s_from_dir(spec_dir)
			for j, hdf5 in enumerate(hdf5s):
				filename = os.path.join(load_dir, os.path.split(hdf5)[-1])
				with h5py.File(filename, 'r') as f:
					assert (field in f), "Can\'t find field \'"+field+"\' in"+\
						" file \'"+filename+"\'!"
					to_return.append(np.array(f[field]))
		return np.concatenate(to_return)


	def read_segment_audio(self):
		"""
		Read all the segmented audio and return it.

		result[audio_dir][audio_filename] = [audio_1, audio_2, ..., audio_n]
		"""
		self.check_for_dirs(['audio_dirs'], 'audio')
		segments = self.request('segments')
		result = {}
		for audio_dir in self.audio_dirs:
			dir_result = {}
			audio_fns = [i for i in os.listdir(audio_dir) if is_wav_file(i) \
				and i in segments[audio_dir]]
			for audio_fn in audio_fns:
				fs, audio = wavfile.read(os.path.join(audio_dir, audio_fn))
				fn_result = []
				for seg in segments[audio_dir][audio_fn]:
					i1 = int(round(seg[0]*fs))
					i2 = int(round(seg[1]*fs))
					fn_result.append(audio[i1:i2])
				dir_result[audio_fn] = fn_result
			result[audio_dir] = dir_result
		return result


	def read_segments(self):
		"""
		Return all the segmenting decisions.

		Return a dictionary mapping audio directories to audio filenames to
		numpy arrays of shape [num_segments,2] containing onset and offset
		times.

		TO DO: add support for other delimiters, file extstensions, etc.
		"""
		self.check_for_dirs(['audio_dirs', 'segment_dirs'], 'segments')
		result = {}
		for audio_dir, seg_dir in zip(self.audio_dirs, self.segment_dirs):
			dir_result = {}
			seg_fns = [os.path.join(seg_dir, i) for i in os.listdir(seg_dir) \
				if is_seg_file(i)]
			audio_fns = [os.path.split(i)[1][:-4]+'.wav' for i in seg_fns]
			for audio_fn, seg_fn in zip(audio_fns, seg_fns):
				segs = read_columns(seg_fn, delimiter='\t', unpack=False, \
					skiprows=0)
				if len(segs) > 0:
					dir_result[audio_fn] = segs
			result[audio_dir] = dir_result
		return result


	def make_latent_means(self):
		"""
		Write latent means for the syllables in self.spec_dirs.

		Returns
		-------
		latent_means : numpy.ndarray
			Latent means of shape (max_num_syllables, z_dim)

		NOTE
		----
		- Duplicated code with <write_projection>?
		"""
		self.check_for_dirs(['projection_dirs', 'spec_dirs', 'model_filename'],\
			'latent_means')
		# First, see how many syllables are in each file.
		hdf5_file = get_hdf5s_from_dir(self.spec_dirs[0])[0]
		with h5py.File(hdf5_file, 'r') as f:
			self.sylls_per_file = len(f['specs'])
		spf = self.sylls_per_file
		# Load the model, making sure to get z_dim correct.
		z_dim = torch.load(self.model_filename)['z_dim']
		model = VAE(z_dim=z_dim)
		model.load_state(self.model_filename)
		# For each directory...
		all_latent = []
		for i in range(len(self.spec_dirs)):
			spec_dir, proj_dir = self.spec_dirs[i], self.projection_dirs[i]
			# Make the projection directory if it doesn't exist.
			if proj_dir != '' and not os.path.exists(proj_dir):
				os.makedirs(proj_dir)
			# Make a DataLoader for the syllables.
			partition = get_syllable_partition([spec_dir], 1, shuffle=False)
			loader = get_syllable_data_loaders(partition, \
				shuffle=(False,False))['train']
			# Get the latent means from the model.
			latent_means = model.get_latent(loader)
			all_latent.append(latent_means)
			# Write them to the corresponding projection directory.
			hdf5s = get_hdf5s_from_dir(spec_dir)
			assert len(latent_means) // len(hdf5s) == spf, "Inconsistent number\
				of syllables per file ("+str(len(latent_means) // len(hdf5s))+\
				") in directory "+spec_dir+". Expected "+str(spf)+"."
			for j in range(len(hdf5s)):
				filename = os.path.join(proj_dir, os.path.split(hdf5s[j])[-1])
				data = latent_means[j*spf:(j+1)*spf]
				with h5py.File(filename, 'a') as f:
					f.create_dataset('latent_means', data=data)
		return np.concatenate(all_latent)


	def make_latent_mean_umap_projection(self):
		"""Project latent means to two dimensions with UMAP."""
		# Get latent means.
		latent_means = self.request('latent_means')
		# UMAP them.
		transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
			metric='euclidean', random_state=42)
		if self.verbose:
			print("Running UMAP...")
		embedding = transform.fit_transform(latent_means)
		if self.verbose:
			print("Done.")
		# Write to files.
		self.write_projection("latent_mean_umap", embedding)
		return embedding


	def make_latent_mean_pca_projection(self):
		"""Project latent means to two dimensions with PCA."""
		# Get latent means.
		latent_means = self.request('latent_means')
		# UMAP them.
		transform = PCA(n_components=2, copy=False, random_state=42)
		if self.verbose:
			print("Running PCA...")
		embedding = transform.fit_transform(latent_means)
		if self.verbose:
			print("Done.")
		# Write to files.
		self.write_projection("latent_mean_pca", embedding)
		return embedding


	def make_feature_field(self, field, kind):
		"""
		Read a feature from a text file and put it in an hdf5 file.

		Read from self.feature_dirs and write to self.projection_dirs.
		This gets a bit tricky because we need to match up the syllables in the
		text file with the ones in the hdf5 file.

		Parameters
		----------
		field : str
			...

		kind : str, 'mupet' or 'deepsqueak'
			...

		TO DO: cleaner error handling
		"""
		self.check_for_dirs( \
			['spec_dirs', 'feature_dirs', 'projection_dirs'], field)
		# FInd which column the field is stored in.
		if kind == 'mupet':
			file_fields = MUPET_FIELDS
			onset_col = MUPET_ONSET_COL
		elif kind == 'deepsqueak':
			file_fields = DEEPSQUEAK_FIELDS
			onset_col = DEEPSQUEAK_ONSET_COL
		elif kind == 'sap':
			file_fields = SAP_FIELDS
			onset_col = SAP_ONSET_COL
		else:
			assert NotImplementedError
		field_col = file_fields.index(field)
		to_return = []
		# Run through each directory.
		for i in range(len(self.spec_dirs)):
			spec_dir = self.spec_dirs[i]
			feature_dir = self.feature_dirs[i]
			proj_dir = self.projection_dirs[i]
			hdf5s = get_hdf5s_from_dir(spec_dir)
			current_fn, k = None, None
			for hdf5 in hdf5s:
				# Get the filenames and onsets from self.spec_dirs.
				with h5py.File(hdf5, 'r') as f:
					audio_filenames = np.array(f['audio_filenames'])
					spec_onsets = np.array(f['onsets'])
					# if kind == 'sap': # SAP writes onsets in milliseconds.
					# 	spec_onsets /= 1e3
				feature_arr = np.zeros(len(spec_onsets))
				# Loop through each syllable.
				for j in range(len(spec_onsets)):
					audio_fn, spec_onset = audio_filenames[j], spec_onsets[j]
					audio_fn = audio_fn.decode('UTF-8')
					# Update the feature file, if needed.
					if audio_fn != current_fn:
						current_fn = audio_fn
						feature_fn = os.path.split(audio_fn)[-1][:-4]
						if kind == 'deepsqueak':   # DeepSqueak appends '_Stats'
							feature_fn += '_Stats' # when exporting features.
						feature_fn += '.csv'
						feature_fn = os.path.join(feature_dir, feature_fn)
						# Read the onsets and features.
						feature_onsets, features = \
							read_columns(feature_fn, [onset_col, field_col])
						if kind == 'sap': # SAP writes onsets in milliseconds.
							feature_onsets /= 1e3
						k = 0
					# Look for the corresponding onset in the feature file.
					while spec_onset > feature_onsets[k] + 0.01:
						k += 1
						assert k < len(feature_onsets)
					if abs(spec_onset - feature_onsets[k]) > 0.01:
						print("Mismatch between spec_dirs and feature_dirs!")
						print("hdf5 file:", hdf5)
						print("\tindex:", j)
						print("audio filename:", audio_fn)
						print("feature filename:", feature_fn)
						print("Didn't find spec_onset", spec_onset)
						print("in feature onsets of min:", \
								np.min(feature_onsets), "max:", \
								np.max(feature_onsets))
						print("field:", field)
						print("kind:", kind)
						quit()
					# And add it to the feature array.
					feature_arr[j] = features[k]
				# Write the fields to self.projection_dirs.
				write_fn = os.path.join(proj_dir, os.path.split(hdf5)[-1])
				with h5py.File(write_fn, 'a') as f:
					f.create_dataset(field, data=feature_arr)
				to_return.append(feature_arr)
		self.fields[field] = 1
		return np.concatenate(to_return)


	def write_projection(self, key, data):
		"""Write the given projection to self.projection_dirs."""
		sylls_per_file = self.sylls_per_file
		# For each directory...
		k = 0
		for i in range(len(self.projection_dirs)):
			spec_dir, proj_dir = self.spec_dirs[i], self.projection_dirs[i]
			hdf5s = get_hdf5s_from_dir(spec_dir)
			for j in range(len(hdf5s)):
				filename = os.path.join(proj_dir, os.path.split(hdf5s[j])[-1])
				to_write = data[k:k+sylls_per_file]
				with h5py.File(filename, 'a') as f:
					f.create_dataset(key, data=to_write)
				k += sylls_per_file


	def check_for_fields(self):
		"""Check to see which fields are saved."""
		fields = {}
		# If self.spec_dirs is registered, assume everything is there.
		if self.spec_dirs is not None:
			for field in SPEC_FIELDS:
				fields[field] = 1
		# Same for self.audio_dirs.
		if self.audio_dirs is not None:
			fields['audio'] = 1
		# Same for self.segment_dirs.
		if self.segment_dirs is not None:
			fields['segments'] = 1
			fields['segment_audio'] = 1
		# If self.projection_dirs is registered, see what we have.
		# If it's in one file, assume it's in all of them.
		if self.projection_dirs is not None:
			if os.path.exists(self.projection_dirs[0]):
				hdf5s = get_hdf5s_from_dir(self.projection_dirs[0])
				if len(hdf5s) > 0:
					hdf5 = hdf5s[0]
					with h5py.File(hdf5, 'r') as f:
						for key in f.keys():
							if key in ALL_FIELDS:
								fields[key] = 1
								self.sylls_per_file = len(f[key])
		return fields


	def check_for_dirs(self, dir_names, field):
		"""Check that the given directories exist."""
		for dir_name in dir_names:
			if dir_name == 'audio_dirs':
				temp = self.audio_dirs
			elif dir_name == 'segment_dirs':
				temp = self.segment_dirs
			elif dir_name == 'spec_dirs':
				temp = self.spec_dirs
			elif dir_name == 'feature_dirs':
				temp = self.feature_dirs
			elif dir_name == 'projection_dirs':
				temp = self.projection_dirs
			elif dir_name == 'model_filename':
				temp = self.model_filename
			else:
				raise NotImplementedError
			assert temp is not None, dir_name + " must be specified before " + \
				field + " is made!"


	def clean_projection_dir(self):
		"""Remove all the latent projections."""
		raise NotImplementedError


	def clean_plots_dir(self):
		"""Remove all the plots."""
		raise NotImplementedError



def read_columns(filename, columns=(0,1), delimiter=',', skiprows=1, \
	unpack=True):
	"""
	A wrapper around numpy.loadtxt to handle empty files.

	TO DO: Add categorical variables.
	"""
	data = np.loadtxt(filename, delimiter=delimiter, usecols=columns, \
		skiprows=skiprows).reshape(-1,len(columns))
	if unpack:
		return tuple(data[:,i] for i in range(data.shape[1]))
	return data


def is_seg_file(filename):
	"""
	Is this a segmenting file?

	TO DO: add csvs, delimiters
	"""
	return len(filename) > 4 and filename[-4:] == '.txt'


def is_wav_file(filename):
	"""Is this a wav file?"""
	return len(filename) > 4 and filename[-4:] == '.wav'


if __name__ == '__main__':
	pass


###
