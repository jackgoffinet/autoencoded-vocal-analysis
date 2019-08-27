"""
Save stuff.

TO DO: replicate in plotting w/ DataContainer.
"""
__author__ = "Jack Goffinet"
__date__ = "April 2019"

import os
from tqdm import tqdm
import h5py

import numpy as np
from scipy.io import savemat

from preprocessing.preprocessing import time_from_filename, get_spec, \
		get_onsets_offsets_from_file, get_syll_specs
from plotting.longitudinal_gif import update_data



# def save_things(data, n, filename='data.mat'):
# 	keys = ['latent', 'file_time', 'time', 'filename', 'duration', 'embedding']
# 	del data['model']
# 	del data['loader']
# 	data = update_data(data, keys, n=n)
# 	savemat(filename, data)


def save_everything(model, load_dir, save_dir, p, reducer, prefix=''):
	if len(load_dir) > 0 and load_dir[-1] != '/':
		load_dir += '/'
	if len(save_dir) > 0 and save_dir[-1] != '/':
		save_dir += '/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	filenames = [load_dir + i for i in os.listdir(load_dir) if i[-4:] in \
			['.wav', '.mat']]
	if p['seg_params']['algorithm'] == get_onsets_offsets_from_file:
		filenames = [i for i in filenames if os.path.exists('.'.join(i.split('.')[:-1]) + '.txt')]
	syll_data = {
		'specs':[],
		'times':[],
		'file_times':[],
		'durations':[],
		'filenames':[],
	}
	sylls_per_file = 1000
	num_time_bins, num_freq_bins = p['num_time_bins'], p['num_freq_bins']
	write_file_num = 0
	print("num files in",load_dir+":", len(filenames))
	for i, load_filename in tqdm(enumerate(filenames)):
		start_time = time_from_filename(load_filename)
		# Get a spectrogram.
		spec, f, dt, i1, i2 = get_spec(load_filename, p)
		# Collect syllable onsets and offsets.
		if 'f' not in p['seg_params']:
			p['seg_params']['f'] = f
		# Get onsets and offsets.
		if p['seg_params']['algorithm'] == get_onsets_offsets_from_file:
			t_onsets, t_offsets = get_onsets_offsets_from_file(load_filename, dt)
		else:
			t_onsets, t_offsets = p['seg_params']['algorithm'](spec, dt, \
					p['seg_params'])
		t_durations = [(b-a+1)*dt for a,b in zip(t_onsets, t_offsets)]
		# Retrieve spectrograms and start times for each detected syllable.
		t_specs, t_times = get_syll_specs(t_onsets, t_offsets, spec, \
				start_time, dt, p)

		syll_data['durations'] += t_durations
		syll_data['specs'] += t_specs
		syll_data['times'] += t_times
		syll_data['file_times'] += [i - start_time for i in t_times]
		syll_data['filenames'] += len(t_durations)*[load_filename.split('/')[-1]]

		while len(syll_data['durations']) >= sylls_per_file:
			# Generate latent descriptions.
			temp = np.zeros((sylls_per_file, num_freq_bins, num_time_bins),
					dtype='float')
			syll_specs = syll_data['specs']
			for i in range(sylls_per_file):
				gap = max(0, (num_time_bins - syll_specs[i].shape[1]) // 2)
				temp_spec = syll_specs[i]
				temp_spec *= 0.9 / np.max(temp_spec)
				temp_spec += 0.05
				temp[i,:,gap:gap+syll_specs[i].shape[1]] = temp_spec[:,:num_time_bins]
			# Call the model.
			latent = model.specs_to_latent(temp)
			# Save things.
			save_filename = os.path.join(save_dir, prefix+str(write_file_num).zfill(2)+'.hdf5')
			with h5py.File(save_filename, "w") as f:
				f.create_dataset('latent', data=latent)
				for k in ['durations', 'times', 'file_times']:
					f.create_dataset(k, data=np.array(syll_data[k][:sylls_per_file]))
				temp = [save_dir + i for i in syll_data['filenames'][:sylls_per_file]]
				f.create_dataset('filenames', data=np.array(temp).astype('S'))
			# Remove the written data from temporary storage.
			for k in syll_data:
				syll_data[k] = syll_data[k][sylls_per_file:]
			write_file_num += 1





###
