"""
Minimal working example for generative modeling of acoustic syllables.

0) Insert directories of .wav files in data/raw/ . Optionally include metadata
	saved as a python dictionary in <meta.npy>.
1) Tune segmenting parameters.
2) Segment audio into syllables.
3) Train a generative model on these syllables.
4) Use the model to get a latent representation of these syllables.
5) Visualize these latent representations.
6) Generate novel audio.

BEFORE RUNNING:
- change load and save directories below.
- look at time_from_filename in preprocessing/preprocessing.py

TO DO:
- Random seed
"""
__author__ = "Jack Goffinet"
__date__ = "December 2018 - May 2019"

import numpy as np

from preprocessing.amplitude_segmentation import get_onsets_offsets as amp_alg
from preprocessing.holy_guo_segmentation import get_onsets_offsets as holy_guo_alg
from preprocessing.preprocessing import get_onsets_offsets_from_file as read_from_file_alg
# from preprocessing.template_segmentation import get_onsets_offsets as template_alg

spec_shape = (128,128)


# Marmoset
marmoset_params = {
	# Spectrogram parameters
	'fs': 96000,
	'min_freq': 1e3,
	'max_freq': 25e3,
	'nperseg': 2048,
	'noverlap': 512-64,
	'num_freq_bins': spec_shape[0],
	'num_time_bins': spec_shape[1],
	'mel': True,
	'time_stretch': True,
	# Segmenting parameters
	'seg_params': {
		'algorithm': read_from_file_alg,
		'max_dur': 3.0,
		'min_dur': 1e-6,
		'spec_thresh': 0.0,
	},
	# I/O parameters
	'max_num_syllables': 1200, # per directory
	'sylls_per_file': 25,
}


# Heliox Mice
heliox_mice_params = {
	# Spectrogram parameters
	'fs': 303030, # Tom is at 300k, Katie & Val at 250k
	'min_freq': 25e3,
	'max_freq': 100e3,
	'nperseg': 1024, # FFT
	'noverlap': 0, # FFT
	'num_freq_bins': spec_shape[0],
	'num_time_bins': spec_shape[1],
	'mel': False, # Frequency spacing
	'time_stretch': True,
	# Segmenting parameters
	'seg_params': {
		'algorithm': amp_alg,
		'spec_thresh': -6.0,
		'th_1':5.5,
		'th_2':6.5,
		'th_3':8.0,
		'min_dur':0.05,
		'max_dur':0.5,
		'freq_smoothing': 3.0,
		'smoothing_timescale': 0.01,
		'num_freq_bins': spec_shape[0],
		'num_time_bins': spec_shape[1],
	},
	# I/O parameters
	'max_num_syllables': 1200, # per directory
	'sylls_per_file': 25,
}


# Other Mice
mouse_params = {
	# Spectrogram parameters
	'fs': 250000,
	'min_freq': 25e3,
	'max_freq': 110e3,
	'nperseg': 1024, # FFT
	'noverlap': 0, # FFT
	'num_freq_bins': spec_shape[0],
	'num_time_bins': spec_shape[1],
	'mel': False, # Frequency spacing
	'time_stretch': True,
	# Segmenting parameters
	'seg_params': {
		'algorithm': holy_guo_alg,
		'spec_thresh': -4.0,
		'th_1': 0.1,
		'th_2': 1.4,
		'min_dur':0.03,
		'max_dur':0.35,
		'num_freq_bins': spec_shape[0],
		'num_time_bins': spec_shape[1],
	},
	# I/O parameters
	'max_num_syllables': 1200, # per directory
	'sylls_per_file': 25,
}

# Normal mice, with amplitude segmentation
mouse_params2 = {
	# Spectrogram parameters
	'fs': 303030, # Tom is at 300k, Katie & Val at 250k, Tom also 303030
	'min_freq': 30e3,
	'max_freq': 135e3,
	'nperseg': 1024, # FFT
	'noverlap': 0, # FFT
	'num_freq_bins': spec_shape[0],
	'num_time_bins': spec_shape[1],
	'mel': False, # Frequency spacing
	'time_stretch': True,
	# Segmenting parameters
	'seg_params': {
		'algorithm': amp_alg,
		'spec_thresh': -5.7, # -5.5
		'th_1':2.0,
		'th_2':2.5,
		'th_3':3.5, # 4.0
		'min_dur':0.02,
		'max_dur':0.5,
		'freq_smoothing': 2.0,
		'smoothing_timescale': 0.006,
		'num_freq_bins': spec_shape[0],
		'num_time_bins': spec_shape[1],
	},
	# I/O parameters
	'max_num_syllables': 2000, # per directory
	'sylls_per_file': 25,
}


# Zebra Finches
zebra_finch_params = {
	# Spectrogram parameters
	'fs': 32000, # 44100/32000
	'min_freq': 300,
	'max_freq': 12e3,
	'nperseg': 512, # FFT
	'noverlap': 512-128-64, # FFT
	'num_freq_bins': spec_shape[0],
	'num_time_bins': spec_shape[1],
	'mel': True, # Frequency spacing
	'time_stretch': True,
	# Segmenting parameters
	'seg_params': {
		'algorithm': amp_alg,
		'spec_thresh': 1.5,
		'th_1':0.9,
		'th_2':1.0,
		'th_3':1.35,
		'min_dur':0.05,
		'max_dur':2.0,
		'freq_smoothing': 3.0,
		'softmax': True,
		'temperature': 0.35,
		'smoothing_timescale': 0.008,
		'num_freq_bins': spec_shape[0],
		'num_time_bins': spec_shape[1],
	},
	# I/O parameters
	'sylls_per_file': 50,
	'max_num_syllables': 1400, # per directory
}


# Set the dimension of the latent space.
latent_dim = 32
nf = 8
encoder_conv_layers =	[
		[1,1*nf,3,1,1],
		[1*nf,1*nf,3,2,1],
		[1*nf,2*nf,3,1,1],
		[2*nf,2*nf,3,2,1],
		[2*nf,3*nf,3,1,1],
		[3*nf,3*nf,3,2,1],
		[3*nf,4*nf,3,1,1],
]

# [in_features, out_features]
encoder_dense_layers =	[
		[2**13, 2**10],
		[2**10, 2**8],
		[2**8, 2**6],
		[2**6, latent_dim],
]

network_dims = {
		'input_shape':spec_shape,
		'input_dim':np.prod(spec_shape),
		'latent_dim':latent_dim,
		'post_conv_shape':(4*nf,16,16),
		'post_conv_dim':np.prod([4*nf,16,16]),
		'encoder_conv_layers':encoder_conv_layers,
		'encoder_fc_layers':encoder_dense_layers,
		'decoder_convt_layers':[[i[1],i[0]]+i[2:] for i in encoder_conv_layers[::-1]],
		'decoder_fc_layers':[i[::-1] for i in encoder_dense_layers[::-1]],
}


"""
# -------------------------------------- #
#        Instantaneous Song Stuff        #
# -------------------------------------- #
from preprocessing.template_segmentation import process_sylls, clean_collected_data
syll_types = ["E", "A", "B", "C", "D", "call"]
# temp = ['09262019/', '09292019/', 'IMAGING_09182018/', 'IMAGING_09242018/', \
# 		'IMAGING_10112018/', 'IMAGING_10162018/']
temp = ['09262019/']
load_dirs = ['data/raw/bird_data/' + i for i in temp]
temp_save_dirs = ['data/processed/bird_data/temp_' + i for i in temp]
save_dirs = ['data/processed/bird_data/' + i for i in temp]
feature_dirs = ['data/features/blk215/' + i for i in syll_types]

p = {
	'songs_per_file': 20,
	'num_freq_bins': 128,
	'num_time_bins': 128,
	'min_freq': 350,
	'max_freq': 10e3,
	'mel': True,
	'spec_thresh': 1.0,
}

# Preprocessing
for syll_type, feature_dir in zip(syll_types, feature_dirs):
	print("Syllable:", syll_type)
	temp_save_dirs = ['data/processed/bird_data/temp_' + syll_type + '_'+ i for i in temp]
	save_dirs = ['data/processed/bird_data/' + syll_type + '_' + i for i in temp]
	for load_dir, temp_save_dir in zip(load_dirs, temp_save_dirs):
		process_sylls(load_dir, temp_save_dir, feature_dir, p)
	clean_collected_data(temp_save_dirs, save_dirs, p)

quit()
# Training: syllables
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
partition = get_partition(save_dirs, split=0.95)
# Check load_dir vs. save_dir!
model = DLGM(network_dims, partition=partition, save_dir='data/models/red215_syll/', sylls_per_file=p['songs_per_file'])
model.train(epochs=250, lr=2e-5)
quit()

# Training: fixed window
from models.fixed_window_dlgm import DLGM
from models.fixed_window_dataset import get_partition, get_data_loaders
partition = get_partition(save_dirs, split=0.95)
# Check load_dir vs. save_dir!
model = DLGM(network_dims, p, partition=partition, load_dir='data/models/blk215_inst/')
# model.train(epochs=250, lr=1.5e-5)

partition = get_partition(['data/processed/bird_data/IMAGING_09182018/'], split=1.0)
# temp = ['IMAGING_09242018/', 'IMAGING_10112018/', \
		# 'IMAGING_10162018/'] # '09262019/', '09292019/',
# temp_dirs = ['data/processed/bird_data/'+i for i in temp]
# partition = get_partition(temp_dirs, split=1.0)
loader, _ = get_data_loaders(partition, p, shuffle=(False, False))


# n = min(len(loader.dataset), 400)
# latent_paths = np.zeros((n,200,latent_dim))
# print("loader:", len(loader.dataset))
# from tqdm import tqdm
# for i in tqdm(range(n)):
# 	latent, ts = model.get_song_latent(loader, i, n=200)
# 	latent_paths[i] = latent
# np.save('latent_paths_other.npy', latent_paths)
# quit()

ts = np.linspace(0,0.85,200)
# latent_paths_1 = np.load('latent_paths_imaging.npy')
# latent_paths_2 = np.load('latent_paths_other.npy')
# latent_paths = np.concatenate((latent_paths_1, latent_paths_2), axis=0)

latent_paths = np.load('latent_paths_imaging.npy')

from plotting.instantaneous_plots import plot_paths_imaging
for unit in range(53):
	plot_paths_imaging(np.copy(latent_paths), ts, loader, unit_num=unit, filename=str(unit).zfill(2)+'.pdf')
quit()
# -------------------------------------- #
"""

# Set which set of parameters to use.
preprocess_params = mouse_params2


load_dirs = ['data/raw/helium_mice/'+str(i)+'_He/' for i in [0,20,30,40,50,60,80]]
save_dirs = ['data/processed/helium_mice/'+str(i)+'_He/' for i in [0,20,30,40,50,60,80]]
load_dirs += ['data/raw/helium_mice/extras/']
save_dirs += ['data/processed/helium_mice/extras']

"""
# 1) Tune segmenting parameters.
from os import listdir
from preprocessing.preprocessing import tune_segmenting_params
seg_params = tune_segmenting_params(load_dirs, preprocess_params)
preprocess_params['seg_params'] = seg_params
quit()
"""

"""
# 2) Tune noise detection.
load_dirs = ['data/raw/mice_data/TVA_'+str(i)+'_fd/' for i in [7,9,17,18,19,27,28,60,63]]
load_dirs +=  ['data/raw/mice_data/VM_'+str(i)+'_fd/' for i in [1,5,6,31,75,86]]
from preprocessing.preprocessing import default_params, get_spec, get_wav_len, get_syll_specs, get_audio
funcs = {
	'default_params':default_params,
	'get_spec':get_spec,
	'get_audio':get_audio,
	'get_wav_len':get_wav_len,
	'get_onsets_offsets':preprocess_params['seg_params']['algorithm'],
	'get_syll_specs':get_syll_specs,
}
from preprocessing.noise_detection import GaussianProcessDetector
detector = GaussianProcessDetector(load_dirs, None, 'mouse_labels.npy', preprocess_params, funcs, ndims=3)
detector.train()
quit()
"""

"""
# 3) Segment audio into syllables.
import os

# template_dir = 'data/templates/red291/'
# load_dirs = ['data/raw/bird_data/'+str(i)+'/' for i in range(80,85)]
# save_dirs = ['hdf5_files/'+str(i)+'/' for i in range(80,85)]

# from preprocessing.template_segmentation import process_sylls
from preprocessing.preprocessing import process_sylls

# process_sylls(load_dirs[0], save_dirs[0], preprocess_params, None)
# quit()

noise_detector = None
from multiprocessing import Pool
from itertools import repeat
with Pool(min(3, os.cpu_count()-1)) as pool:
	pool.starmap(process_sylls, zip(load_dirs, save_dirs, repeat(preprocess_params), repeat(noise_detector)))
quit()
"""


# 4) Train a generative model on these syllables.
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
partition = get_partition(save_dirs, split=0.95)
# Check load_dir vs. save_dir!
model = DLGM(network_dims, partition=partition, save_dir='data/models/helium_mice_cpu/', sylls_per_file=preprocess_params['sylls_per_file'])
model.train(epochs=300, lr=1e-5)
quit()


# 5) Use the model to get a latent representation of these syllables.
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
model = DLGM(network_dims, load_dir='data/models/helium_mice', sylls_per_file=preprocess_params['sylls_per_file'])
partition = get_partition(save_dirs, split=1.0)
loader, _ = get_data_loaders(partition, shuffle=(False,False), batch_size=32, sylls_per_file=preprocess_params['sylls_per_file'])

d = {'model':model, 'loader':loader}

from plotting.longitudinal_gif import make_projection, plot_generated_cluster_means, make_dot_gif, make_html_plot


title = ""
n = 3*10**4


print("making projection")
d = make_projection(d, title=title, n=n, axis=False)
quit()

"""
print("making gif")
d = make_dot_gif(d, title=title, n=n)
"""

print("making html")
make_html_plot(d, output_dir='temp/', n=n, num_imgs=2000, title=title)

np.save('d.npy', d)
quit()

# print("Saving everything...")
# # Save a bunch of data.
# from scipy.io import savemat

# from plotting.longitudinal_gif import update_data
# keys = ['latent', 'file_time', 'time', 'filename', 'duration', 'embedding']
# d = update_data(d, keys, n=n) # + ['image']

"""
savemat('images.mat', {'images':d['image']})
del d['image']

del d['model']
del d['loader']
savemat('data.mat', d)
"""

import joblib
reducer = joblib.load('temp_reducer.sav')
from save_stuff import save_everything
for i in range(len(load_dirs)):
	save_everything(model, load_dirs[i], save_dirs[i], preprocess_params, reducer)
quit()


# 7) Generate novel audio.
# to be continued....


if __name__ == '__main__':
	pass
