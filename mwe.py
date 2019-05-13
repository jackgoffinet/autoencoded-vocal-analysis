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
__date__ = "December 2018 - April 2019"

import numpy as np

from preprocessing.amplitude_segmentation import get_onsets_offsets as amp_alg
from preprocessing.holy_guo_segmentation import get_onsets_offsets as holy_guo_alg
from preprocessing.preprocessing import get_onsets_offsets_from_file as read_from_file_alg


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
	'fs': 300000, # Tom is at 300k, Katie & Val at 250k
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
		'th_1':6.0,
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
# TO DO: pass params for holy/guo
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
	'fs': 250000, # Tom is at 300k, Katie & Val at 250k
	'min_freq': 35e3,
	'max_freq': 110e3,
	'nperseg': 1024, # FFT
	'noverlap': 0, # FFT
	'num_freq_bins': spec_shape[0],
	'num_time_bins': spec_shape[1],
	'mel': False, # Frequency spacing
	'time_stretch': True,
	# Segmenting parameters
	'seg_params': {
		'algorithm': amp_alg,
		'spec_thresh': -4.0,
		'th_1':6.0,
		'th_2':6.5,
		'th_3':8.0,
		'min_dur':0.02,
		'max_dur':0.5,
		'freq_smoothing': 3.0,
		'smoothing_timescale': 0.008,
		'num_freq_bins': spec_shape[0],
		'num_time_bins': spec_shape[1],
	},
	# I/O parameters
	'max_num_syllables': 1200, # per directory
	'sylls_per_file': 25,
}


# Zebra Finches
zebra_finch_params = {
	# Spectrogram parameters
	'fs': 44100, # 44100/32000
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
		'algorithm': read_from_file_alg, # amp_alg
		'spec_thresh': 2.5,
		'th_1':0.9,
		'th_2':1.0,
		'th_3':1.4,
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
	'max_num_syllables': 1200, # per directory
}


# Set the dimension of the latent space.
latent_dim = 8 # NOTE: TEMP!
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


# Set which set of parameters to use.
preprocess_params = zebra_finch_params

from preprocessing.template_segmentation import process_sylls, clean_collected_data
load_dir = 'data/raw/bird_data/84/'
save_dir = 'data/processed/bird_data/temp_84/'
save_dir2 = 'data/processed/bird_data/temp2_84/'
feature_dir = 'data/features/red291/'

p = {
	'songs_per_file': 50,
	'num_freq_bins': 128,
	'num_time_bins': 128,
	'min_freq': 350,
	'max_freq': 12e3,
	'mel': True,
	'spec_thresh': -4.0,
	'fs': 44100,
	'spec_dur': 0.1,
}
# process_sylls(load_dir, save_dir, feature_dir, p)
# clean_collected_data([save_dir], [save_dir2], p)
from models.fixed_window_dlgm import DLGM
from models.fixed_window_dataset import get_partition, get_data_loaders

# partition = get_partition([save_dir2], split=0.95)
# Check load_dir vs. save_dir!
# model = DLGM(network_dims, p, partition=partition, load_dir='data/models/red291_inst/', songs_per_file=p['songs_per_file'])
# model.train(epochs=250, lr=1e-5)

partition = get_partition([save_dir2], split=1.0)
loader, _ = get_data_loaders(partition, p, shuffle=(False,False), batch_size=32, songs_per_file=p['songs_per_file'])
model = DLGM(network_dims, p, partition=partition, load_dir='data/models/red291_inst/', songs_per_file=p['songs_per_file'])

n = 100
latent_paths = np.zeros((n,200,8))
for i in range(1):
	latent, ts = model.get_song_latent(loader, i, n=200)
	latent_paths[i] = latent

# np.save('latent_paths.npy', latent_paths)
latent_paths = np.load('latent_paths.npy')

from plotting.instantaneous_plots import plot_paths
plot_paths(latent_paths, ts)
quit()
# np.save('latent.npy', latent)
from sklearn.decomposition import PCA
pca = PCA(n_components=8)
latent_1 = pca.fit_transform(latent_1)
print(pca.explained_variance_ratio_)
latent_2 = pca.transform(latent_2)
latent_3 = pca.transform(latent_3)
# latent_2 = pca.

# embed = latent[:]
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.plot(ts, latent_1[:,1], lw=0.5)
plt.plot(ts, latent_2[:,1], lw=0.5)
plt.plot(ts, latent_3[:,1], lw=0.5)
# plt.scatter(latent_2[:,0], latent_2[:,1])
plt.savefig('temp1.pdf')
quit()

load_dirs = ['data/raw/bird_data/'+str(i)+'/' for i in range(42,85)]
save_dirs = ['data/processed/bird_data/'+str(i)+'/' for i in range(42,85)]
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


# 3) Segment audio into syllables.
import os

template_dir = 'data/templates/red291/'
load_dirs = ['data/raw/bird_data/'+str(i)+'/' for i in range(80,85)]
save_dirs = ['hdf5_files/'+str(i)+'/' for i in range(80,85)]

from preprocessing.template_segmentation import process_sylls
# from preprocessing.preprocessing import process_sylls
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
model = DLGM(network_dims, partition=partition, save_dir='data/models/red291/', sylls_per_file=preprocess_params['sylls_per_file'])
model.train(epochs=250, lr=2e-5)
quit()
"""

# 5) Use the model to get a latent representation of these syllables.
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
model = DLGM(network_dims, load_dir='data/models/red291', sylls_per_file=preprocess_params['sylls_per_file'])
partition = get_partition(save_dirs, split=1.0)
loader, _ = get_data_loaders(partition, shuffle=(False,False), batch_size=32, sylls_per_file=preprocess_params['sylls_per_file'])

d = {'model':model, 'loader':loader}

from plotting.longitudinal_gif import make_projection, plot_generated_cluster_means, make_dot_gif, make_html_plot

load_dirs = ['data/raw/bird_data/'+str(i)+'/' for i in range(42,85)]
save_dirs = ['hdf5_files/'+str(i)+'/' for i in range(42,85)]


title = "red291"
n = 3*10**4

"""
print("making projection")
d = make_projection(d, title=title, n=n, axis=False)


print("making gif")
d = make_dot_gif(d, title=title, n=n)

# quit()
print("making html")
make_html_plot(d, output_dir='temp/', n=n, num_imgs=2000, title=title)
# quit()


print("Saving everything...")
# Save a bunch of data.
from scipy.io import savemat

from plotting.longitudinal_gif import update_data
keys = ['latent', 'file_time', 'time', 'filename', 'duration', 'embedding']
d = update_data(d, keys, n=n) # + ['image']

# savemat('images.mat', {'images':d['image']})
# del d['image']

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
