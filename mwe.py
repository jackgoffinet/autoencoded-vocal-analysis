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
- Noise detection and preprocessing.process_sylls have redundant segmentation
"""
__author__ = "Jack Goffinet"
__date__ = "December 2018 - February 2019"

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
	'max_freq': 22e3,
	'nperseg': 1024,
	'noverlap': 0,
	'num_freq_bins': spec_shape[0],
	'num_time_bins': spec_shape[1],
	'spacing': 'mel',
	'time_stretch': True,
	# Segmenting parameters
	'seg_params': {
		'algorithm': read_from_file_alg,
	},
	# I/O parameters
	'max_num_files': 100,
	'sylls_per_file': 200,
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
	'max_num_files': 100,
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
	'max_num_files': 200,
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
	'max_num_files': 100,
	'sylls_per_file': 25,
}


# Zebra Finches
zebra_finch_params = {
	# Spectrogram parameters
	'fs': 44100,
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
		'spec_thresh': 0.0,
		'th_1':3.0,
		'th_2':3.5,
		'th_3':3.8,
		'min_dur':0.05,
		'max_dur':2.0,
		'freq_smoothing': 3.0,
		'softmax': True,
		'temperature': 0.5,
		'smoothing_timescale': 0.01,
		'num_freq_bins': spec_shape[0],
		'num_time_bins': spec_shape[1],
	},
	# I/O parameters
	'max_num_files': 100,
	'sylls_per_file': 50,
}


# Set the dimension of the latent space.
latent_dim = 64
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
preprocess_params = mouse_params2

"""
# 1) Tune segmenting parameters.
from os import listdir
from preprocessing.preprocessing import tune_segmenting_params
load_dirs = ['data/raw/mice_data/TVA_28_fd/']
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

mice = ['Ai_14_female', 'RAm_2_fd', 'RAm_female_2', 'RAm_6_fd', 'VM_8_fd', 'VM_31_fd', 'VM_31_opto', 'VM_47_fd', 'VM_47_opto', 'VM_75_fd', 'VM_75_opto', 'TVA_28_fd', 'TVA_28_7d_retest', 'TVA_28_14d_retest']

load_dirs = ['data/raw/mice_data/'+mouse+'/' for mouse in mice]
save_dirs = ['data/processed/mice_data/'+mouse+'/' for mouse in mice]
"""
from preprocessing.noise_detection import GaussianProcessDetector
from preprocessing.preprocessing import get_spec, get_syll_specs, get_wav_len, default_params
funcs = {
		'get_spec': get_spec,
		'get_onsets_offsets': preprocess_params['seg_params']['algorithm'],
		'get_syll_specs': get_syll_specs,
		'get_wav_len': get_wav_len,
		'default_params': default_params,
}
from preprocessing.preprocessing import process_sylls
# noise_detector = GaussianProcessDetector(load_dirs, 'mouse_labels.npy', None, \
# 			preprocess_params, funcs, ndims=3, max_num_files=100)
noise_detector = None
from multiprocessing import Pool
from itertools import repeat
# NOTE: watch the memory usage here!
with Pool(min(3, os.cpu_count()-1)) as pool:
	pool.starmap(process_sylls, zip(load_dirs, save_dirs, repeat(preprocess_params), repeat(noise_detector)))
# for load_dir, save_dir in zip(load_dirs, save_dirs):
	# process_sylls(load_dir, save_dir, preprocess_params, noise_detector=noise_detector)
quit()
"""

"""
# 4) Train a generative model on these syllables.
# save_dirs = ['data/processed/bird_data/grn288/'+str(i)+'/' for i in range(43,82,1)]
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
partition = get_partition(save_dirs, split=0.95)
# Check load_dir vs. save_dir!
model = DLGM(network_dims, partition=partition, load_dir='data/models/all_mice/', sylls_per_file=preprocess_params['sylls_per_file'])
model.train(epochs=100, lr=2e-5)
quit()
"""

# 5) Use the model to get a latent representation of these syllables.
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
model = DLGM(network_dims, load_dir='data/models/all_mice/', sylls_per_file=preprocess_params['sylls_per_file'])
partition = get_partition(save_dirs, split=1.0)
loader, _ = get_data_loaders(partition, shuffle=(False,False), batch_size=32, sylls_per_file=preprocess_params['sylls_per_file'])

from plotting.longitudinal_gif import make_projection, generate_syllables, make_dot_gif, get_embeddings_times
from plotting.html_plots import make_html_plot

title = ""

n = 10**5
print("making projection")
make_projection(loader, model, title=title, n=n, axis=False)
quit()
# print("making gif")
# make_dot_gif(loader, model, title=title, n=n)
# quit()
print("making html")
make_html_plot(loader, model, output_dir='temp/', num_imgs=2000, title=title, n=n)
quit()



print("Saving everything...")
# Save a bunch of data.
from scipy.io import savemat
import umap
return_fields = ['time', 'image', 'filename', 'duration', 'file_time']
latent, times, images, filenames, durations, file_times = \
		model.get_latent(loader, random_subset=True, return_fields=return_fields, n=10**10)
reducer = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
n = len(latent)
embedding = reducer.fit_transform(latent[:n])
np.save('grn288_reducer.npy', reducer)

d = {
	'latent':latent[:n//3],
	'times':times[:n//3],
	'file_time':file_times[:n//3],
	'images':images[:n//3],
	'filenames': filenames[:n//3],
	'durations': durations[:n//3],
	'embedding':embedding[:n//3],
}
savemat('grn288_1.mat', d)
d = {
	'latent':latent[n//3:2*n//3],
	'times':times[n//3:2*n//3],
	'file_time':file_times[n//3:2*n//3],
	'images':images[n//3:2*n//3],
	'filenames': filenames[n//3:2*n//3],
	'durations': durations[n//3:2*n//3],
	'embedding':embedding[n//3:2*n//3],
}
savemat('grn288_2.mat', d)
d = {
	'latent':latent[2*n//3:],
	'times':times[2*n//3:],
	'file_time':file_times[2*n//3:],
	'images':images[2*n//3:],
	'filenames': filenames[2*n//3:],
	'durations': durations[2*n//3:],
	'embedding':embedding[2*n//3:],
}
savemat('grn288_3.mat', d)
quit()




# 7) Generate novel audio.
# to be continued....


if __name__ == '__main__':
	pass
