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
__date__ = "December 2018 - February 2019"

import numpy as np

from preprocessing.amplitude_segmentation import get_onsets_offsets as amp_alg
from preprocessing.holy_guo_segmentation import get_onsets_offsets as holy_guo_alg
from preprocessing.preprocessing import get_onsets_offsets_from_file as read_from_file_alg


spec_shape = (128,128)


# Marmoset
preprocess_params = {
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
		'spec_thresh': -4.2,
		'th_1':0.1,
		'th_2':1.3,
		'th_3':1.0,
		'min_dur':0.03,
		'max_dur':0.35,
		'num_freq_bins': spec_shape[0],
		'num_time_bins': spec_shape[1],
	},
	# I/O parameters
	'max_num_files': 100,
	'sylls_per_file': 50,
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
		'th_1':280,
		'th_2':320,
		'th_3':375,
		'min_dur':0.05,
		'max_dur':2.0,
		'freq_smoothing': 3.0,
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
preprocess_params = mouse_params


"""
# 1) Tune segmenting parameters.
from os import listdir
from preprocessing.preprocessing import tune_segmenting_params
load_dirs = ['data/raw/mice_data/TVA_'+str(i)+'_fd/' for i in [7,9,18]]
preprocess_params = mouse_params
seg_params = tune_segmenting_params(load_dirs, preprocess_params)
preprocess_params['seg_params'] = seg_params
quit()
"""


"""
# 2) Tune noise detection.
# days = [i for i in range(59,123,1) if i not in [62]]
# load_dirs = ['data/raw/bird_data/blu258/'+str(i)+'/' for i in days]
load_dirs = ['data/raw/mice_data/TVA_'+str(i)+'_fd/' for i in [7,9,17,18,19,27,28,60]]
from preprocessing.preprocessing import default_params, get_spec, get_wav_len, get_onsets_offsets, get_syll_specs, get_audio
funcs = {
	'default_params':default_params,
	'get_spec':get_spec,
	'get_audio':get_audio,
	'get_wav_len':get_wav_len,
	'get_onsets_offsets':get_onsets_offsets,
	'get_syll_specs':get_syll_specs,
}
from preprocessing.noise_detection import GaussianProcessDetector
detector = GaussianProcessDetector(load_dirs, 'TVA_labels.npy', 'TVA_labels.npy', preprocess_params, funcs, ndims=3)
detector.train()
quit()
"""


"""
# 3) Segment audio into syllables.
from preprocessing.noise_detection import GaussianProcessDetector
from preprocessing.preprocessing import get_spec, get_onsets_offsets, get_syll_specs, get_wav_len, default_params
funcs = {
		'get_spec': get_spec,
		'get_onsets_offsets': get_onsets_offsets,
		'get_syll_specs': get_syll_specs,
		'get_wav_len': get_wav_len,
		'default_params': default_params,
}
from preprocessing.preprocessing import process_sylls
load_dirs = ['data/raw/mice_data/TVA_'+str(i)+'_fd/' for i in [7,9,17,18,19,27,28,60]]
save_dirs = ['data/processed/mice_data/TVA_'+str(i)+'_fd/' for i in [7,9,17,18,19,27,28,60]]
noise_detector = GaussianProcessDetector(load_dirs, 'TVA_labels.npy', None, \
			preprocess_params, funcs, ndims=3, max_num_files=10)
for load_dir, save_dir in zip(load_dirs, save_dirs):
	process_sylls(load_dir, save_dir, preprocess_params, noise_detector=noise_detector)
quit()
"""


"""
# 4) Train a generative model on these syllables.
save_dirs = ['data/processed/mice_data/TVA_'+str(i)+'_fd/' for i in [7,9,17,18,19,27,28,60]]
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
partition = get_partition(save_dirs, split=0.8)
# Check load_dir vs. save_dir!
model = DLGM(network_dims, partition=partition, load_dir='data/models/TVA_mice/', sylls_per_file=preprocess_params['sylls_per_file'])
model.train(epochs=100, lr=1.5e-5)
quit()
"""


# 5) Use the model to get a latent representation of these syllables.
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
model = DLGM(network_dims, load_dir='data/models/TVA_mice/', sylls_per_file=preprocess_params['sylls_per_file'])
save_dirs = ['data/processed/mice_data/TVA_'+str(i)+'_fd/' for i in [7,9,17,18,19,27,28,60]]
partition = get_partition(save_dirs, split=1.0)
loader, _ = get_data_loaders(partition, shuffle=(False,False), batch_size=32, sylls_per_file=preprocess_params['sylls_per_file'])

from plotting.longitudinal_gif import make_projection, generate_syllables, make_kde_gif, make_time_heatmap, make_dot_gif, get_embeddings_times
from plotting.html_plots import make_html_plot

print("making projection")
make_projection(loader, model, title="TVA USV distribution")
quit()
# print("making gif")
# make_dot_gif(loader, model)
# quit()
print("making html")
make_html_plot(loader, model, output_dir='temp5/', num_imgs=2000, title="TVA USV distribution")
quit()



print("Saving everything...")
# Save a bunch of data.
from scipy.io import savemat
import umap
return_fields = ['time', 'image', 'filename', 'duration']
latent, times, images, filenames, durations = model.get_latent(loader, random_subset=True, return_fields=return_fields)
reducer = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(latent)
d = {
	'latent':latent,
	'times':times,
	'images':images,
	'filenames': filenames,
	'durations': durations,
	'embedding':embedding,
}
savemat('TVA.mat', d)
np.save('TVA_reducer.npy', reducer)
quit()


make_dot_gif(loader, model)
quit()


# 7) Generate novel audio.
# to be continued....


if __name__ == '__main__':
	pass
