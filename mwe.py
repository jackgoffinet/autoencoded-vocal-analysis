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
- Thresholds not being passed
- Random seed
- More efficient dataloader implementation?
"""
__author__ = "Jack Goffinet"
__date__ = "December 2018 - January 2019"

import numpy as np

spec_shape = (128,128)


# # Marmoset
# freq_response = np.ones(spec_shape[0])
# # freq_response[int(0.55*spec_shape[0]):int(0.70*spec_shape[0])] = 1.0
# preprocess_params = {
# 	# Spectrogram parameters
# 	'fs': 96000,
# 	'min_freq': 1e3,
# 	'max_freq': 22e3,
# 	'nperseg': 1024,
# 	'noverlap': 0,
# 	'spec_percentile': 90.0,
# 	'num_freq_bins': spec_shape[0],
# 	'num_time_bins': spec_shape[1],
# 	'spacing': 'mel',
# 	# Segmenting parameters
# 	'seg_params': {
# 		'th_1':0.1,
# 		'th_2':0.2,
# 		'min_var':0.0,
# 		'min_dur':0.05,
# 		'max_dur':0.5,
# 		'freq_smoothing': 3.0,
# 		'smoothing_timescale': 0.01,
# 		'num_freq_bins': spec_shape[0],
# 		'num_time_bins': spec_shape[1],
# 		'freq_response': freq_response,
# 	},
# 	# I/O parameters
# 	'max_num_files': 100,
# 	'sylls_per_file': 200,
# 	'meta': {},
# }

# # Helium Mice
# freq_response = np.ones(spec_shape[0])
# freq_response[spec_shape[0]//4:] = 1.0
# preprocess_params = {
# 	# Spectrogram parameters
# 	'fs': 300000, # Tom is at 300k, Katie & Val at 250k
# 	'min_freq': 300,
# 	'max_freq': 12e3,
# 	'nperseg': 1024, # FFT
# 	'noverlap': 0, # FFT
# 	'num_freq_bins': spec_shape[0],
# 	'num_time_bins': spec_shape[1],
# 	'spacing': 'mel', # Frequency spacing
# 	'time_stretch': True,
# 	# Segmenting parameters
# 	'seg_params': {
# 		'spec_thresh': -6.0,
# 		'th_1':6.0,
# 		'th_2':8.0,
# 		'min_var':0.0,
# 		'min_dur':0.05,
# 		'max_dur':0.5,
# 		'freq_smoothing': 3.0,
# 		'smoothing_timescale': 0.01,
# 		'num_freq_bins': spec_shape[0],
# 		'num_time_bins': spec_shape[1],
# 		'freq_response': freq_response,
# 		'is_noise': lambda x : 3 * np.sum(x[:len(x)//4]) > np.sum(x[len(x)//4:]),
# 		# 'is_noise': lambda x : False,
# 	},
# 	# I/O parameters
# 	'max_num_files': 100,
# 	'sylls_per_file': 25,
# 	'meta': {},
# }

# Zebra Finches
freq_response = np.ones(spec_shape[0])
preprocess_params = {
	# Spectrogram parameters
	'fs': 44100,
	'min_freq': 300,
	'max_freq': 12e3,
	'nperseg': 512, # FFT
	'noverlap': 512-128-64, # FFT
	'num_freq_bins': spec_shape[0],
	'num_time_bins': spec_shape[1],
	'spacing': 'mel', # Frequency spacing
	'time_stretch': True,
	# Segmenting parameters
	'seg_params': {
		'spec_thresh': 0.0,
		'th_1':200,
		'th_2':240,
		'th_3':280,
		'min_dur':0.05,
		'max_dur':2.0,
		'freq_smoothing': 3.0,
		'smoothing_timescale': 0.01,
		'num_freq_bins': spec_shape[0],
		'num_time_bins': spec_shape[1],
		'freq_response': freq_response,
		# 'is_noise': lambda x : np.max(np.mean(np.abs(np.diff(x, axis=0)), axis=0)) < 0.56,
		'is_noise': lambda x : False,
	},
	# I/O parameters
	'max_num_files': 100,
	'sylls_per_file': 50,
	'meta': {},
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



# 1) Tune segmenting parameters.
load_dirs = ['data/raw/bird_data/grn151/'+i+'/' for i in ['4020','4030','4040','4070']]
from preprocessing.preprocessing import tune_segmenting_params
seg_params = tune_segmenting_params(load_dirs, preprocess_params)
preprocess_params['seg_params'] = seg_params

quit()




# 2) Segment audio into syllables.
from preprocessing.preprocessing import process_sylls
load_dirs = ['data/raw/bird_data/'+str(i)+'/' for i in range(48,91,1)]
save_dirs = ['data/processed/bird_data/'+str(i)+'/' for i in range(48,91,1)]
"""
for load_dir, save_dir in zip(load_dirs, save_dirs):
	try:
		preprocess_params['meta'] = np.load(load_dir + 'meta.npy').item()
	except FileNotFoundError:
		pass
	process_sylls(load_dir, save_dir, preprocess_params)
quit()
"""

"""
# 3) Train a generative model on these syllables.
# save_root = 'data/processed/helium_mice/BM_00'
# save_dirs = [save_root+str(i)+'_s'+str(j)+'/' for i in [3,4,5] for j in range(1,6,1)]
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
partition = get_partition(save_dirs, split=0.8)
# Check load_dir vs. save_dir!
model = DLGM(network_dims, partition=partition, save_dir='data/models/pur224/', sylls_per_file=preprocess_params['sylls_per_file'])
model.train(epochs=400)
quit()
"""

# 4) Use the model to get a latent representation of these syllables.
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
model = DLGM(network_dims, load_dir='data/models/helium/', sylls_per_file=preprocess_params['sylls_per_file'])
save_root = 'data/processed/helium_mice/BM_00'
save_dirs = [save_root+str(i)+'_s'+str(j)+'/' for i in [3,4,5] for j in range(1,6,1)]
partition = get_partition(save_dirs, split=1.0)
loader, _ = get_data_loaders(partition, shuffle=(False,False), time_shift=(False, False), sylls_per_file=preprocess_params['sylls_per_file'])

from plotting.longitudinal_gif import make_projection, generate_syllables, make_kde_gif, make_time_heatmap
from plotting.html_plots import make_html_plot
make_html_plot(loader, model, output_dir='temp2/')
quit()
# latent = model.get_latent(loader, n=10000)
# np.save('latent.npy', latent)


# 6) Generate novel audio.
# to be continued....


if __name__ == '__main__':
	pass
