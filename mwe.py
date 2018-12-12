from __future__ import print_function, division
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
__date__ = "December 2018"

import numpy as np

spec_shape = (128,128)

# # Zebra Finch
# preprocess_params = {
# 	'fs':44100.0,
# 	'min_freq':300,
# 	'max_freq':12e3,
# 	'nperseg':512,
# 	'noverlap':512-128-64,
# 	'num_freq_bins':spec_shape[0],
# 	'num_time_bins':spec_shape[1],
# 	'meta': {},
# }

# Marmoset
freq_response = np.zeros(spec_shape[0])
freq_response[int(0.55*spec_shape[0]):int(0.70*spec_shape[0])] = 1.0
preprocess_params = {
	# Spectrogram parameters
	'fs': 96000,
	'min_freq': 1e3,
	'max_freq': 22e3,
	'nperseg': 1024,
	'noverlap': 0,
	'spec_percentile': 90.0,
	'num_freq_bins': spec_shape[0],
	'num_time_bins': spec_shape[1],
	'spacing': 'mel',
	# Segmenting parameters
	'seg_params': {
		'a_onset':0.1,
		'a_offset':0.05,
		'a_dot_onset':0.0,
		'a_dot_offset':0.0,
		'min_var':0.0,
		'min_dur':0.1,
		'max_dur':2.0,
		'freq_smoothing': 3.0,
		'smoothing_timescale': 0.01,
		'num_freq_bins': spec_shape[0],
		'num_time_bins': spec_shape[1],
		'freq_response': freq_response,
	},
	# I/O parameters
	'max_num_files': 100,
	'sylls_per_file': 200,
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

"""
# Visualize some stuff.
from plotting.longitudinal_gif import make_gif, make_kde_gif, make_projection, make_time_heatmap
from plotting.step_sizes import make_step_size_plot
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders

save_directories = ['data/processed/bird_data/'+str(i)+'/' for i in range(50,61,1)]

partition = get_partition(save_directories, split=1.0)
loader, _ = get_data_loaders(partition, shuffle=(False,False), time_shift=(False, False))
model = DLGM(network_dims, load_dir='data/models/sam/')

make_step_size_plot(loader, model)
quit()
"""

"""
# 1) Tune segmenting parameters.
load_dirs = ['data/raw/marmosets/S'+str(i)+'/' for i in range(1,2,1)]
from preprocessing.preprocessing import tune_segmenting_params
seg_params = tune_segmenting_params(load_dirs, preprocess_params)
preprocess_params['seg_params'] = seg_params

quit()
"""

"""
# 2) Segment audio into syllables.
from preprocessing.preprocessing import process_sylls
# load_directories = ['data/raw/bird_data/'+str(i)+'/' for i in range(52,61,1)] # TEMP
# save_directories = ['data/processed/bird_data/'+str(i)+'/' for i in range(52,61,1)]
load_directories = ['data/raw/marmosets/S'+str(i)+'/' for i in [1,2,3,4,5]]
save_directories = ['data/processed/hage/S'+str(i)+'/' for i in [1,2,3,4,5]]
for load_dir, save_dir in zip(load_directories, save_directories):
	try:
		preprocess_params['meta'] = np.load(load_dir + 'meta.npy').item()
	except FileNotFoundError:
		pass
	process_sylls(load_dir, save_dir, preprocess_params)
quit()
"""

save_directories = ['data/processed/hage/S'+str(i)+'/' for i in [1,2,3,4,5]]
"""
# 3) Train a generative model on these syllables.
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
partition = get_partition(save_directories, split=0.80)
model = DLGM(network_dims, partition=partition, save_dir='data/models/hage/', sylls_per_file=preprocess_params['sylls_per_file'])
model.train(epochs=100)
quit()
"""
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
model = DLGM(network_dims, load_dir='data/models/hage/', sylls_per_file=preprocess_params['sylls_per_file'])

# 4) Use the model to get a latent representation of these syllables.
# save_directories = ['data/processed/bird_data/'+str(i)+'/' for i in range(59,61,1)]

partition = get_partition(save_directories, split=1.0)
loader, _ = get_data_loaders(partition, shuffle=(False,False), time_shift=(False, False), sylls_per_file=preprocess_params['sylls_per_file'])
# latent = model.get_latent(loader, n=10000)
# np.save('latent.npy', latent)



# 5) Visualize these latent representations.
from plotting.longitudinal_gif import generate_syllables
generate_syllables(loader, model)
quit()


# 6) Generate novel audio.
from generate.inversion import invert_spec
interp = np.linspace(latent[0], latent[1], 10) # interpolate between two syllables.
specs = [model.decode(point) for point in interp]
audio = np.concatenate([invert_spec(i) for i in specs])
from scipy.io import wavfile
wavfile.write('interpolation.wav', params['fs'], audio)


if __name__ == '__main__':
	pass
