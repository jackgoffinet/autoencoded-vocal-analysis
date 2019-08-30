"""
Minimal working example for time-warped birdsong.

0) Define directories and parameters.
1) Tune preprocessing parameters.
2) Warp song renditions.
3) Train a generative model.
4) Plot and analyze.

"""
__author__ = "Jack Goffinet"
__date__ = "December 2018 - August 2019"


from itertools import repeat
from joblib import Parallel, delayed
import numpy as np
import os

from ava.data.data_container import DataContainer
from ava.models.vae import X_SHAPE
from ava.models.vae import VAE
# from ava.models.window_vae import VAE as WindowVAE
from ava.models.window_vae_dataset import get_warped_window_data_loaders
from ava.preprocessing.preprocessing import get_spec, process_sylls, \
	tune_preprocessing_params


#########################################
# 0) Define directories and parameters. #
#########################################
zebra_finch_params_warped_window = {
	'preprocess': {
		'sliding_window': True,
		'window_length': 0.08,
		'get_spec': get_spec,
		'num_freq_bins': X_SHAPE[0],
		'num_time_bins': X_SHAPE[1],
		'min_freq': 400,
		'max_freq': 10e3,
		'nperseg': 512, # FFT
		'noverlap': 256, # FFT
		'mel': True, # Frequency spacing
		'spec_min_val': 2.0,
		'spec_max_val': 6.5,
		'time_stretch': False,
		'within_syll_normalize': False,
		'max_dur': 1e9, # Big number
	},

	'seg_extension': '.txt',
	'delimiter': '\t',
	'skiprows': 0,
	'usecols': (0,1),

	'max_num_syllables': None, # per directory
	# 'sylls_per_file': 20,
	'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
			'spec_max_val', 'max_dur'),
	'int_preprocess_params': ('nperseg',),
	'binary_preprocess_params': ('time_stretch', 'mel', 'within_syll_normalize')
}


root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/'
params = zebra_finch_params_warped_window
audio_dirs = [os.path.join(root, i) for i in ['songs/DIR', 'songs/UNDIR']]
template_dir = root + 'templates'
spec_dirs = [root+'h5s']
proj_dirs = [root+'song_window/proj/']
model_filename = root + 'song_window/checkpoint_201.tar'
plots_dir = root + 'song_window/plots/'


dc = DataContainer(projection_dirs=proj_dirs, \
	spec_dirs=spec_dirs, plots_dir=root, model_filename=model_filename)


#####################################
# 3) Tune preprocessing parameters. #
#####################################
preprocess_params = tune_preprocessing_params(audio_dirs, None, \
		params['preprocess'])
params['preprocess'] = preprocess_params
quit()



###################################################
# 5) Train a generative model on these syllables. #
###################################################
model = VAE(save_dir=root)
num_workers = min(7, os.cpu_count()-1)
loaders = get_warped_window_data_loaders(partition, num_workers=num_workers)
loaders['test'] = loaders['train']
model.train_loop(loaders, epochs=201, test_freq=None)
quit()


########################
# 6) Plot and analyze. #
########################
from ava.plotting.tooltip_plot import tooltip_plot_DC
from ava.plotting.latent_projection import latent_projection_plot_DC

latent_projection_plot_DC(dc)
tooltip_plot_DC(dc, num_imgs=2000)




if __name__ == '__main__':
	pass


###
