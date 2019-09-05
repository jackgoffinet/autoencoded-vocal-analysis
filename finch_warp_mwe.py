"""
Minimal working example for time-warped birdsong.

0) Define directories and parameters.
1) Tune preprocessing parameters.
2) Warp song renditions & train a generative model.
3) Plot and analyze.

TO DO: figure out best way for window to work with DataContainer
"""

from itertools import repeat
from joblib import Parallel, delayed
import numpy as np
import os

from ava.data.data_container import DataContainer
from ava.models.vae import X_SHAPE
from ava.models.vae import VAE
from ava.models.window_vae_dataset import get_warped_window_data_loaders
from ava.preprocessing.preprocess import tune_window_preprocessing_params
from ava.preprocessing.utils import get_spec


#########################################
# 0) Define directories and parameters. #
#########################################
zebra_finch_params_warped_window = {
	'fs': 32000,
	'get_spec': get_spec,
	'num_freq_bins': X_SHAPE[0],
	'num_time_bins': X_SHAPE[1],
	'nperseg': 512, # FFT
	'noverlap': 256, # FFT
	'max_dur': 1e9, # Big number
	'window_length': 0.08,
	'min_freq': 400,
	'max_freq': 10e3,
	'spec_min_val': 2.0,
	'spec_max_val': 6.5,
	'mel': True, # Frequency spacing
	'time_stretch': False,
	'within_syll_normalize': False,
	'n_knots': 3, # For time-warping.
	'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
			'spec_max_val'),
	'int_preprocess_params': tuple([]),
	'binary_preprocess_params': ('mel', 'within_syll_normalize'),
}


root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/'
params = zebra_finch_params_warped_window
audio_dirs = [os.path.join(root, i) for i in ['songs/DIR', 'songs/UNDIR']]
template_dir = root + 'templates'
spec_dirs = [root+'h5s']
proj_dirs = [root+'song_window/proj/']
# model_filename = root + 'song_window/checkpoint_201.tar'
model_filename = root + 'checkpoint_200.tar'
plots_dir = root + 'song_window/plots/'


dc = DataContainer(projection_dirs=proj_dirs, audio_dirs=audio_dirs, \
	spec_dirs=spec_dirs, plots_dir=root, model_filename=model_filename, \
	template_dir=template_dir)


# #####################################
# # 1) Tune preprocessing parameters. #
# #####################################
# params['preprocess'] = tune_window_preprocessing_params(audio_dirs, params)


# ###################################################
# # 2) Train a generative model on these syllables. #
# ###################################################
# num_workers = min(7, os.cpu_count()-1)
# loaders = get_warped_window_data_loaders(audio_dirs, template_dir, params, \
# 		num_workers=num_workers, load_warp=True)
# loaders['test'] = loaders['train']
# model = VAE(save_dir=root)
# model.train_loop(loaders, epochs=201, test_freq=None)


########################
# 3) Plot and analyze. #
########################
from ava.plotting.tooltip_plot import tooltip_plot_DC
from ava.plotting.latent_projection import latent_projection_plot_DC
from ava.plotting.trace_plot import warped_trace_plot_DC

latent_projection_plot_DC(dc)
# tooltip_plot_DC(dc, num_imgs=2000)
warped_trace_plot_DC(dc, params, load_warp=True)




if __name__ == '__main__':
	pass


###
