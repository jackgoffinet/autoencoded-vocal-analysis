"""
Minimal working example for generative modeling of acoustic syllables.

1) Tune segmenting parameters.
2) Tune preprocessing parameters.
3) Segment audio into syllables.
4) Train a generative model on these syllables.
5) Plot and analyze.

Notes
-----


"""
__author__ = "Jack Goffinet"
__date__ = "December 2018 - August 2019"

import numpy as np
import os

from models.vae import X_SHAPE
from preprocessing.preprocessing import get_spec


mouse_params = {
	'sliding_window': False,
	'get_spec': get_spec,
	'num_freq_bins': X_SHAPE[0],
	'num_time_bins': X_SHAPE[1],
	'min_freq': 30e3,
	'max_freq': 110e3,
	'nperseg': 1024, # FFT
	'noverlap': 0, # FFT
	'mel': False, # Frequency spacing
	'spec_min_val': -6,
	'spec_max_val': -3,
	'time_stretch': True,
	'within_syll_normalize': False,
	'MAD': True,
	'seg_extension': '.csv',
	'delimiter': ',',
	'skiprows': 1,
	'usecols': (1,2),
	'max_dur': 0.2,
	'max_num_syllables': None, # per directory
	'sylls_per_file': 20,
	'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
			'spec_max_val', 'max_dur'),
	'int_preprocess_params': ('nperseg',),
	'binary_preprocess_params': ('time_stretch', 'mel', 'within_syll_normalize', 'MAD')
}



zebra_finch_params_sliding_window = {
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
	'seg_extension': '.txt',
	'delimiter': '\t',
	'skiprows': 0,
	'usecols': (0,1),
	'max_dur': 0.08,
	'max_num_syllables': None, # per directory
	# 'sylls_per_file': 20,
	'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
			'spec_max_val', 'max_dur'),
	'int_preprocess_params': ('nperseg',),
	'binary_preprocess_params': ('time_stretch', 'mel', 'within_syll_normalize')
}

zebra_finch_params = {
	'sliding_window': False,
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
	'time_stretch': True,
	'within_syll_normalize': False,
	'seg_extension': '.txt',
	'delimiter': ' ',
	'skiprows': 0,
	'usecols': (0,1),
	'max_dur': 0.2,
	'max_num_syllables': None, # per directory
	'sylls_per_file': 5,
	'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
			'spec_max_val', 'max_dur'),
	'int_preprocess_params': ('nperseg',),
	'binary_preprocess_params': ('time_stretch', 'mel', 'within_syll_normalize')
}


# Set which set of parameters to use.
preprocess_params = zebra_finch_params_sliding_window
root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/'
audio_dirs = [os.path.join(root, i) for i in ['songs/DIR', 'songs/UNDIR']]
template_dir = root + 'templates'
spec_dirs = [root+'h5s']
proj_dirs = [root+'song_window/proj/']
model_filename = root + 'song_window/checkpoint_201.tar'
plots_dir = root + 'song_window/plots/'
feature_dirs = None
seg_dirs = None
# hdf5_dirs = [os.path.join(root, i) for i in ['DIR_SAP_h5', 'UNDIR_SAP_h5']]
# plots_dir = root + 'song_window/plots'
# proj_dirs = [root + 'syll_proj_DIR', root + 'syll_proj_UNDIR']
# feature_dirs = [root + i for i in ['DIR_SAP_features', 'UNDIR_SAP_features']]
# model_filename = root + 'checkpoint_080.tar'


# preprocess_params = mouse_params
# root = '/media/jackg/Jacks_Animal_Sounds/mice/Tom_control/'
# nums = [0] + list(range(2,31)) + list(range(44,57))
# audio_dirs = [root+'BM'+str(i).zfill(3)+'/audio/' for i in nums]
# seg_dirs = [root+'BM'+str(i).zfill(3)+'/mupet/' for i in nums]
# save_dirs = [root+'BM'+str(i).zfill(3)+'/hdf5s/' for i in nums]



"""
# 1) Tune segmenting parameters.
from os import listdir
from preprocessing.preprocessing import tune_segmenting_params
seg_params = tune_segmenting_params(load_dirs, preprocess_params)
preprocess_params['seg_params'] = seg_params
quit()
"""

"""
# 2) Tune preprocessing parameters.
from preprocessing.preprocessing import tune_preprocessing_params
preprocess_params = tune_preprocessing_params(audio_dirs, seg_dirs, preprocess_params)
quit()
"""

"""
# 3) Segment audio into syllables.
import os
from preprocessing.preprocessing import process_sylls
from multiprocessing import Pool
from itertools import repeat
with Pool(3) as pool: # min(3, os.cpu_count()-1)
	pool.starmap(process_sylls, zip(audio_dirs, seg_dirs, hdf5_dirs, repeat(preprocess_params)))
quit()
"""

"""
# 4) Train a generative model on these syllables.
# from models.vae import VAE
from models.window_vae import VAE
from models.vae_dataset import get_warped_window_data_loaders
model = VAE(save_dir=root+'song_window')
# model.load_state(root + 'syll_checkpoint_080.tar')
# partition = get_syllable_partition(hdf5_dirs, split=1)
loaders = get_warped_window_data_loaders(audio_dirs, template_dir, \
	preprocess_params, num_workers=4)
loaders['train'].dataset.write_hdf5_files(root+'h5s/')
quit()
# loaders['test'] = loaders['train']
model.train_loop(loaders, epochs=201, test_freq=1000)
quit()
"""

from plotting.data_container import DataContainer
from plotting.trace_plot import trace_plot_DC, inst_variability_plot_DC
from plotting.tooltip_plot import tooltip_plot_DC
from plotting.mmd_plots import mmd_matrix_DC, make_g_2
from plotting.latent_projection import latent_projection_plot_DC
from plotting.feature_correlation_plots import  correlation_plot_DC, \
	knn_variance_explained_plot_DC, pairwise_correlation_plot_DC, feature_pca_plot_DC
from plotting.pairwise_distance_plots import pairwise_distance_scatter_DC, \
	knn_display_DC, bridge_plot, random_walk_plot, indexed_grid_plot, \
	plot_paths_on_projection
from plotting.cluster_pca_plot import cluster_pca_plot_DC, cluster_pca_feature_plot_DC


fields = ['syllable_duration_sap', 'mean_amplitude',
	'mean_pitch', 'mean_FM', 'mean_AM2', 'mean_entropy', 'mean_pitch_goodness',
	'mean_mean_freq', 'pitch_variance', 'FM_variance', 'entropy_variance',
	'pitch_goodness_variance', 'mean_freq_variance', 'AM_variance']

dc = DataContainer(projection_dirs=proj_dirs, \
	spec_dirs=spec_dirs, plots_dir=plots_dir, model_filename=model_filename,
	feature_dirs=feature_dirs)


latent_projection_plot_DC(dc, filename='latent.png')
quit()
# [3,10,-15,-5]
# [5,15,-2,5]
cluster_pca_plot_DC(dc, [5,15,-2,5], filename='cluster_pca_1.pdf')
quit()
cluster_pca_feature_plot_DC(dc, [3,10,-15,-5], fields)

quit()



if __name__ == '__main__':
	pass


###
