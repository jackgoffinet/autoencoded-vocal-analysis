"""
Minimal working example.

0) Define directories and parameters.
1) Tune segmenting parameters.
2) Segment.
3) Tune preprocessing parameters.
4) Preprocess.
5) Train a generative model on these syllables.
6) Plot and analyze.


"""
__author__ = "Jack Goffinet"
__date__ = "December 2018 - August 2019"


from itertools import repeat
from joblib import Parallel, delayed
import numpy as np
import os

from ava.models.vae import X_SHAPE
from ava.models.vae import VAE
from ava.models.window_vae import VAE as WindowVAE
from ava.models.vae_dataset import get_warped_window_data_loaders
from ava.preprocessing.preprocessing import get_spec, process_sylls, \
	tune_preprocessing_params
from ava.segmenting.segmenting import tune_segmenting_params
from ava.segmenting.amplitude_segmentation_v2 import get_onsets_offsets



# 0) Define directories and parameters.

mouse_params = {
	'sliding_window': False,
	'get_spec': get_spec,
	'num_freq_bins': X_SHAPE[0],
	'num_time_bins': X_SHAPE[1],
	'MAD': True,
	'seg_extension': '.csv',
	'delimiter': ',',
	'skiprows': 1,
	'usecols': (1,2),
	'seg_params': {
		'max_dur': 0.2,
		'min_freq': 30e3,
		'max_freq': 110e3,
		'nperseg': 1024, # FFT
		'noverlap': 0, # FFT
		'spec_min_val': 0.0,
		'spec_max_val': 11.0,
		'fs': 250000,
		'th_1':0.1,
		'th_2':1.0,
		'th_3':3.0,
		'min_dur':0.03,
		'max_dur':0.2,
		'freq_smoothing': 3.0,
		'smoothing_timescale': 0.005,
		'num_time_bins': 128,
		'num_freq_bins': 128,
		'softmax': False,
		'temperature':0.5,
		'algorithm': get_onsets_offsets,
	},
	'preprocess_params': {
		'max_dur': 0.2,
		'min_freq': 30e3,
		'max_freq': 110e3,
		'nperseg': 1024, # FFT
		'noverlap': 0, # FFT
		'spec_min_val': -6,
		'spec_max_val': -3,
		'fs': 250000,
		'mel': False, # Frequency spacing
		'time_stretch': True,
		'within_syll_normalize': False,
	},
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
	'max_dur': 1e9, # Big number
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

# root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/'
# params = zebra_finch_params_sliding_window
# audio_dirs = [os.path.join(root, i) for i in ['songs/DIR', 'songs/UNDIR']]
# template_dir = root + 'templates'
# spec_dirs = [root+'h5s']
# proj_dirs = [root+'song_window/proj/']
# model_filename = root + 'song_window/checkpoint_201.tar'
# plots_dir = root + 'song_window/plots/'
# feature_dirs = None
# seg_dirs = None

root = '/media/jackg/Jacks_Animal_Sounds/mice/MUPET/'
audio_dirs = [root + i for i in ['C57', 'DBA']]
seg_dirs = [root + i for i in ['temp_seg_C57', 'temp_seg_DBA']]
params = mouse_params


# 1) Tune segmenting parameters.
seg_params = tune_segmenting_params(audio_dirs, params['seg_params'])
params['seg_params'] = seg_params
# quit()



# 2) Segment.
# n_jobs = min(3, os.cpu_count()-1)
# gen = zip(audio_dirs, seg_dirs, spec_dirs, repeat(preprocess_params))
# Parallel(n_jobs=n_jobs)(delayed(process_sylls)(i) for args in gen)

for audio_dir, seg_dir in zip(audio_dirs, seg_dirs):
	segment(audio_dir, seg_dir, params['seg_params'])
quit()


"""
# 3) Tune preprocessing parameters.
preprocess_params = tune_preprocessing_params(audio_dirs, seg_dirs, preprocess_params)
"""

"""
# 4) Preprocess.
n_jobs = min(3, os.cpu_count()-1)
gen = zip(audio_dirs, seg_dirs, spec_dirs, repeat(preprocess_params))
Parallel(n_jobs=n_jobs)(delayed(process_sylls)(i) for args in gen)
# pool.starmap(process_sylls, zip(audio_dirs, seg_dirs, hdf5_dirs, repeat(preprocess_params)))
"""

# 5) Train a generative model on these syllables.
model = VAE(save_dir=root+'song_window')
model.load_state(model_filename)
# partition = get_syllable_partition(hdf5_dirs, split=1)
loaders = get_warped_window_data_loaders(audio_dirs, template_dir, \
	params, num_workers=4, load_warp=True)
# loaders['train'].dataset.write_hdf5_files(root+'h5s/')
# quit()
loaders['test'] = loaders['train']
model.train_loop(loaders, epochs=201, test_freq=1000)
quit()


# 6) Plot and analyze.
from ava.data.data_container import DataContainer
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
cluster_pca_plot_DC(dc, [5,15,-2,5], filename='cluster_pca_1.pdf')
cluster_pca_feature_plot_DC(dc, [3,10,-15,-5], fields)




if __name__ == '__main__':
	pass


###
