"""
Minimal working example.

0) Define directories and parameters.
1) Tune segmenting parameters.
2) Segment.
	2.5) Clean segmenting decisions.
3) Tune preprocessing parameters.
4) Preprocess.
5) Train a generative model on these syllables.
6) Plot and analyze.

"""

from itertools import repeat
from joblib import Parallel, delayed
import numpy as np
import os

from ava.data.data_container import DataContainer
from ava.models.vae import X_SHAPE
from ava.models.vae import VAE
from ava.models.vae_dataset import get_syllable_partition, \
	get_syllable_data_loaders
from ava.preprocessing.preprocess import process_sylls, \
	tune_syll_preprocessing_params
from ava.preprocessing.utils import get_spec
from ava.segmenting.refine_segments import refine_segments_pre_vae
from ava.segmenting.segment import tune_segmenting_params, segment
from ava.segmenting.amplitude_segmentation_v2 import get_onsets_offsets


#########################################
# 0) Define directories and parameters. #
#########################################
mouse_params = {
	'segment': {
		'max_dur': 0.2,
		'min_freq': 30e3,
		'max_freq': 110e3,
		'nperseg': 1024, # FFT
		'noverlap': 512, # FFT
		'spec_min_val': 2.0,
		'spec_max_val': 6.0,
		'fs': 250000,
		'th_1':1.5,
		'th_2':2.0,
		'th_3':2.5,
		'min_dur':0.03,
		'max_dur':0.2,
		'freq_smoothing': 3.0,
		'smoothing_timescale': 0.007,
		'softmax': False,
		'temperature':0.5,
		'algorithm': get_onsets_offsets,
		'window_dir': 0.6,
	},
	'preprocess': {
		'get_spec': get_spec,
		'max_dur': 0.2, # min duration?
		'min_freq': 30e3,
		'max_freq': 110e3,
		'num_freq_bins': X_SHAPE[0],
		'num_time_bins': X_SHAPE[1],
		'nperseg': 1024, # FFT
		'noverlap': 512, # FFT
		'spec_min_val': 2.0,
		'spec_max_val': 6.0,
		'fs': 250000,
		'mel': False, # Frequency spacing
		'MAD': True,
		'time_stretch': True,
		'within_syll_normalize': False,
		# 'seg_extension': '.txt',
		# 'delimiter': '\t',
		# 'skiprows': 0,
		# 'usecols': (1,2),
		'max_num_syllables': None, # per directory
		'sylls_per_file': 20,
		'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
				'spec_max_val', 'max_dur'),
		'int_preprocess_params': ('nperseg','noverlap'),
		'binary_preprocess_params': ('time_stretch', 'mel', \
				'within_syll_normalize', 'MAD'),
	},
}

params = mouse_params
root = '/media/jackg/Jacks_Animal_Sounds/mice/MUPET/'
model_filename = root + 'checkpoint_010.tar'
audio_dirs = 	[root+'C57', 		root+'DBA']
seg_dirs = 		[root+'C57_segs', 	root+'DBA_segs']
proj_dirs = 	[root+'C57_proj', 	root+'DBA_proj']
spec_dirs = 	[root+'C57_specs', 	root+'DBA_specs']
feature_dirs = 	[root+'C57_mupet',	root+'DBA_mupet']

dc = DataContainer(projection_dirs=proj_dirs, feature_dirs=feature_dirs,
	spec_dirs=spec_dirs, plots_dir=root, model_filename=model_filename)


# ##################################
# # 1) Tune segmenting parameters. #
# ##################################
# params['segment'] = tune_segmenting_params(audio_dirs, params['segment'])


# ###############
# # 2) Segment. #
# ###############
# n_jobs = min(len(audio_dirs), os.cpu_count()-1)
# gen = zip(audio_dirs, seg_dirs, repeat(params['segment']))
# Parallel(n_jobs=n_jobs)(delayed(segment)(*args) for args in gen)



# ####################################
# # 2.5) Clean segmenting decisions. #
# ####################################
# refine_segments_pre_vae(seg_dirs, audio_dirs, new_seg_dirs, params['segment'])



# #####################################
# # 3) Tune preprocessing parameters. #
# #####################################
# preprocess_params = tune_syll_preprocessing_params(audio_dirs, seg_dirs, \
# 		params['preprocess'])
# params['preprocess'] = preprocess_params


# ##################
# # 4) Preprocess. #
# ##################
# n_jobs = min(3, os.cpu_count()-1)
# gen = zip(audio_dirs, seg_dirs, spec_dirs, repeat(params['preprocess']))
# Parallel(n_jobs=n_jobs)(delayed(process_sylls)(*args) for args in gen)


###################################################
# 5) Train a generative model on these syllables. #
###################################################
model = VAE(save_dir=root)
model.load_state(root+'checkpoint_080.tar')
partition = get_syllable_partition(spec_dirs, split=1)
num_workers = min(7, os.cpu_count()-1)
loaders = get_syllable_data_loaders(partition, num_workers=num_workers)
loaders['test'] = loaders['train']
model.train_loop(loaders, epochs=201, test_freq=None)


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
