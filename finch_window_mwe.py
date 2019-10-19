"""
Minimal working example for time-warped birdsong.

0) Define directories and parameters.
1) Tune preprocessing parameters.
2) Warp song renditions & train a generative model.
3) Plot and analyze.

"""

from itertools import repeat
from joblib import Parallel, delayed
import numpy as np
import os

from ava.data.data_container import DataContainer
from ava.models.vae import X_SHAPE
from ava.models.vae import VAE
from ava.models.window_vae_dataset import get_window_partition, \
	get_fixed_window_data_loaders
from ava.preprocessing.preprocess import tune_window_preprocessing_params
from ava.preprocessing.utils import get_spec


#########################################
# 0) Define directories and parameters. #
#########################################
zebra_finch_params = {
	'fs': 32000,
	'get_spec': get_spec,
	'num_freq_bins': X_SHAPE[0],
	'num_time_bins': X_SHAPE[1],
	'nperseg': 512, # FFT
	'noverlap': 256, # FFT
	'max_dur': 1e9, # Big number
	'window_length': 0.12,
	'min_freq': 400,
	'max_freq': 10e3,
	'spec_min_val': 2.0,
	'spec_max_val': 6.5,
	'mel': True, # Frequency spacing
	'time_stretch': False,
	'within_syll_normalize': False,
	'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
			'spec_max_val'),
	'int_preprocess_params': tuple([]),
	'binary_preprocess_params': ('mel', 'within_syll_normalize'),
}
root = '/path/to/directory/'
params = zebra_finch_params
audio_dirs = [os.path.join(root, 'audio')]
roi_dirs = [os.path.join(root, 'segs')]
spec_dirs = [os.path.join(root, 'h5s')]
proj_dirs = [os.path.join(root, 'proj')]
model_filename = os.path.join(root, 'checkpoint_050.tar')
plots_dir = root


dc = DataContainer(projection_dirs=proj_dirs, audio_dirs=audio_dirs, \
	spec_dirs=spec_dirs, plots_dir=root, model_filename=model_filename)


#####################################
# 1) Tune preprocessing parameters. #
#####################################
params = tune_window_preprocessing_params(audio_dirs, params)


###################################################
# 2) Train a generative model on these syllables. #
###################################################
partition = get_window_partition(audio_dirs, roi_dirs, 1)
partition['test'] = partition['train']
num_workers = min(7, os.cpu_count()-1)
loaders = get_fixed_window_data_loaders(partition, params, \
	num_workers=num_workers, batch_size=128)
loaders['test'] = loaders['train']
model = VAE(save_dir=root)
model.train_loop(loaders, epochs=101, test_freq=None)


########################
# 3) Plot and analyze. #
########################
from ava.plotting.tooltip_plot import tooltip_plot_DC
from ava.plotting.latent_projection import latent_projection_plot_DC
from ava.plotting.trace_plot import warped_trace_plot_DC

loaders['test'].dataset.write_hdf5_files(spec_dirs[0], num_files=1000)
latent_projection_plot_DC(dc, alpha=0.25, s=0.5)
tooltip_plot_DC(dc, num_imgs=2000)




if __name__ == '__main__':
	pass


###
