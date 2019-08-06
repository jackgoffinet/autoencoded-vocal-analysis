"""
Minimal working example for generative modeling of acoustic syllables.

1) Tune segmenting parameters.
2) Segment audio into syllables.
3) Train a generative model on these syllables.
4) Use the model to get a latent representation of these syllables.
5) Visualize these latent representations.

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
	'freq_shift': 0.0, # Frequency shift
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



zebra_finch_params = {
	'sliding_window': True,
	'window_length': 0.1,
	'get_spec': get_spec,
	'num_freq_bins': X_SHAPE[0],
	'num_time_bins': X_SHAPE[1],
	'min_freq': 400,
	'max_freq': 10e3,
	'nperseg': 512, # FFT
	'noverlap': 256, # FFT
	'mel': True, # Frequency spacing
	'freq_shift': 0.0, # Frequency shift
	'spec_min_val': 2.0,
	'spec_max_val': 6.5,
	'time_stretch': False,
	'within_syll_normalize': False,
	'seg_extension': '.txt',
	'delimiter': '\t',
	'skiprows': 0,
	'usecols': (0,1),
	'max_dur': 0.1,
	'max_num_syllables': None, # per directory
	# 'sylls_per_file': 20,
	'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
			'spec_max_val', 'max_dur'),
	'int_preprocess_params': ('nperseg',),
	'binary_preprocess_params': ('time_stretch', 'mel', 'within_syll_normalize')
}




# # Set which set of parameters to use.
# preprocess_params = zebra_finch_params
# root = '/media/jackg/Jacks_Animal_Sounds/birds/jonna/blu285/'
# audio_dirs = [root + i +'/' for i in ['DIR', 'OPTO', 'UNDIR']]
# seg_dirs = audio_dirs

preprocess_params = mouse_params
root = '/media/jackg/Jacks_Animal_Sounds/mice/Tom_control/'
nums = [15]
audio_dirs = [root+'BM'+str(i).zfill(3)+'/audio/' for i in nums]
seg_dirs = [root+'BM'+str(i).zfill(3)+'/mupet/' for i in nums]
save_dirs = [root+'BM'+str(i).zfill(3)+'/hdf5s/' for i in nums]


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
with Pool(1) as pool: # min(3, os.cpu_count()-1)
	pool.starmap(process_sylls, zip(audio_dirs, seg_dirs, save_dirs, repeat(preprocess_params)))
quit()
"""

"""
# 4) Train a generative model on these syllables.
from models.window_vae import VAE
from models.vae_dataset import get_syllable_partition, get_syllable_data_loaders
VAE = VAE(save_dir=root)
partition = get_syllable_partition(audio_dirs, seg_dirs, split=0.9)
loaders = get_syllable_data_loaders(partition, preprocess_params)
print(len(loader['test'].dataset))
quit()
VAE.train_loop(loaders, epochs=201)
# quit()
"""

from plotting.data_container import DataContainer
from plotting.trace_plot import trace_plot_DC, inst_variability_plot_DC
from plotting.tooltip_plot import tooltip_plot_DC
from plotting.mmd_plots import mmd_matrix_DC
from plotting.latent_projection import latent_projection_plot_DC
from plotting.feature_correlation_plots import  correlation_plot_DC, \
	knn_variance_explained_plot_DC, pairwise_correlation_plot_DC, feature_pca_plot_DC
from plotting.pairwise_distance_plots import pairwise_distance_scatter_DC, \
	knn_display_DC, bridge_plot, random_walk_plot, indexed_grid_plot, \
	plot_paths_on_projection

root = '/media/jackg/Jacks_Animal_Sounds/mice/MUPET/'
proj_dirs = [root+'C57_projections', root+'DBA_projections']
feature_dirs = [root+'C57_MUPET_detect', root+'DBA_MUPET_detect']
spec_dirs = [root+'C57_hdf5s', root+'DBA_hdf5s']

dc = DataContainer(projection_dirs=proj_dirs, feature_dirs=feature_dirs, \
	spec_dirs=spec_dirs, plots_dir=root+'plots')
tooltip_plot_DC(dc)

quit()


# 5) Use the model to get a latent representation of these syllables.
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
model = DLGM(network_dims, load_dir='data/models/', sylls_per_file=preprocess_params['sylls_per_file'])
partition = get_partition(save_dirs, split=1.0)
loaders = get_data_loaders(partition, shuffle=(False,False), batch_size=32, sylls_per_file=preprocess_params['sylls_per_file'])

d = {'model':model, 'loader':loader}

from plotting.longitudinal_gif import make_projection, make_dot_gif, \
		make_html_plot, plot_random_interp


title = ""
n = 3*10**4

# d = np.load('d.npy').item()

# d = plot_random_interp(d, i1=14, i2=15)
# quit()

# print("making projection")
d = make_projection(d, title=title, n=n, axis=False)
quit()

# print("making gif")
# d = make_dot_gif(d, title=title, n=n)

# print("making html")
# make_html_plot(d, output_dir='temp/', n=n, num_imgs=2000, title=title)
#
# np.save('d.npy', d)
# quit()

print("Saving everything...")
# Save a bunch of data.
from scipy.io import savemat

from plotting.longitudinal_gif import update_data
keys = ['latent', 'file_time', 'time', 'filename', 'duration', 'embedding']
d = update_data(d, keys, n=n) # + ['image']


# savemat('images.mat', {'images':d['image']})
# del d['image']

del d['model']
del d['loader']
savemat('data.mat', d)
quit()

import joblib
reducer = joblib.load('temp_reducer.sav')
from save_stuff import save_everything

# Parallel loop
import os
from multiprocessing import Pool
from itertools import repeat
with Pool(min(3, os.cpu_count()-1)) as pool:
	pool.starmap(save_everything, zip(repeat(model), load_dirs, save_dirs, repeat(preprocess_params), repeat(reducer)))
quit()

# # Old serial loop
# for i in range(len(load_dirs)):
# 	save_everything(model, load_dirs[i], save_dirs[i], preprocess_params, reducer)
# quit()


if __name__ == '__main__':
	pass


###
