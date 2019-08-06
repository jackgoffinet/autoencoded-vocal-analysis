"""
Interface between the syllable_modeling package and the Dash GUI.

"""
__author__ = "Jack Goffinet"
__date__ = "July 2019"

import os
import base64

from plotting.data_container import DataContainer
from preprocessing.preprocessing import get_spec
from models.vae import X_SHAPE



segment_img_fn = os.path.join('assets', 'segment.png')
preprocess_img_fn = os.path.join('assets', 'preprocess.png')
background_img_fn = os.path.join('assets', 'zebra_finch.png')



class GUIState():
	"""
	Object for maintaining the GUI state.

	Mostly a wrapper around plotting.data_container.DataContainer, but with
	restricted functionality.
	"""

	def __init__(self, audio_dirs=None, spec_dirs=None, feature_dirs=None, \
		projection_dirs=None, plots_dir='', model_dir=None):
		"""
		Parameters
		----------

		"""
		self.dc = DataContainer(
			audio_dirs=audio_dirs,
			spec_dirs=spec_dirs,
			feature_dirs=feature_dirs,
			projection_dirs=projection_dirs,
			plots_dir=plots_dir,
			model_filename=None,
		)
		self.model_dir = model_dir
		self.p = {
			'sliding_window': True,
			'window_length': 0.15,
			'get_spec': get_spec,
			'num_freq_bins': X_SHAPE[0],
			'num_time_bins': X_SHAPE[1],
			'min_freq': 400,
			'max_freq': 8e3,
			'nperseg': 512, # FFT
			'noverlap': 512-128-64, # FFT
			'mel': True, # Frequency spacing
			'freq_shift': 0.0, # Frequency shift
			'spec_min_val': 7.0,
			'spec_max_val': 12.0,
			'time_stretch': False,
			'within_syll_normalize': False,
			'seg_extension': '.txt',
			'delimiter': '\t',
			'skiprows': 0,
			'usecols': (0,1),
			'max_dur': 0.15,
			'max_num_syllables': None, # per directory
			# 'sylls_per_file': 20,
			'real_preprocess_params': ('min_freq', 'max_freq', 'spec_min_val', \
					'spec_max_val', 'max_dur'),
			'int_preprocess_params': ('nperseg',),
			'binary_preprocess_params': ('sliding_window', 'time_stretch', 'mel', 'within_syll_normalize')
		}


	def get_background(self):
		"""Get background image."""
		return base64.b64encode(open(background_img_fn, 'rb').read()).decode()


	def get_possible_saved_models(self):
		"""Get all the models saved in self."""




if __name__ == '__main__':
	pass


###
