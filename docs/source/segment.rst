Segmenting
==========


Importing syllable segments
###########################

AVA reads syllable segments from :code:`.txt` files with two tab-separated
columns containing onsets and offsets for each syllable. Lines beginning with
:code:`#` are ignored.

.. note:: This is the default format for Numpy's :code:`loadtxt`
	function. It also happens to be Audacity's label format.

AVA provides a function for copying onsets and offsets from other formats to the
standard format:

.. code:: Python3

	from ava.segmenting.utils import copy_segments_to_standard_format
	help(copy_segments_to_standard_format)



Syllable segmenting in AVA
##########################

You can also use AVA's built-in segmenting functions. Here, we'll go through the
amplitude segmentation method. First, import the
segmenting function and set a bunch of segmenting parameters:

.. code:: Python3

	from ava.segmenting.amplitude_segmentation import get_onsets_offsets

	seg_params = {
	    'min_freq': 30e3, # minimum frequency
	    'max_freq': 110e3, # maximum frequency
	    'nperseg': 1024, # FFT
	    'noverlap': 512, # FFT
	    'spec_min_val': 2.0, # minimum log-spectrogram value
	    'spec_max_val': 6.0, # maximum log-spectrogram value
	    'fs': 250000, # audio samplerate
	    'th_1':1.5, # segmenting threshold 1
	    'th_2':2.0, # segmenting threshold 2
	    'th_3':2.5, # segmenting threshold 3
	    'min_dur':0.03, # minimum syllable duration
	    'max_dur': 0.2, # maximum syllable duration
	    'smoothing_timescale': 0.007, # amplitude
	    'softmax': False, # apply softmax to the frequency bins to calculate
	                      # amplitude
	    'temperature':0.5, # softmax temperature parameter
	    'algorithm': get_onsets_offsets, # (defined above)
	}

.. note:: AVA only reads audio files in :code:`.wav` format!

Then we can tune these parameter values by visualizing segmenting decisions:

.. code:: Python3

	from ava.segmenting.segment import tune_segmenting_params
	audio_directories = [...] # list of audio directories
	seg_params = tune_segmenting_params(audio_directories, seg_params)



This will start an interactive tuning process, where parameters can be adjusted
and the resulting segmenting decisions will be displayed in a saved image, by
default :code:`temp.pdf`. The three thresholds will be displayed with an
amplitude trace, detected onsets and offsets, and a spectrogram.

From :code:`ava.segmenting.amplitude_segmentation.get_onsets_offsets`:

	A syllable is detected if the amplitude trace exceeds ``p['th_3']``. An offset
	is then detected if there is a subsequent local minimum in the amplitude
	trace with amplitude less than ``p['th_2']``, or when the amplitude drops
	below ``p['th_1']``, whichever comes first. Syllable onset is determined
	analogously.

Once we're happy with a particular set of parameters, we can go through a whole
collection of audio files and write segmenting decisions in corresponding
directories. It's useful to have a 1-to-1 correspondence between audio
directories and segmenting directories.

.. code:: Python3

	from ava.segmenting.segment import segment
	audio_dirs = ['path/to/animal1/audio/', 'path/to/animal2/audio/']
	segment_dirs = ['path/to/animal1/segments/', 'path/to/animal2/segments/']
	for audio_dir, segment_dir in zip(audio_dirs, segment_dirs):
		segment(audio_dir, segment_dir, seg_params)



Or, parallelized this time:

.. code:: Python3

	from joblib import Parallel, delayed
	from itertools import repeat
	gen = zip(audio_dirs, segment_dirs, repeat(seg_params))
	Parallel(n_jobs=4)(delayed(segment)(*args) for args in gen)



Song segmenting in AVA
######################

TO DO

Syllable segmenting from song segments in AVA
#############################################

TO DO
