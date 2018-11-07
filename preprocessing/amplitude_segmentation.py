from __future__ import print_function, division
"""
Amplitude based syllable segmentation.

TO DO: Add lambda filtering
TO DO: kwargs
TO DO: visualize
TO DO: change thresholds to interpretable values?
"""

import matplotlib.pyplot as plt # TEMP!
plt.switch_backend('agg')

import numpy as np
import torch
import soundfile as sf
from os import listdir
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.signal import convolve2d, stft, medfilt
from scipy.stats import linregress
from scipy.interpolate import interp1d


# STFT params
fs = 44100
nperseg, noverlap = 256, 128
epsilon = 1e-12


# thresholds for syllable onset & offset
th_1, th_2, th_3, th_4 = 0.25, 0.00, 0.0, 0.3
min_var = 1.2
max_syll_len = 192
min_syll_len = 2


def get_onsets_offsets(spec, **kwargs):
    """
    Segment the spectrogram using hard thresholds on its ampltiude & derivative.

    TO DO: pass filtering functions
    """
    # Get amplitude data and its derivative.
    amps = np.mean(spec, axis=0)
    smooth = gaussian_filter(spec, [2,2])
    filter = np.array([[1,0,-1]]) # Sobel filter
    t_grads = convolve2d(smooth, filter, mode='same', boundary='symm')
    t_grads = np.mean(t_grads, axis=0)
    t_grads = gaussian_filter1d(t_grads, 3.0)

    # Normalize.
    amps -= np.percentile(amps, 2)
    amps /= np.percentile(amps, 98)
    amps = gaussian_filter1d(amps, 2.0)
    t_grads /= np.percentile(np.abs(t_grads), 98)

    # Collect syllable times using hard thresholds for detecting onsets and
    # offsets.
    onsets = []
    offsets = []
    last = 'off'
    for i in range(len(amps)):
        if last == 'off':
            if t_grads[i] > th_1 and amps[i] > th_2:
                onsets.append(i)
                last = 'on'
        else:

            if t_grads[i] < th_3 and amps[i] < th_4:
                # NOTE: HERE
                var = np.mean(np.var(spec[:,onsets[-1]:i], axis=0))
                if var < min_var:
                    onsets = onsets[:-1]
                    discarded += 1
                else:
                    offsets.append(i)
                last = 'off'
            elif i - onsets[-1] >= max_syll_len:
                last = 'off'
                onsets = onsets[:-1]
    onsets = onsets[:len(offsets)] # We may have picked up an unmatched onset.

    # Throw away syllables that are too long or too short.
    new_onsets = []
    new_offsets = []
    for i in range(len(off_segs)):
        t1, t2 = on_segs[i], off_segs[i]
        if t2 - t1 + 1 <= max_syll_len and t2 - t1 + 1 >= min_syll_len:
            new_onsets.append(t1)
            new_offsets.append(t2)
    return new_onsets, new_offsets



def get_syll_specs(onsets, offsets, spec, log_spacing=True):
    """
    Return an array of spectrograms, one for each syllable.

    Arguments
    ---------
        - onsets: ...
    """
    specs = np.zeros((len(onsets),) + spec_shape ), dtype='float')
    durations = np.zeros(len(onsets), dtype='int')
    start_times = np.zeros(len(onsets), dtype='float')
    log_f = np.geomspace(f[0], f[-1], num=i2-i1, endpoint=True)
    log_f[-1] = f[-1] # Correct for numerical errors.
    log_f[0] = f[0]
    # For each syllable...
    for t1, t2 in zip(onsets, offsets):
        # Take a slice of the spectrogram.
        temp_spec = spec[:,t1:t2+1]

        # Apply a logarithmmic frequency interpolation.
        if log_spacing:
            log_f_spec = np.zeros_like(temp_spec)
            for j in range(temp_spec.shape[1]):
                interp = interp1d(log_f, temp_spec[:,j], kind='cubic')
                log_f_spec[:,j] = interp(f)
            temp_spec = log_f_spec

        # Normalize.
        temp_spec -= np.min(temp_spec)
        temp_spec /= np.max(temp_spec)

        # Collect spectrogram and duration.
        duration = temp_spec.shape[1]
        durations.append(duration)
        specs.append(temp_spec)

    return specs, durations




###
