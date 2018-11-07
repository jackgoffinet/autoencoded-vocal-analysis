from __future__ import print_function, division
"""
Process syllables and save data.

TO DO: denoise?
"""
__author__ = "Jack Goffinet"
__date__ = "November 2018"


from os import listdir
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
import h5py
from tqdm import tqdm

import amplitude_segmentation as amp_seg


EPS = 1e-12
nperseg, noverlap = 256, 128
sylls_per_file = 1000



def process_sylls(load_dir, save_dir, metadata={}, min_freq=300, max_freq=12e3,
                    num_freq_bins=64, min_dur=6e-3, max_dur=None,
                    num_time_bins=128, verbose=True):
    """
    Process files in <load_dir> and save to <save_dir>.

    Notes
    -----
        -   Assumes the .wav files are contiguous recordings with alphabetically
            ordered filenames.


    Arguments
    ---------
    - load_dir :

    - save_dir :

    ...

    """
    if 'load_dir' not in metadata:
        metadata['load_dir'] = load_dir
    filenames = [i for i in listdir(load_dir) if i[-4:] == '.wav']
    filenames = np.sorted(filenames)
    start_time = 0.0
    write_file_num = 0
    onsets, offsets, syll_specs, syll_lens, syll_times = [], [], [], [], []
    if verbose:
        print("Processing .wav files from ", load_dir)
    for filename in tqdm(filenames):
        # Make sure the samplerate is correct and the audio is mono.
        temp_fs, audio = wavfile.read(filename)
        assert(temp_fs == fs)
        if audio.dim > 1:
            audio = audio[0,:]

        # Convert to a magnitude-only spectrogram.
        f, _, Zxx = stft(audio, fs=fs, nperseg=nperseg, noverlap=noverlap)
        i1 = np.searchsorted(f, min_freq)
        i2 = np.searchsorted(f, max_freq)
        spec = np.log(np.abs(Zxx[i1:i2,:]) + EPS)
        print("min freq:", f[0], 0.0, f[-1], fs/2)
        f = f[i1:i2]

        # Collect syllable onsets and offsets.
        onsets, offsets = amp_seg.get_onsets_offsets(spec, onsets, offsets, **params)
        syll_specs, syll_lens , syll_times = amp_seg.get_specs(onsets, offsets, spec,
                                    start_time, syll_specs, syll_lens, syll_times, **params)

        # Write a file when we have enough syllables.
        while len(onsets) >= sylls_per_file:
            save_filename = "syllables_"+str(write_file_num).zfill(3)+'.hdf5'
            with h5py.File(save_filename, "w") as f:
                f.create_dataset('syll_specs', data=syll_specs)
                f.create_dataset('syll_lens', data=syll_lens)
                f.create_dataset('syll_times', data=syll_times)
                for key in metadata:
                    f.attrs[key] = metadata[key]
            for l in [onsets, offsets, syll_specs, syll_lens, syll_times]:
                l = l[sylls_per_file:]
            write_file_num += 1
        start_time += len(audio) / fs


















    ###
