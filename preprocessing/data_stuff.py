from __future__ import print_function, division
"""
Read a ton of data, write a ton of data.

"""
__author__ = "Jack Goffinet"
__date__ = "September 2018"

import os
import numpy as np
from skimage.transform import resize

import denoise
import syllable as helper
from syll_class import SyllableCollection

min_freq = 35e3
max_freq = 110e3

sep = '/'
read_dir = 'raw_data/'
write_dir = 'data/'

f = np.load('freq_array.npy')
i1 = np.searchsorted(f, min_freq)
i2 = np.searchsorted(f, max_freq)

amp_dict = {}
for folder in os.listdir(read_dir):
    if 'opto' not in folder and 'fd' not in folder:
        continue
    # if 'retest' not in folder: # NOTE: TEMP
    #     continue
    print(folder)
    write_filename_root = write_dir
    if "opto" in folder:
        write_filename_root += "opto/"
    else:
        write_filename_root += "fd/"
    write_filename_root += folder + sep
    if not os.path.exists(write_filename_root):
        os.makedirs(write_filename_root)
    filenames = os.listdir(read_dir + sep + folder)
    counter = 0
    for filename in filenames:
        if filename[-4:] != '.mat':
            continue
        path = read_dir +sep + folder + sep + filename
        _, spectrogram, _, _, _, dt = denoise.get_spectrogram(path)
        norm_spectrogram = helper.normalize(spectrogram)
        sylls = helper.get_syllables(norm_spectrogram, f, 9, dt)
        times = helper.get_syllable_times(sylls)
        spectrogram = np.abs(spectrogram)
        # For each syllable...
        for i in range(len(times)):
            time = times[i]
            if time[0] < 5:
                continue
            data = spectrogram[i1:i2,time[0]-5:time[1]+5]

            if data.shape[1] % 2 == 1:
                data = data[:,:-1]
            # NOTE: TEMP!!!
            # resized = resize(data, (32, data.shape[1] // 2))
            write_filename = write_filename_root + str(counter).zfill(4) + '.npy'
            amp_dict[write_filename] = (np.max(data), np.mean(data), np.mean(np.max(data, axis=0)))
            # np.save(write_filename, resized)
            counter += 1
    np.save('amp_dict.npy', amp_dict)

    print(counter)






#####
