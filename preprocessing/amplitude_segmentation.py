from __future__ import print_function, division
"""
Amplitude based syllable segmentation.
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
i1, i2 = 2, 66
epsilon = 1e-12


# thresholds for syllable onset & offset
th_1, th_2, th_3, th_4 = 0.25, 0.00, 0.0, 0.3
min_var = 1.2
max_syll_len = 192
min_syll_len = 2
sylls_per_file = 1000

if __name__ == '__main__':
    # Get all the filenames.
    dirs = ['bird_data/' + i for i in ['pur224_55dph/', 'pur224_70dph/', 'yel118_adult/']]
    filenames = []
    for dir in dirs:
        filenames = [dir + i for i in listdir(dir) if i[-4:] == '.wav']
        sylls = np.zeros((sylls_per_file, i2-i1, max_syll_len))
        durations = np.zeros(sylls_per_file, dtype='int')
        out_n = 0
        discarded = 0
        # Loop over the files.
        for filename in filenames:
            if out_n == 50000:
                break
            # Load audio, convert to spectrogram.
            audio, temp_fs = sf.read(filename)
            assert(temp_fs == fs)
            f, _, Zxx = stft(audio, fs=fs, nperseg=nperseg, noverlap=noverlap)
            spec = np.log(np.abs(Zxx[i1:i2,:])+epsilon)
            spec = np.flip(spec, axis=0)
            f = f[i1:i2]
            # Get amplitude data and its derivative.
            amps = np.mean(spec, axis=0)
            smooth = gaussian_filter(spec, [2,2])
            x_grads = convolve2d(smooth, np.array([[1,0,-1]]), mode='same', boundary='symm') # Sobel filter
            x_grads = np.mean(x_grads, axis=0)
            x_grads = gaussian_filter1d(x_grads, 3.0)

            # Normalize.
            amps -= np.percentile(amps, 2)
            amps /= np.percentile(amps, 98)
            amps = gaussian_filter1d(amps, 2.0)
            x_grads /= np.percentile(np.abs(x_grads), 98)


            # Collect syllable times using hard thresholds.
            on_segs = []
            off_segs = []
            last = 'off'
            for i in range(len(amps)):
                if last == 'off':
                    if x_grads[i] > th_1 and amps[i] > th_2:
                        on_segs.append(i)
                        last = 'on'
                else:

                    if x_grads[i] < th_3 and amps[i] < th_4:
                        var = np.mean(np.var(spec[:,on_segs[-1]:i], axis=0))
                        if var < min_var:
                            on_segs = on_segs[:-1]
                            discarded += 1
                        else:
                            off_segs.append(i)
                        last = 'off'
                    elif i - on_segs[-1] >= max_syll_len:
                        last = 'off'
                        on_segs = on_segs[:-1]

            # if np.random.rand() < 0.03:
            # # if filename == 'bird_data/pur224_70dph/pur224_43347.47491890_9_4_13_11_31.wav':
            #     print(filename)
            #     _, axarr = plt.subplots(3,1,sharex=True)
            #     axarr[0].imshow(spec[:,1600:2350], aspect='auto')
            #     axarr[1].imshow(spec[:,1600:2350], aspect='auto')
            #     for i, t in enumerate(on_segs):
            #         if t > 2350:
            #             break
            #         if t < 1600:
            #             continue
            #         var = np.mean(np.var(spec[:,t:off_segs[i]], axis=0))
            #         # temp = np.mean(spec[:,t:off_segs[i]], axis=1)
            #         # print(temp.shape)
            #         # _,_,_,_,std_err = linregress(range(len(temp)), temp)
            #         print(t,var)
            #         # quit()
            #
            #
            #         axarr[1].axvline(x=t-1600, c='k', linewidth=0.5)
            #     for t in off_segs:
            #         if t > 2350:
            #             break
            #         if t < 1600:
            #             continue
            #         axarr[1].axvline(x=t-1600, c='r', linewidth=0.5)
            #     axarr[2].plot(amps[1600:2350], c='b', linewidth=0.5)
            #     axarr[2].plot(x_grads[1600:2350], c='r', linewidth=0.5)
            #     axarr[2].axhline(y=0.0, linewidth='0.5', c='k')
            #     plt.savefig('temp.pdf')
            #     quit()

            for i in range(len(off_segs)):
                t1, t2 = on_segs[i], off_segs[i]
                if t2 - t1 + 1 > max_syll_len or t2-t1 < min_syll_len:
                    discarded += 1
                    continue

                temp_spec = spec[:,t1:t2+1]

                log_f = np.geomspace(f[0], f[-1], num=i2-i1, endpoint=True)
                log_f[-1] = f[-1]
                log_f[0] = f[0]
                log_f_spec = np.zeros_like(temp_spec)
                # for each time...
                for j in range(temp_spec.shape[1]):
                    interp = interp1d(log_f, temp_spec[:,j], kind='cubic')
                    log_f_spec[:,j] = interp(f)

                # Normalize spectrogram.
                log_f_spec -= np.min(log_f_spec)
                log_f_spec /= np.max(log_f_spec)


                sylls[out_n%sylls_per_file,:,:t2-t1+1] = log_f_spec
                durations[out_n%sylls_per_file] = t2 - t1 + 1
                out_n += 1

                if out_n % sylls_per_file == 0:
                    temp = out_n // sylls_per_file
                    save_filename = dir + "sylls/syll_" + str(temp).zfill(2) + '.npy'
                    np.save(save_filename, sylls)
                    save_filename = dir + "sylls/dur_" + str(temp).zfill(2) + '.npy'
                    np.save(save_filename, durations)
                    sylls = np.zeros_like(sylls)



        print('saved: ',out_n)
        print('discarded: ',discarded)



###
