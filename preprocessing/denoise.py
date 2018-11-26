from __future__ import print_function, division
"""Fourier denoise

Usage:
$ python denoise.py RAm_3_male_USVs_to_female/15minFemale_sparse_chunk001.mat

TO DO:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.check_COLA.html#scipy.signal.check_COLA
"""
__author__ = "Jack Goffinet"
__date__ = "August-September 2018"

import sys
import numpy as np
import soundfile as sf
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt


fs = 250000 # samplerate
lowcut = 25000 # for bandpass filter
highcut = 110000
nperseg = 2 ** 9 # STFT window size
noverlap = nperseg // 2
amp_threshold = 0.08 # for STFT denoising, amp = np.max(np.abs(Zxx))



def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = signal.butter(order, [low, high], btype='band')
	return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = signal.lfilter(b, a, data)
	return y


def get_spectrogram(filename):
	# Load matlab data.
	data = loadmat(filename)['spike2Chunk'].reshape(-1)
	# Apply Butterworth filter.
	data = butter_bandpass_filter(data, lowcut, highcut, fs, order=6)
	# Compute STFT.
	f, t, Zxx = signal.stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
	df = f[1] - f[0]
	dt = t[1] - t[0]
	# Denoise by thresholding values.
	Zxx_denoise = np.where(np.abs(Zxx) >= amp_threshold, Zxx, 0.0)
	return Zxx, Zxx_denoise, f, df, t, dt



if __name__ == '__main__':
	if len(sys.argv) != 2:
		print('Usage: $ python denoise.py <filename>')
		quit()
	Zxx, f, df, t, dt = get_spectrogram(sys.argv[1])
	plt.imshow(np.abs(Zxx[:,1500:2000]), cmap="Greys")
	plt.savefig('temp.pdf')
