from scipy.io import loadmat, wavfile
import numpy as np
import os

FS = 300000

filenames = [i for i in os.listdir() if len(i)>4 and i[-4:] == '.mat']
for fn in filenames:
	d = loadmat(fn)
	assert d['fs'] == FS, "found fs="+str(d['fs'])+" in "+fn
	audio = d['spike2Chunk'].flatten()
	filename = fn[:-4] + '.wav'
	wavfile.write(filename, FS, audio)
