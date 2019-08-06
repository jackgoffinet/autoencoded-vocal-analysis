import numpy as np
from scipy.io import wavfile
import os

FS = 32000

wav_files = sorted([i for i in os.listdir() if i[-4:] == '.wav'])

result = []
for wav_file in wav_files:
	fs, audio = wavfile.read(wav_file)
	assert FS == fs, "found fs="+str(fs)+" in "+wav_file
	result.append(audio)

audio = np.concatenate(result)
print(audio.shape)
wavfile.write('big_audio.wav', FS, audio)
