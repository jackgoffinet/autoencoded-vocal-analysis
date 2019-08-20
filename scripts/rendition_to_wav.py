"""
Write song renditions as separate wave files.

"""

import numpy as np
from scipy.io import wavfile
import os
import sys

PRE_POST = 0.010

if __name__ == '__main__':
	read_dir = sys.argv[1]
	write_dir = sys.argv[2]
	if not os.path.exists(write_dir):
		os.makedirs(write_dir)
	wav_files = [os.path.join(read_dir,i) for i in os.listdir(read_dir) if i[-4:] == '.wav']
	wav_files = sorted(wav_files)
	out_file_num = 0
	for wav_file in wav_files:
		fs, audio = wavfile.read(wav_file)
		txt_file = wav_file[:-4] + '.txt'
		segments = np.loadtxt(txt_file)
		if len(segments) > 0:
			segments = segments.reshape(-1,2)
		for segment in segments:
			try:
				i1 = max(0, int(fs * (segment[0] - PRE_POST)))
			except:
				print(segment)
				quit()
			i2 = min(len(audio), int(fs * (segment[1] + PRE_POST)))
			audio_chunk = audio[i1:i2]
			out_fn = os.path.join(write_dir, str(out_file_num).zfill(4)+'.wav')
			wavfile.write(out_fn, fs, audio_chunk)
			out_file_num += 1





###
