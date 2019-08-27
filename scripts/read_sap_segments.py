"""
Read SAP CSVs and write onset/offset times.

"""
import numpy as np
import sys
import os


if __name__ == '__main__':
	csv_file = sys.argv[1]
	audio_dir = sys.argv[2]

	fn_convert = lambda x: int(str(x)[2:-5])
	durations, onsets, file_nums = np.loadtxt(csv_file, delimiter=',', \
		skiprows=2, usecols=(2,3,-2), converters={-2: fn_convert}, unpack=True)
	offsets = onsets + durations
	onsets, offsets = onsets/1e3, offsets/1e3
	for file_num in np.unique(file_nums):
		indices = np.argwhere(file_nums == file_num).flatten()
		result = np.zeros((len(indices),2))
		result[:,0] = onsets[indices]
		result[:,1] = offsets[indices]
		write_fn = os.path.join(audio_dir, str(int(round(file_num))).zfill(4)+'.txt')
		np.savetxt(write_fn, result, fmt='%.5f')
