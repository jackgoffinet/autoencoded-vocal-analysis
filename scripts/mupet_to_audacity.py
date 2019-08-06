"""
Read MUPET CSVs and write text files with onsets and offsets that can be read with Audacity.

"""
import sys
import numpy as np
import os


def convert(filename):
	d = np.loadtxt(filename, skiprows=1, usecols=(1,2), delimiter=',')
	save_fn = filename[:-4] + '_sylls.txt'
	np.savetxt(save_fn, d, fmt='%.6f', delimiter='\t')


if __name__ == '__main__':
	assert len(sys.argv) == 2
	filenames = [os.path.join(sys.argv[1],i) for i in os.listdir(sys.argv[1])]
	filenames = [i for i in filenames if i[-4:] == '.csv']
	for filename in filenames:
		convert(filename)
