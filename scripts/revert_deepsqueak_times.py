"""
Replace DeepSqueak times with MUPET times.

The DeepSqueak `import from MUPET` feature changes the times in the MUPET file,
so this script reverts these times to the original MUPET times.

"""
__author__ = "Jack Goffinet"
__date__ = "September 2019"

import numpy as np
import os
import sys

USAGE = "$ python revert_deepsqueak_times.py deepsqueak_dir mupet_dir"


if __name__ == '__main__':
	assert len(sys.argv) == 3, USAGE
	ds_dir = sys.argv[1]
	mupet_dir = sys.argv[2]

	mupet_files = [i for i in os.listdir(mupet_dir) if i[-4:] == '.csv']
	ds_files = [i[:-4]+'_Stats.csv' for i in mupet_files]
	mupet_files = [os.path.join(mupet_dir, i) for i in mupet_files]
	ds_files = [os.path.join(ds_dir, i) for i in ds_files]

	for mupet_file, ds_file in zip(mupet_files, ds_files):
		print(mupet_file, ds_file)
		onsets, offsets = np.loadtxt(mupet_file, delimiter=',', skiprows=1, \
				usecols=(1,2), unpack=True)

		with open(ds_file, 'r') as f:
			lines = f.readlines()

		with open(ds_file, 'w') as f:
			f.writelines(lines[0])
			for i in range(1, len(lines)):
				line = lines[i]
				line = line.split(',')
				line[4] = '{0:.4f}'.format(onsets[i-1])
				line[5] = '{0:.4f}'.format(offsets[i-1])
				line = ','.join(line)
				f.writelines(line)



###
