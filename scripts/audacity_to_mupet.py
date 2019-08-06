"""
Read Audacity syllable onset and offset labels and write in MUPET CSV format.

"""
import sys
import numpy as np
import os

NUM_COLS = 13
HEADER = "Syllable number,Syllable start time (sec),Syllable end time (sec),inter-syllable interval (sec),syllable duration (msec),starting frequency (kHz),final frequency (kHz),minimum frequency (kHz),maximum frequency (kHz),mean frequency (kHz),frequency bandwidth (kHz),total syllable energy (dB),peak syllable amplitude (dB)"
FMT = ['%.2f'] * NUM_COLS
FMT[0] = '%.0f'
FMT[1] = '%.4f'
FMT[2] = '%.4f'
FMT[3] = '%.5f'


if __name__ == '__main__':
	assert len(sys.argv) == 2
	filenames = [os.path.join(sys.argv[1],i) for i in os.listdir(sys.argv[1])]
	filenames = [i for i in filenames if i[-4:] == '.txt']
	for filename in filenames:
		d = np.loadtxt(filename)
		arr = np.zeros((len(d), NUM_COLS))
		arr[:,1:3] = d # onset, offset
		arr[:-1,3] = arr[1:,1] - arr[:-1,2] # inter-syllable interval
		arr[:,4] = (arr[:,2] - arr[:,1]) * 1e3 # duration (ms)
		arr[:,5] = 50.0 # start freq (kHz)
		arr[:,6] = 50.0 # stop freq (kHz)
		arr[:,7] = 30.0	# min freq
		arr[:,8] = 120.0 # max freq
		arr[:,9] = 50.0 # mean freq
		arr[:,10] = 70.0 # bandwidth
		arr[:,11] = 20.0 # total syllable energy (dB)
		arr[:,12] = 15.0 # peak syllable amplitude (dB)
		arr[:,0] = 1 + np.arange(len(arr))
		save_fn = filename[:-4] + '_mupet.csv'
		np.savetxt(save_fn, arr, fmt=FMT, delimiter=',', header=HEADER, comments='')
		
