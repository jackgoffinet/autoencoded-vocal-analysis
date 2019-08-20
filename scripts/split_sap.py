"""
Split a SAP CSV by filename and write to a feature directory.

This makes the output of SAP more similar to the output of MUPET or DeepSqueak.
The SAP CSV can be made by saving the Excel spreadsheet output of a SAP 2011
syllable table as a .csv file. I've tested this using LibreOffice Calc, but
Excel should behave the same. You can automate this conversion by using the
Python package xlrd: https://xlrd.readthedocs.io/en/latest/

Usage:
------
$ python split_sap.py sap_features.csv feature_directory

"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"

import numpy as np
import os
import sys

HEADER = "duration,start,amplitude,pitch,FM,AM^2,entropy,goodness,mean freq,"+\
	"pitch variance,FM variance,entropy variance,goodness variance,"+\
	"mean frequency variance,AM variance,month,day,hour,minute,second,cluster"



if __name__ == '__main__':
	sap_csv = sys.argv[1]
	output_dir = sys.argv[2]

	usecols = tuple(list(range(2,23)))
	d = np.loadtxt(sap_csv, delimiter=',', skiprows=2, usecols=usecols)

	fns = []
	with open(sap_csv, 'r') as f:
		f.readline()
		f.readline()
		for i in range(len(d)):
			fns.append(str(f.readline().split(',')[-2]))

	fns = np.array(fns)
	for fn in np.unique(fns):
		indices = np.argwhere(fns == fn).flatten()
		out_fn = os.path.join(output_dir, fn[:-4] + '.csv')
		np.savetxt(out_fn, d[indices], delimiter=',', fmt='%.5f', header=HEADER)





###
