"""
Add MUPET cluster assignments to MUPET syllable feature files.

"""

import numpy as np
import sys
import os


def add_cluster(fn, c_files, c_onsets, c_clusters):
	with open(fn, 'r') as f:
		header = f.readline()
		lines = f.readlines()
	to_write = ''
	i = 0
	fn_prefix = os.path.split(fn)[1][:-4]
	header = header[:-1]+',cluster\n'
	new_lines = [header]
	for line in lines:
		temp = line.split(',')
		onset = float(temp[1])
		while c_files[i] != fn_prefix:
			i += 1
		while c_onsets[i] < onset:
			i += 1
		assert c_onsets[i] == onset
		new_lines.append(line[:-1]+','+str(c_clusters[i])+'\n')
	with open(fn, 'w') as f:
		f.writelines(new_lines)



if __name__ == '__main__':
	with open('mupet_clusters.csv', 'r') as f:
		_ = f.readline()
		lines = f.readlines()
	c_files, c_onsets, c_clusters = [], [], []
	for line in lines:
		temp = line.split(',')
		c_files.append(temp[1])
		c_onsets.append(float(temp[3]))
		c_clusters.append(int(temp[5]))
	csv_dir = 'C57_MUPET_detect'
	fns = [os.path.join(csv_dir,i) for i in os.listdir(csv_dir) if i[-4:] == '.csv']
	for fn in fns:
		add_cluster(fn, c_files, c_onsets, c_clusters)
	
		
