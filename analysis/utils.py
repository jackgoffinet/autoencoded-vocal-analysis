"""
Various helpful functions.
"""
__author__ = "Jack Goffinet"
__date__ = "May 2019"

import h5py
from matplotlib import cm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from scipy.io import loadmat



def plot_freq_distribution():
	helium = [0,20,30,40,50,60,80]
	bin_freqs = np.linspace(30,135,128,endpoint=True)
	freqs = np.zeros((128,len(helium)))
	for i in range(len(helium)):
		load_dir = 'data/processed/helium_mice/'+str(helium[i])+'_He/'
		filenames = [os.path.join(load_dir, i) for i in os.listdir(load_dir) if i[-5:] == '.hdf5']
		for filename in filenames:
			f = h5py.File(filename, 'r')
			temp = np.sum(f['specs'], axis=(0,-1))
			temp /= np.sum(temp)
			freqs[:,i] += temp
		# freqs /= np.sum(freqs)
	np.save('freqs.npy', freqs)
	freqs = np.load('freqs.npy')
	for i in range(freqs.shape[1]):
		plt.plot(bin_freqs, freqs[:,i])
		plt.xlabel('Frequecy (kHz)')
		plt.ylabel('Probability')
		plt.savefig('temp_'+str(helium[i])+'.pdf')
		plt.close('all')
	fig = plt.figure()
	Z = freqs
	X, Y = bin_freqs, np.array(helium)
	indices = np.argwhere(X > 40).flatten()
	X = X[indices]
	X, Y = np.meshgrid(X, Y)
	Z = Z[indices]
	# quit()
	# ax = fig.gca(projection='3d')
	# ax.plot_surface(X, Y, Z.T,
	# 		linewidth=0, antialiased=False, cmap=cm.viridis)
	ax = fig.gca()
	cs = ax.contourf(X, Y, Z.T, cmap=cm.viridis)
	ax.contour(cs, colors='k')
	plt.savefig('temp.pdf')




def split_latent_by_individual(load_filename, split_func):
	d = loadmat(load_filename)
	latent = d['latent']
	results = {}
	for i, filename in enumerate(d['filenames']):
		individual = split_func(filname)
		if individual in results:
			results[individual].append(latent[i])
		else:
			results[individual] = [latent[i]]
	return results


def mouse_filename_to_individual(filename):
	return filename.split('/')[-2]


def save_everything(loader, model, save_filename):
	pass


if __name__ == '__main__':
	plot_freq_distribution()




###
