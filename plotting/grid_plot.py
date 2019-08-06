"""
Plot a grid of images.

"""
__author__ = "Jack Goffinet"
__date__ = "July 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np



def grid_plot(specs, filename, gap=3):
	"""
	Parameters
	----------
	specs : numpy.ndarray
		...

	filename : str
		Save the image here.
	"""
	a, b, c, d = specs.shape
	dx, dy = d+gap, c+gap
	height = a*c + (a-1)*gap
	width = b*d + (b-1)*gap
	img = np.zeros((height, width))
	for j in range(a):
		for i in range(b):
			img[j*dy:j*dy+c,i*dx:i*dx+d] = specs[-j-1,i]
	for i in range(1,b):
		img[:,i*dx-gap:i*dx] = np.nan
	for j in range(1,a):
		img[j*dy-gap:j*dy,:] = np.nan
	plt.imshow(img, aspect='equal', origin='lower', interpolation='none',
		vmin=0, vmax=1)
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(filename)
	plt.close('all')



if __name__ == '__main__':
	pass


###
