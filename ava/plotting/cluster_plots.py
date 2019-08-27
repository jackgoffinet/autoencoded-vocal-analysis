"""
Cluster and calculate BIC scores for syllable features.

"""
__author__ = "Jack Goffinet"
__date__ = "July 2019"

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from sklearn.mixture import GaussianMixture


colors = ['navy', 'turquoise', 'darkorange', 'mediumseagreen', 'orchid']


def plot_toy_problem():
	np.random.seed(42)
	data = np.random.normal(size=(3000,2))
	np.random.seed(None)
	data[:,0] *= 2
	bics = []
	for i in range(1,20):
		gmm = GaussianMixture(n_components=i, covariance_type='spherical').fit(data)
		bics.append(gmm.bic(data))
	bics = np.array(bics)
	n_components = np.argmin(bics)+1
	print("n components:", n_components)
	gmm = GaussianMixture(n_components=n_components, covariance_type='spherical').fit(data)
	plt.scatter(data[:,0], data[:,1], c='k', alpha=0.8, marker='+', s=0.9)
	plt.xlim(np.min(data[:,0]), np.max(data[:,0]))
	plt.ylim(np.min(data[:,1]), np.max(data[:,1]))
	plt.axis('off')
	plt.gca().set_aspect('equal', 'datalim')
	plt.savefig('just_scatter_gmm_plot.png')
	make_ellipses(gmm, plt.gca())
	plt.savefig('gmm_plot.png')
	plt.close('all')


def make_ellipses(gmm, ax):
	# Author: Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
	# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
	# License: BSD 3 clause
	for n in range(len(gmm.means_)):
		if gmm.covariance_type == 'full':
			covariances = gmm.covariances_[n][:2, :2]
		elif gmm.covariance_type == 'tied':
			covariances = gmm.covariances_[:2, :2]
		elif gmm.covariance_type == 'diag':
			covariances = np.diag(gmm.covariances_[n][:2])
		elif gmm.covariance_type == 'spherical':
			covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
		v, w = np.linalg.eigh(covariances)
		u = w[0] / np.linalg.norm(w[0])
		angle = np.arctan2(u[1], u[0])
		angle = 180 * angle / np.pi  # convert to degrees
		v = 2. * np.sqrt(2.) * np.sqrt(v)
		ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
								  180 + angle, color=colors[n])
		ell.set_clip_box(ax.bbox)
		ell.set_alpha(0.5)
		ax.add_artist(ell)
		ax.set_aspect('equal', 'datalim')


if __name__ == '__main__':
	plot_toy_problem()


###
