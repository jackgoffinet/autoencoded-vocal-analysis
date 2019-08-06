"""
Get correlations between latent features and traditional features.

TO DO:
	from sklearn.preprocessing import PolynomialFeatures
"""
__author__ = "Jack Goffinet"
__date__ = "July 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA

from .data_container import PRETTY_NAMES



def correlation_plot_DC(dc, fields, filename='feature_correlation.pdf'):
	"""


	"""
	latent = dc.request('latent_means')
	field_data = {}
	field_corrs = {}
	for field in fields:
		field_data[field] = dc.request(field)
		field_corrs[field] = get_correlation(latent, field_data[field])
	# Sort.
	corrs = np.array([field_corrs[field] for field in fields])
	perm = np.argsort(corrs)
	corrs = corrs[perm]
	fields = np.array(fields)[perm]
	# Plot.
	X = np.arange(len(field_data))
	tick_labels = [PRETTY_NAMES[field] for field in fields]
	plt.bar(X, corrs)
	plt.xticks(X, tick_labels, rotation=60)
	plt.ylim(0,1)
	plt.tight_layout()
	plt.title("Latent/Traditional Feature $R^2$ Values")
	plt.savefig(os.path.join(dc.plots_dir, filename))
	plt.close('all')


def knn_variance_explained_plot_DC(dc, fields, k=8, n_fold=5, \
	filename='knn_variance_explained.pdf'):
	"""

	"""
	# Do k-fold cross-validation to get average test set variance explained.
	latent = dc.request('latent_means')
	k_fold = KFold(n_splits=n_fold, shuffle=True, random_state=42)
	r2s = np.zeros(len(fields))
	for i, field in enumerate(fields):
		feature = dc.request(field)
		r2 = 0.0
		for fold_num, (train, test) in enumerate(k_fold.split(latent, feature)):
			reg = KNeighborsRegressor(n_neighbors=k)
			reg.fit(latent[train], feature[train])
			r2 += (reg.score(latent[test], feature[test]) - r2) / (fold_num + 1)
		r2s[i] = r2
	# Sort.
	perm = np.argsort(r2s)
	r2s = r2s[perm]
	fields = np.array(fields)[perm]
	# Plot.
	X = np.arange(len(fields))
	tick_labels = [PRETTY_NAMES[field] for field in fields]
	plt.bar(X, r2s)
	plt.xticks(X, tick_labels, rotation=80)
	plt.ylim(0,1)
	plt.tight_layout()
	plt.title("Latent/Traditional Feature $R^2$ Values")
	plt.savefig(os.path.join(dc.plots_dir, filename))
	plt.close('all')


def pairwise_correlation_plot_DC(dc, fields, filename='pairwise.pdf'):
	field_data = {}
	for field in fields:
		temp = dc.request(field)
		temp -= np.mean(temp)
		std_dev = np.std(temp, ddof=1)
		field_data[field] = temp / std_dev
	result = np.ones((len(fields), len(fields)))
	for i in range(len(fields)-1):
		for j in range(i+1,len(fields),1):
			d1 = field_data[fields[i]]
			d2 = field_data[fields[j]]
			temp = np.dot(d1, d2) / (len(d1)-1)
			result[i,j] = temp
			result[j,i] = temp
	plt.imshow(result, vmin=-1, vmax=1, cmap='bwr', origin='lower')
	cbar = plt.colorbar(orientation='horizontal', pad=0.06, fraction=0.046)
	cbar.set_ticks([-1, 0, 1])
	cbar.set_ticklabels([-1, 0, 1])
	plt.title("Pairwise feature correlations")
	tick_labels = [PRETTY_NAMES[field] for field in fields]
	plt.yticks(np.arange(len(fields)), tick_labels)
	plt.xticks([],[])
	plt.tight_layout()
	plt.savefig(os.path.join(dc.plots_dir, filename))
	plt.close('all')


def feature_pca_plot_DC(dc, fields, filename='feature_pca.pdf'):
	field_data = {}
	for field in fields:
		field_data[field] = dc.request(field)
	field_arr = np.zeros((len(field_data[fields[0]]), len(fields)))
	for i in range(len(fields)):
		field_arr[:,i] = field_data[fields[i]]
	latent = dc.request('latent_means')
	# Find variance explained.
	transform = PCA(n_components=len(fields), whiten=True).fit(field_arr)
	feature_variance = np.cumsum(transform.explained_variance_ratio_).tolist()
	transform = PCA(n_components=latent.shape[1], whiten=True).fit(latent)
	latent_variance = np.cumsum(transform.explained_variance_ratio_).tolist()
	plt.plot(list(range(len(fields)+1)), [0]+feature_variance, label='MUPET features')
	plt.plot(list(range(latent.shape[1]+1)), [0]+latent_variance, label='Latent features')
	plt.title("Variance Explained vs. Number of Components")
	plt.ylabel("Portion of Variance Explained")
	plt.xlabel("Number of Components")
	plt.legend(loc='lower right')
	plt.axhline(y=0.9, c='r', ls='--')
	plt.xlim(0,6)
	plt.ylim(0,1)
	plt.savefig(os.path.join(dc.plots_dir, filename))
	plt.close('all')



def get_correlation(latent, feature_vals):
	""" """
	reg = LinearRegression().fit(latent, feature_vals.reshape(-1,1))
	return reg.score(latent, feature_vals.reshape(-1,1))



if __name__ == '__main__':
	pass



###
