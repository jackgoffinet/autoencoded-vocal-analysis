"""
Get correlations between latent features and traditional features.


"""
__author__ = "Jack Goffinet"
__date__ = "July-August 2019"


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn import preprocessing

from ava.data.data_container import PRETTY_NAMES_NO_UNITS



def correlation_plot_DC(dcs, all_fields, axarr=None, save_and_close=True, \
	filename='feature_correlation.pdf'):
	"""


	"""
	trad_r2s, latent_r2s = [], []
	trad_names, latent_names = [], []
	for dc, fields in zip(dcs, all_fields):
		latent = dc.request('latent_means')
		latent = PCA(n_components=latent.shape[1]).fit_transform(latent)
		field_data = []
		field_r2s = {}
		for field in fields:
			data = zscore(dc.request(field))
			field_data.append(data)
			r2 = get_r2(latent, field_data[-1])
			trad_r2s.append(r2)
			trad_names.append(PRETTY_NAMES_NO_UNITS[field])
		field_data = np.stack(field_data)
		for latent_pc in range(latent.shape[1]):
			r2 = get_r2(field_data, latent[:,latent_pc])
			latent_r2s.append(r2)
			latent_names.append("Latent "+str(latent_pc+1))

	# Sort.
	trad_r2s = np.array([field_r2s[field] for field in fields])
	perm = np.argsort(r2s)
	r2s = r2s[perm]
	fields = np.array(fields)[perm]
	# Plot.
	if axarr is None:
		_, axarr = plt.subplots(1,2)
	Y = np.arange(len(field_data), dtype='int')
	tick_labels = [PRETTY_NAMES_NO_UNITS[field] for field in fields]
	axarr[0].barh(Y, r2s)
	axarr[0].set_xticks([0,0.25,0.5,0.75,1.0])
	axarr[0].set_xticklabels([0.0,'',0.5,'',1.0])
	axarr[0].set_xlabel("Variance Explained by Latent Features")
	# ax.set_ylabel("MUPET Feature")
	# for val in [0.25, 0.50, 0.75]:
	# 	ax.axvline(x=val, c='k', alpha=0.5, lw=0.5)
	axarr[0].set_yticks([])
	# ax.set_yticklabels(tick_labels, fontdict={'fontsize':8})
	for i in Y:
		axarr[0].text(0.01, i-0.1, tick_labels[i], fontsize=8, color='w')
	axarr[0].set_xlim(0,1)
	# ax.set_title("Latent/Traditional Feature Correlations", fontsize=10)
	if save_and_close:
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
	tick_labels = [PRETTY_NAMES_NO_UNITS[field] for field in fields]
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
	plt.imshow(result, vmin=-1, vmax=1, cmap='Grays', origin='lower')
	cbar = plt.colorbar(orientation='horizontal', pad=0.06, fraction=0.046)
	cbar.set_ticks([-1, 0, 1])
	cbar.set_ticklabels([-1, 0, 1])
	plt.title("Pairwise feature correlations")
	tick_labels = [PRETTY_NAMES_NO_UNITS[field] for field in fields]
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


def two_subplot_correlation_plot_DC(dcs, all_fields, colors=None, axs=None, \
	save_and_close=True, labels=None, filename='temp.pdf'):
	"""
	TO DO: shuffle scatter
	"""
	if colors is None:
		colors = ['k'] * len(dcs)
	# Collect data.
	plot_1_xs, plot_2_xs, plot_1_ys, plot_2_ys = [], [], [], []
	plot_1_colors, plot_2_colors = [], []
	for i, dc, fields in zip(range(len(dcs)), dcs, all_fields):
		latent = dc.request('latent_means')
		all_field_data = [dc.request(field) for field in fields]
		all_field_data = np.stack(all_field_data).T
		all_field_data = preprocessing.scale(all_field_data) # z-score
		pca = PCA()
		all_field_data = pca.fit_transform(all_field_data)
		variances = pca.explained_variance_ratio_
		for field_num in range(all_field_data.shape[1]):
			single_field = all_field_data[:,field_num].reshape(-1,1)
			reg = LinearRegression().fit(latent, single_field)
			plot_1_xs.append(variances[field_num])
			plot_1_ys.append(plot_1_xs[-1]*reg.score(latent, single_field))
			plot_1_colors.append(colors[i])
		pca = PCA()
		latent = pca.fit_transform(latent)
		variances = pca.explained_variance_ratio_
		for latent_num in range(latent.shape[1]):
			single_latent = latent[:,latent_num].reshape(-1,1)
			reg = LinearRegression().fit(all_field_data, single_latent)
			plot_2_xs.append(variances[latent_num])
			plot_2_ys.append(plot_2_xs[-1]*reg.score(all_field_data, single_latent))
			plot_2_colors.append(colors[i])
	# Plot.
	if axs is None:
		_, axs = plt.subplots(2,1)
	axs[0].scatter(plot_1_xs, plot_1_ys, c=plot_1_colors, alpha=0.6)
	# axs[0].set_title('Latent Features redicts trad')
	axs[0].set_ylabel('Traditional Feature Variance')
	axs[0].set_xlabel('Variance Explained by Latent Features')
	max_val = max(plot_1_xs)
	# axs[0].set_xlim(-0.02,max_val+0.02)
	# axs[0].set_ylim(-0.02,max_val+0.02)
	# axs[0].plot([0,max_val], [0,max_val], ls='--')
	axs[1].scatter(plot_2_xs, plot_2_ys, c=plot_2_colors, alpha=0.6)
	# axs[1].set_title('trad predicts latent')
	axs[1].set_ylabel('Latent Feature Variance')
	axs[1].set_xlabel('Variance Explained by Traditional Features')
	max_val = max(plot_2_xs)
	# axs[1].plot([0,max_val], [0,max_val], ls='--')
	if labels is not None:
		patches = [mpatches.Patch(color=colors[i], label=labels[i]) \
				for i in range(len(labels))]
		axs[0].legend(handles=patches, loc='upper left') # bbox_to_anchor=(0.5, 0.5)
	# axs[1].set_xlim(-0.02,max_val+0.02)
	# axs[1].set_ylim(-0.02,max_val+0.02)
	if save_and_close:
		plt.savefig(filename)
		plt.close('all')


def triptych_correlation_plot_DC(dcs, all_fields, colors=None, ax=None, \
	save_and_close=True, jitter=0.25, filename='triptych.pdf'):
	"""
	A correlation scatter plot.

	Three parts: 1) portion of traditional feature variance explained by linear
	combinations of latent features. 2) portion of 'used' latent feature
	variance explained by linear combinations of traditional features. 3)
	portion of unused latent feature variance explained by linear combinations
	of traditional features.

	Parameters
	----------
	dcs : ...
		....

	"""
	if colors is None:
		colors = ['k'] * len(dcs)
	# Collect data.
	plot_1_vars, plot_2_vars, plot_3_vars = [], [], []
	plot_1_colors, plot_2_colors, plot_3_colors = [], [], []
	for i, dc, fields in zip(range(len(dcs)), dcs, all_fields):
		latent = dc.request('latent_means')
		all_field_data = [dc.request(field) for field in fields]
		all_field_data = np.stack(all_field_data).T
		print("all field data", all_field_data.shape)
		print("latent", latent.shape)
		for field_num in range(all_field_data.shape[1]):
			field_data = all_field_data[:,field_num]
			var = get_var_explained(latent, field_data)
			plot_1_vars.append(var)
			plot_1_colors.append(colors[i])
		# CCA
		pca = PCA()
		latent = pca.fit_transform(latent)
		var_explained = np.cumsum(pca.explained_variance_ratio_)
		magic_index = np.searchsorted(var_explained, 0.98)
		for latent_dim in range(latent.shape[1]):
			latent_feature = latent[:,latent_dim]
			var = get_var_explained(all_field_data, latent_feature)
			if latent_dim < magic_index:
				plot_2_vars.append(var)
				plot_2_colors.append(colors[i])
			else:
				plot_3_vars.append(var)
				plot_3_colors.append(colors[i])
	# Plot.
	if ax is None:
		ax = plt.gca()
	np.random.seed(42)
	y_vals = jitter * np.random.rand(len(plot_1_vars))
	ax.scatter(y_vals, plot_1_vars, c=plot_1_colors, alpha=0.5)
	y_vals = 1.0 + jitter * np.random.rand(len(plot_2_vars))
	ax.scatter(y_vals, plot_2_vars, c=plot_2_colors, alpha=0.5)
	y_vals = 2.0 + jitter * np.random.rand(len(plot_3_vars))
	ax.scatter(y_vals, plot_3_vars, c=plot_3_colors, alpha=0.5)
	np.random.seed(None)
	if save_and_close:
		plt.savefig(filename)
		plt.close('all')



def get_r2(latent, feature_vals):
	""" """
	reg = LinearRegression().fit(latent, feature_vals.reshape(-1,1))
	return reg.score(latent, feature_vals.reshape(-1,1))


def get_var_explained(vals_1, vals_2):
	"""Get variance of vals_2 explained by vals_1."""
	reg = LinearRegression().fit(vals_1, vals_2.reshape(-1,1))
	return reg.score(vals_1, vals_2.reshape(-1,1))



if __name__ == '__main__':
	pass



###
