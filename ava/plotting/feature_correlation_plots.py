"""
Plot correlations between latent features and traditional features.


"""
__author__ = "Jack Goffinet"
__date__ = "July-September 2019"


from matplotlib.colors import to_rgba
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.patches import Rectangle
import os
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn import preprocessing

from ava.data.data_container import PRETTY_NAMES_NO_UNITS



def correlation_bar_chart_DC(dcs, dc_fields, axs=None, colors=None, top_n=5, \
	bottom_n=5, save_and_close=True, load_data=False, \
	filename='correlation_bar_chart.pdf'):
	"""
	Two-panel horizontal bar chart.

	Parameters
	----------
	dcs : list of ava.data.data_container.DataContainer
		DataContainers
	dc_fields : list of str
		All the fields to plot
	axs : {..., None}
		Axes to plot on. Defaults to ``None``.
	colors : {None, list of str}
		Colors corresponding to `dcs`. Defaults to ``None``.
	top_n : int, optional
		...
	bottom_n : int, optional
		...
	save_and_close : bool, optional
		Defaults to ``True``.
	filename : str
		Defaults to ``'correlation_bar_chart.pdf'``.

	"""
	loaded_data = False
	if load_data:
		try:
			d = np.load('temp_data/bar_chart_data.npy', allow_pickle=True).item()
			trad_r2s, latent_r2s = d['trad_r2s'], d['latent_r2s']
			trad_names, latent_names = d['trad_names'], d['latent_names']
			trad_colors, latent_colors = d['trad_colors'], d['latent_colors']
			loaded_data = True
			print("Loaded data.")
		except:
			print("Unable to load data!")
			pass
	if not loaded_data:
		trad_r2s, latent_r2s = [], []
		trad_names, latent_names = [], []
		trad_colors, latent_colors = [], []
		if colors is None:
			colors = ['b'] * len(dcs)
		for dc, fields, color in zip(dcs, dc_fields, colors):
			latent = dc.request('latent_means')
			pca = PCA(n_components=latent.shape[1])
			latent = pca.fit_transform(latent)
			sig_dims = np.argwhere(pca.explained_variance_ratio_ > 0.01).flatten()
			print("sig_dims", sig_dims)
			latent = latent[:,sig_dims]
			field_data = []
			field_r2s = {}
			for field in fields:
				data = zscore(dc.request(field))
				try:
					r2 = _get_knn_r2(latent, data, k=5)
					field_data.append(data)
					trad_r2s.append(r2)
					trad_names.append(PRETTY_NAMES_NO_UNITS[field])
					trad_colors.append(color)
				except ValueError:
					print("Found low-variance feature:", field)
					print(np.isnan(data).any())
					print(np.mean(np.power(data - np.mean(data), 2)))
					pass
			field_data = np.stack(field_data).T
			for latent_pc in range(latent.shape[1]):
				try:
					r2 = _get_knn_r2(field_data, latent[:,latent_pc], k=5)
					latent_r2s.append(r2)
					latent_names.append("Latent "+str(latent_pc+1))
					latent_colors.append(color)
				except ValueError:
					print("Found low-variance PC:", latent_pc)
					print(field_data.shape, latent.shape)
					data = latent[:,latent_pc]
					print(np.mean(np.power(data - np.mean(data), 2)))
					pass
		d = {
			'trad_r2s': trad_r2s,
			'latent_r2s': latent_r2s,
			'trad_names': trad_names,
			'latent_names': latent_names,
			'trad_colors': trad_colors,
			'latent_colors': latent_colors,
		}
		np.save('temp_data/bar_chart_data.npy', d)

	# Sort.
	trad_r2s = np.array(trad_r2s)
	perm = np.argsort(trad_r2s)
	trad_r2s = trad_r2s[perm]
	trad_names = np.array(trad_names)[perm]
	trad_colors = np.array(trad_colors)[perm]
	latent_r2s = np.array(latent_r2s)
	perm = np.argsort(latent_r2s)
	latent_r2s = latent_r2s[perm]
	latent_names = np.array(latent_names)[perm]
	latent_colors = np.array(latent_colors)[perm]

	# Plot.
	if axs is None:
		_, axs = plt.subplots(1,2)

	for ax in axs:
		ax.set_xticks([0,0.25,0.5,0.75,1.0])
		ax.set_xticklabels([0.0,'',0.5,'',1.0])
		ax.set_yticks([])
		ax.set_xlim(0,1.0)
		for x in [0.25,0.75]:
			ax.axvline(x=x, c='k', ls='--', lw=0.5, alpha=0.3, zorder=1)
		for x in [0.5,1.0]:
			ax.axvline(x=x, c='k', ls='-', lw=0.8, alpha=0.6, zorder=1)
		for direction in ['right', 'top']:
			ax.spines[direction].set_visible(False)
		ax.xaxis.set_ticks_position('bottom')

	# Create axis breaks.
	delta = 1.0 / (bottom_n + top_n + 1)
	y_top = (bottom_n + 0.8) * delta
	y_bottom = (bottom_n + 0.35) * delta
	rect1 = Rectangle((-0.05,y_top), 0.1, y_bottom-y_top, \
		edgecolor='white', facecolor='white', zorder=10, \
		transform=axs[0].transAxes, clip_on=False, linewidth=0.3)
	axs[0].add_patch(rect1)
	rect2 = Rectangle((-0.05,y_top), 0.1, y_bottom-y_top, \
		edgecolor='white', facecolor='white', zorder=10, \
		transform=axs[1].transAxes, clip_on=False, linewidth=0.3)
	axs[1].add_patch(rect2)
	for ax in axs:
		kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, \
			zorder=11, lw=1.0)
		ax.plot([-0.04,0.04], [y_top+0.01, y_top-0.01], **kwargs)
		ax.plot([-0.04,0.04], [y_bottom+0.01, y_bottom-0.01], **kwargs)


	Y = np.arange(top_n + bottom_n + 1)
	axs[0].barh(Y[:bottom_n], trad_r2s[:bottom_n], color=trad_colors[:bottom_n], zorder=2)
	axs[0].barh(Y[-top_n:], trad_r2s[-top_n:], color=trad_colors[-top_n:], zorder=2)
	axs[0].set_xlabel("% Variance Explained\nby Latent Features")

	for i in range(bottom_n):
		axs[0].text(trad_r2s[i]+0.04, Y[i]-0.105, trad_names[i], fontsize=7, \
			color='k', bbox=dict(facecolor='w', edgecolor='w', alpha=0.7))
	for i in range(1,top_n+1):
		axs[0].text(0.015, Y[-i]-0.1, trad_names[-i], fontsize=7, color='w')

	Y = np.arange(top_n + bottom_n + 1)
	axs[1].barh(Y[:bottom_n], latent_r2s[:bottom_n], color=latent_colors[:bottom_n], zorder=2)
	axs[1].barh(Y[-top_n:], latent_r2s[-top_n:], color=latent_colors[-top_n:], zorder=2)
	axs[1].set_xlabel("% Variance Explained\nby Traditional Features")

	for i in range(bottom_n):
		axs[1].text(latent_r2s[i]+0.04, Y[i]-0.105, latent_names[i], fontsize=7, \
			color='k', bbox=dict(facecolor='w', edgecolor='w', alpha=0.7))
	for i in range(1,top_n+1):
		axs[1].text(0.015, Y[-i]-0.1, latent_names[-i], fontsize=7, color='w')

	if save_and_close:
		plt.savefig(os.path.join(dc.plots_dir, filename))
		plt.close('all')



def pairwise_correlation_plot_DC(dc, fields, ax=None, cax=None, \
	save_and_close=True,filename='pairwise.pdf'):
	"""
	Note:
	- add axes, etc...

	"""
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
			result[i,j] = abs(temp)
			result[j,i] = abs(temp)
	# # Sort.
	# tsne = TSNE(n_components=1, metric='precomputed', random_state=42)
	# flat_result = tsne.fit_transform(1.0 - result).flatten()
	# perm = np.argsort(flat_result).flatten()
	# new_result = np.zeros_like(result)
	# for i in range(len(new_result)):
	# 	for j in range(len(new_result)):
	# 		new_result[i,j] = result[perm[i],perm[j]]
	# 		# new_result[perm[i],perm[j]] = result[i,j]
	# result = new_result

	if ax is None:
		ax = plt.gca()
	im = ax.imshow(result, vmin=0, vmax=1, cmap='Greys', aspect='equal') # , origin='lower'
	fig = plt.gcf()
	if cax is None:
		cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
	cbar = fig.colorbar(im, cax=cax, pad=0.06, fraction=0.046, \
			orientation="horizontal")
	cbar.solids.set_edgecolor("face")
	cbar.solids.set_rasterized(True)
	cbar.set_ticks([0, 1])
	cbar.set_ticklabels([0,1])
	ax.set_title("MUPET Feature\nPairwise Absolute\nCorrelations", fontsize=8)
	tick_labels = [PRETTY_NAMES_NO_UNITS[field] for field in fields]
	# ax.set_yticks(np.arange(len(fields)), tick_labels)
	ax.set_xticks([],[])
	ax.set_yticks([],[])
	if save_and_close:
		plt.tight_layout()
		plt.savefig(os.path.join(dc.plots_dir, filename))
		plt.close('all')


def feature_pca_plot_DC(dcs, dc_fields, colors, alpha=0.7, lw=2.5, ax=None, \
	save_and_close=True, filename='feature_pca.pdf'):
	"""
	TO DO: add axes, etc...

	"""
	if ax is None:
		ax = plt.gca()
	for dc, fields, color in zip(dcs, dc_fields, colors):
		latent = dc.request('latent_means')
		for i in range(latent.shape[1]):
			latent[:,i] = zscore(latent[:,i])
		pca = PCA(n_components=latent.shape[1], whiten=False)
		latent = pca.fit_transform(latent)
		var = [0] + np.cumsum(pca.explained_variance_ratio_).tolist()
		ax.plot(np.arange(latent.shape[1]+1), var, c=color, alpha=alpha, lw=lw)

		field_data = []
		for field in fields:
			data = zscore(dc.request(field))
			if not np.isnan(data).any():
				field_data.append(data)
		field_data = np.stack(field_data).T
		pca = PCA(n_components=field_data.shape[1], whiten=False)
		field_data = pca.fit_transform(field_data)
		var = [0] + np.cumsum(pca.explained_variance_ratio_).tolist()
		ax.plot(np.arange(field_data.shape[1]+1), var, ls='--', c=color, alpha=alpha, lw=lw)

	ax.set_title("Cumulative Feature Variance\nExplained by Feature Set", fontsize=8)
	ax.set_ylabel("Portion of Feature\nVariance Explained", fontsize=7)
	ax.set_xlabel("Number of Principal Components", fontsize=7)
	# plt.legend(loc='lower right')
	for y in [0.25,0.5,0.75]:
		ax.axhline(y=y, c='k', ls='-', lw=0.9, alpha=0.5)
	ax.set_xticks([0,2,4,6,8,10,12])
	ax.set_yticks([0.0,1.0])
	ax.set_yticklabels(['0','1'])
	# ax.axhline(y=0.95, c='r', lw=0.5, alpha=0.7)
	ax.set_xlim(0,12)
	ax.set_ylim(0,1)
	ax.plot([-1,-1], ls='--', c='k', lw=2, alpha=0.9, label='Traditional Features')
	ax.plot([-1,-1], ls='-', c='k', lw=2, alpha=0.9, label='Latent Features')
	ax.legend(loc='lower right', fontsize=7)

	if save_and_close:
		plt.savefig(os.path.join(dc.plots_dir, filename))
		plt.close('all')


def _boxplot_DC(colors, hatch='o', ax=None, save_and_close=True, \
	filename='correlation_boxplot.pdf'):
	"""
	Pairs of boxplots.
	"""
	d = np.load('temp_data/bar_chart_data.npy', allow_pickle=True).item()
	trad_r2s, latent_r2s = d['trad_r2s'], d['latent_r2s']
	trad_names, latent_names = d['trad_names'], d['latent_names']

	trad_colors, latent_colors = d['trad_colors'], d['latent_colors']
	trad_colors = np.array([colors.index(i) for i in trad_colors])
	latent_colors = np.array([colors.index(i) for i in latent_colors])

	x = 0
	A = [[] for i in range(2*len(colors))]
	for i in range(len(colors)):

		for j in range(len(latent_r2s)):
			if latent_colors[j] == i:
				A[2*i].append(latent_r2s[j])

		for j in range(len(trad_r2s)):
			if trad_colors[j] == i:
				A[2*i+1].append(trad_r2s[j])

	if ax is None:
		ax = plt.gca()
	props = {'color': to_rgba('k', alpha=1.0), 'linewidth': 1.2}
	bp1 = ax.boxplot([A[0],A[1]], positions=[0,1], patch_artist=True, \
		widths=0.6, medianprops=props, whiskerprops=props, capprops=props)
	bp2 = ax.boxplot([A[2],A[3]], positions=[2.5,3.5], patch_artist=True, \
		widths=0.6, medianprops=props, whiskerprops=props, capprops=props)
	bp3 = ax.boxplot([A[4],A[5]], positions=[5,6], patch_artist=True, \
		widths=0.6, medianprops=props, whiskerprops=props, capprops=props)
	for bp in [bp1, bp2, bp3]:
		bp['boxes'][1].set_hatch(hatch)
	for bplot, color in zip((bp1, bp2, bp3), colors):
		color = to_rgba(color, alpha=0.7)
		for patch in bplot['boxes']:
			patch.set_facecolor(color)
			patch.set_edgecolor(to_rgba('k'))
			patch.set_linewidth(1.5)
	for direction in ['top', 'right', 'bottom']:
		ax.spines[direction].set_visible(False)
	ax.text(0.505, 0.0, 'MUPET', c=colors[0], ha='center', fontsize=7)
	ax.text(3.05, 0.0, 'DeepSqueak', c=colors[1], ha='center', fontsize=7)
	ax.text(5.55, 0.0, 'SAP', c=colors[2], ha='center', fontsize=7)
	ax.set_title('Latent/Traditional Prediction Asymmetry', fontsize=8)
	ax.set_ylabel('Portion of Feature\nVariance Explained', fontsize=7)
	ax.set_yticks([0,0.25,0.5,0.75,1.0])
	ax.set_yticklabels([0,'','','',1])
	ax.set_ylim(0,1)
	ax.set_xticks([])
	if save_and_close:
		plt.savefig(filename)
		plt.close('all')

# def two_subplot_correlation_plot_DC(dcs, all_fields, colors=None, axs=None, \
# 	save_and_close=True, labels=None, filename='temp.pdf'):
# 	"""
# 	TO DO: shuffle scatter
# 	"""
# 	if colors is None:
# 		colors = ['k'] * len(dcs)
# 	# Collect data.
# 	plot_1_xs, plot_2_xs, plot_1_ys, plot_2_ys = [], [], [], []
# 	plot_1_colors, plot_2_colors = [], []
# 	for i, dc, fields in zip(range(len(dcs)), dcs, all_fields):
# 		latent = dc.request('latent_means')
# 		all_field_data = [dc.request(field) for field in fields]
# 		all_field_data = np.stack(all_field_data).T
# 		all_field_data = preprocessing.scale(all_field_data) # z-score
# 		pca = PCA()
# 		all_field_data = pca.fit_transform(all_field_data)
# 		variances = pca.explained_variance_ratio_
# 		for field_num in range(all_field_data.shape[1]):
# 			single_field = all_field_data[:,field_num].reshape(-1,1)
# 			reg = LinearRegression().fit(latent, single_field)
# 			plot_1_xs.append(variances[field_num])
# 			plot_1_ys.append(plot_1_xs[-1]*reg.score(latent, single_field))
# 			plot_1_colors.append(colors[i])
# 		pca = PCA()
# 		latent = pca.fit_transform(latent)
# 		variances = pca.explained_variance_ratio_
# 		for latent_num in range(latent.shape[1]):
# 			single_latent = latent[:,latent_num].reshape(-1,1)
# 			reg = LinearRegression().fit(all_field_data, single_latent)
# 			plot_2_xs.append(variances[latent_num])
# 			plot_2_ys.append(plot_2_xs[-1]*reg.score(all_field_data, single_latent))
# 			plot_2_colors.append(colors[i])
# 	# Plot.
# 	if axs is None:
# 		_, axs = plt.subplots(2,1)
# 	axs[0].scatter(plot_1_xs, plot_1_ys, c=plot_1_colors, alpha=0.6)
# 	axs[0].set_ylabel('Traditional Feature Variance')
# 	axs[0].set_xlabel('Variance Explained by Latent Features')
# 	max_val = max(plot_1_xs)
# 	axs[1].scatter(plot_2_xs, plot_2_ys, c=plot_2_colors, alpha=0.6)
# 	axs[1].set_ylabel('Latent Feature Variance')
# 	axs[1].set_xlabel('Variance Explained by Traditional Features')
# 	max_val = max(plot_2_xs)
# 	if labels is not None:
# 		patches = [mpatches.Patch(color=colors[i], label=labels[i]) \
# 				for i in range(len(labels))]
# 		axs[0].legend(handles=patches, loc='upper left') # bbox_to_anchor=(0.5, 0.5)
# 	if save_and_close:
# 		plt.savefig(filename)
# 		plt.close('all')


# def triptych_correlation_plot_DC(dcs, all_fields, colors=None, ax=None, \
# 	save_and_close=True, jitter=0.25, filename='triptych.pdf'):
# 	"""
# 	A correlation scatter plot.
#
# 	Three parts: 1) portion of traditional feature variance explained by linear
# 	combinations of latent features. 2) portion of 'used' latent feature
# 	variance explained by linear combinations of traditional features. 3)
# 	portion of unused latent feature variance explained by linear combinations
# 	of traditional features.
#
# 	Parameters
# 	----------
# 	dcs : ...
# 		....
#
# 	"""
# 	if colors is None:
# 		colors = ['k'] * len(dcs)
# 	# Collect data.
# 	plot_1_vars, plot_2_vars, plot_3_vars = [], [], []
# 	plot_1_colors, plot_2_colors, plot_3_colors = [], [], []
# 	for i, dc, fields in zip(range(len(dcs)), dcs, all_fields):
# 		latent = dc.request('latent_means')
# 		all_field_data = [dc.request(field) for field in fields]
# 		all_field_data = np.stack(all_field_data).T
# 		print("all field data", all_field_data.shape)
# 		print("latent", latent.shape)
# 		for field_num in range(all_field_data.shape[1]):
# 			field_data = all_field_data[:,field_num]
# 			var = _get_linear_r2(latent, field_data)
# 			plot_1_vars.append(var)
# 			plot_1_colors.append(colors[i])
# 		# CCA
# 		pca = PCA()
# 		latent = pca.fit_transform(latent)
# 		var_explained = np.cumsum(pca.explained_variance_ratio_)
# 		magic_index = np.searchsorted(var_explained, 0.98)
# 		for latent_dim in range(latent.shape[1]):
# 			latent_feature = latent[:,latent_dim]
# 			var = _get_linear_r2(all_field_data, latent_feature)
# 			if latent_dim < magic_index:
# 				plot_2_vars.append(var)
# 				plot_2_colors.append(colors[i])
# 			else:
# 				plot_3_vars.append(var)
# 				plot_3_colors.append(colors[i])
# 	# Plot.
# 	if ax is None:
# 		ax = plt.gca()
# 	np.random.seed(42)
# 	y_vals = jitter * np.random.rand(len(plot_1_vars))
# 	ax.scatter(y_vals, plot_1_vars, c=plot_1_colors, alpha=0.5)
# 	y_vals = 1.0 + jitter * np.random.rand(len(plot_2_vars))
# 	ax.scatter(y_vals, plot_2_vars, c=plot_2_colors, alpha=0.5)
# 	y_vals = 2.0 + jitter * np.random.rand(len(plot_3_vars))
# 	ax.scatter(y_vals, plot_3_vars, c=plot_3_colors, alpha=0.5)
# 	np.random.seed(None)
# 	if save_and_close:
# 		plt.savefig(filename)
# 		plt.close('all')


def _get_linear_r2(vals_1, vals_2):
	"""Get % variance of vals_2 explained by vals_1."""
	reg = LinearRegression().fit(vals_1, vals_2.reshape(-1,1))
	return reg.score(vals_1, vals_2.reshape(-1,1))


def _get_knn_r2(vals_1, vals_2, k=10, n_fold=5):
	"""Get % variance of vals_2 explained by vals_1."""
	# Do k-fold cross-validation to get average test set variance explained.
	k_fold = KFold(n_splits=n_fold, shuffle=True, random_state=42)
	r2 = 0.0
	for fold_num, (train, test) in enumerate(k_fold.split(vals_1, vals_2)):
		reg = KNeighborsRegressor(n_neighbors=k)
		reg.fit(vals_1[train], vals_2[train])
		r2 += (reg.score(vals_1[test], vals_2[test]) - r2) / (fold_num + 1)
	return r2



if __name__ == '__main__':
	pass



###
