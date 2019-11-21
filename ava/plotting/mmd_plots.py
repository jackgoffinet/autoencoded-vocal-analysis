"""
MMD plots.

Reference
---------
.. [1] Gretton, Arthur, et al. "A kernel two-sample test." Journal of Machine
	Learning Research 13.Mar (2012): 723-773.

	`<http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf>`_
"""
__date__ = "August-November 2019"


from itertools import repeat
from joblib import Parallel, delayed
import numpy as np
import os
from matplotlib.collections import PolyCollection
from matplotlib.colors import cnames
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE, MDS


# Define a list of random colors, excluding near-white colors.
NEAR_WHITE_COLORS = ['silver', 'whitesmoke', 'floralwhite', 'aliceblue', \
	'lightgoldenrodyellow', 'lightgray', 'w', 'seashell', 'ivory', \
	'lemonchiffon','ghostwhite', 'white', 'beige', 'honeydew', 'azure', \
	'lavender', 'snow', 'linen', 'antiquewhite', 'papayawhip', 'oldlace', \
	'cornsilk', 'lightyellow', 'mintcream', 'lightcyan', 'lavenderblush', \
	'blanchedalmond', 'lightcoral']
COLOR_LIST = []
for name, hex in cnames.items():
	if name not in NEAR_WHITE_COLORS:
		COLOR_LIST.append(name)
COLOR_LIST = np.array(COLOR_LIST)
np.random.seed(42)
np.random.shuffle(COLOR_LIST)
np.random.seed(None)


def mmd_matrix_plot_DC(dc, condition_from_fn, mmd_fn, condition_fn, \
	parallel=False, load_data=False, cluster=True, alg='quadratic', max_n=None,\
	sigma=None, cmap='Greys', colorbar=True, cax=None, ticks=[0.0,0.3], \
	filename='mmd_matrix.pdf', ax=None, save_and_close=True):
	"""
	Plot a pairwise MMD matrix.

	Parameters
	----------
	dc : ava.data.data_container.DataContainer
		DataContainer object.
	condition_from_fn : function
		Returns an int representing condition, given a filename.
	mmd_fn : str
		Where MMD values are saved to/loaded from.
	condition_fn : str
		Where conditions are saved to/loaded from.
	parallel : bool, optional
		Whether to calculate different MMD values in parallel. If ``True``, MMD
		values are printed out to stdout and can then be saved and formed into a
		proper matrix using the ``_matrix_from_txt`` helper function.
	load_data : bool, optional
		Whether to load precomputed data. Defaults to ``False``.
	cluster : bool, optional
		Whether to order conditions by a clustering algorithm. Defaults to
		``True``.
	alg : {``'linear'``, ``'quadratic'``}, optional
		Use the linear-time or quadratic time MMD estimate. Defaults to
		``'quadratic'``.
	max_n : int or ``None``, optional
		Maximum number of samples from each distribution. If ``None``, no
		maximum is set. Only applies if ``alg == 'quadratic'``. Defaults to
		``None``.
	sigma : {float, None}, optional
		Kernel bandwidth. If ``None``, the median distance is used. Defaults to
		``None``.
	cmap : str, optional
		Name of matplotlib colormap. Defaults to ``'viridis'``.
	colorbar : bool, optional
		Whether to plot a colorbar. Defaults to ``True``.
	cax : matplotlib.axes._subplots.AxesSubplot or ``None``, optional
		Colorbar axis. If ``None``, a new axis is made. Defaults to ``None``.
	ticks : list of floats, optional
		Colorbar ticks. Defaults to ``[0.0, 0.3]``.
	filename : str, optional
		Where to save plot, relative to ``dc.plots_dir``. Defaults to
		``'mmd_matrix.pdf'``.
	ax : matplotlib.axes._subplots.AxesSubplot, optional
		Matplotlib axis. Defaults to the current axis, ``plt.gca()``.
	save_and_close : bool, optional
		Whether to save and close the plot. Defaults to ``True``.
	"""
	assert mmd_fn is not None
	loaded = False
	if load_data:
		try:
			mmd = np.load(mmd_fn)
			loaded = True
		except:
			print("Unable to load data!")
	if not loaded:
		mmd, _ = _calculate_mmd(dc, condition_from_fn, parallel=parallel, \
				alg=alg, max_n=max_n, sigma=sigma, mmd_fn=mmd_fn, \
				condition_fn=condition_fn)
	filename = os.path.join(dc.plots_dir, filename)
	mmd_matrix_plot(mmd, ax=ax, save_and_close=save_and_close, \
			cluster=cluster, cmap=cmap, filename=filename, colorbar=colorbar, \
			cax=cax, ticks=ticks)


def mmd_matrix_plot(mmd, cluster=True, cmap='viridis', ax=None, \
	colorbar=True, cax=None, ticks=[0.0,0.3], filename='mmd_matrix.pdf', \
	save_and_close=True):
	"""
	Plot a pairwise MMD matrix.

	Parameters
	----------
	mmd : numpy.ndarray
		Pairwise MMD values, a square matrix.
	cluster : bool, optional
		Whether to order conditions by a clustering algorithm. Defaults to
		``True``.
	cmap : str, optional
		Name of matplotlib colormap. Defaults to ``'viridis'``.
	ax : matplotlib.axes._subplots.AxesSubplot, optional
		Matplotlib axis. Defaults to the current axis, ``plt.gca()``.
	colorbar : bool, optional
		Whether to plot a colorbar. Defaults to ``True``.
	cax : matplotlib.axes._subplots.AxesSubplot or ``None``, optional
		Colorbar axis. If ``None``, a new axis is made. Defaults to ``None``.
	ticks : list of floats, optional
		Colorbar ticks. Defaults to ``[0.0, 0.3]``.
	filename : str, optional
		Where to save plot. Defaults to ``'mmd_matrix.pdf'``.
	save_and_close : bool, optional
		Save and close the figure. Defaults to ``True``.
	"""
	mmd = np.clip(mmd, 0.0, None)
	if cluster:
		mmd = _cluster_matrix(mmd)
	if ax is None:
		ax = plt.gca()
	im = ax.imshow(mmd, cmap=cmap)
	ax.axis('off')
	if colorbar:
		fig = plt.gcf()
		cbar = fig.colorbar(im, cax=cax, fraction=0.046, \
			orientation="horizontal", ticks=ticks)
		cbar.solids.set_edgecolor("face")
		cbar.solids.set_rasterized(True)
		labels = ["{0:.1f}".format(round(tick,1)) for tick in ticks]
		cbar.ax.set_xticklabels(labels)
	if save_and_close:
		plt.savefig(filename)
		plt.close('all')


def mmd_tsne_plot_DC(dc, mmd_fn=None, condition_fn=None, mmd=None, \
	conditions=None, perplexity=30.0, s=4.0, alpha=0.8, label_func=None, \
	ax=None, save_and_close=True, filename='mmd_tsne.pdf', load_data=False):
	"""
	Compute and plot a t-SNE layout from an MMD matrix.

	Either pass ``mmd`` and ``conditions`` directly, or specify ``mmd_fn`` and
	``condition_fn`` and set ``load_data=True``.

	TO DO:
	-----
	* add option to calculate MMD.

	Parameters
	----------
	dc : ava.data.data_container.DataContainer
		DataContainer object.
	mmd_fn : str
		Where MMD values are saved to/loaded from.
	condition_fn : str
		Where conditions are saved to/loaded from.
	mmd : {numpy.ndarray, None}, optional
		MMD matrix. Defaults to ``None``.
	conditions : {numpy.ndarray, None}, optional
		Condition for each entry of the MMD array. Defaults to ``None``.
	perplexity : float, optional
		Passed to t-SNE. Defaults to ``30.0``.
	s : float, optional
		Passed to ``matplotlib.pyplot.scatter``. Defaults to ``4.0``.
	alpha : float, optional
		Passed to ``matplotlib.pyplot.scatter``. Defaults to ``0.8``.
	label_func : {function, None}, optional
		Maps a conditions to a label (string) for annotating points. Defaults
		to ``None``.
	ax : matplotlib.axes._subplots.AxesSubplot, optional
		Matplotlib axis. Defaults to the current axis, ``plt.gca()``.
	save_and_close : bool, optional
		Save and close the figure. Defaults to ``True``.
	filename : str, optional
		Where to save plot. Defaults to ``'mmd_tsne.pdf'``.
	load_data : bool, optional
		Whether to load the MMD and condition data from ``mmd_fn`` and
		``condition_fn``. Defaults to ``False``.
	"""
	if load_data:
		assert mmd_fn is not None and condition_fn is not None
		try:
			mmd = np.load(mmd_fn)
			conditions = np.load(condition_fn)
		except:
			print("Unable to load data!")
			return
	else:
		assert mmd is not None and conditions is not None
	mmd = np.clip(mmd, 0, None)
	all_conditions = list(np.unique(conditions)) # np.unique sorts things
	colors = [COLOR_LIST[i%len(COLOR_LIST)] for i in conditions]
	all_colors = [COLOR_LIST[i%len(COLOR_LIST)] for i in all_conditions]
	transform = TSNE(n_components=2, random_state=42, metric='precomputed', \
			method='exact', perplexity=perplexity)
	embed = transform.fit_transform(mmd)
	if ax is None:
		ax = plt.gca()
	poly_colors = []
	poly_vals = []
	for i in range(len(conditions)-1):
		for j in range(i+1, len(conditions)):
			if conditions[i] == conditions[j]:
				color = to_rgba(colors[i], alpha=0.7)
				ax.plot([embed[i,0],embed[j,0]], [embed[i,1],embed[j,1]], \
					c=color, lw=0.5)
				for k in range(j+1, len(conditions)):
					if conditions[k] == conditions[j]:
						arr = np.stack([embed[i], embed[j], embed[k]])
						poly_colors.append(to_rgba(colors[i], alpha=0.2))
						poly_vals.append(arr)
	pc = PolyCollection(poly_vals, color=poly_colors)
	ax.add_collection(pc)
	ax.scatter(embed[:,0], embed[:,1], color=colors, s=s, alpha=alpha)
	if label_func is not None:
		for i in range(len(embed)):
			ax.annotate(label_func(conditions[i]), embed[i])
	plt.axis('off')
	if save_and_close:
		plt.savefig(os.path.join(dc.plots_dir, filename))
		plt.close('all')


# def mmd_mds_plot_DC(dc, mmd_fn=None, condition_fn=None, mmd=None, \
# 	conditions=None, s=4.0, alpha=0.8, label_func=None, metric=True, \
# 	ax=None, save_and_close=True, filename='mmd_mds.pdf', load_data=True):
# 	"""
# 	Compute and plot an MDS layout from an MMD matrix.
#
# 	Note
# 	----
# 	* add option to calculate MMD.
#
# 	Parameters
# 	----------
# 	dc : ava.data.data_container.DataContainer
# 		DataContainer object.
# 	mmd_fn : str
# 		Where MMD values are saved to/loaded from.
# 	condition_fn : str
# 		Where conditions are saved to/loaded from.
# 	"""
# 	if load_data:
# 		assert mmd_fn is not None and condition_fn is not None
# 		try:
# 			mmd = np.load(mmd_fn)
# 			conditions = np.load(condition_fn)
# 		except:
# 			print("Unable to load data!")
# 			return
# 	else:
# 		assert mmd is not None and conditions is not None
# 	mmd = np.clip(mmd, 0, None)
# 	all_conditions = list(np.unique(conditions)) # np.unique sorts things
# 	colors = [COLOR_LIST[i%len(COLOR_LIST)] for i in conditions]
# 	all_colors = [COLOR_LIST[i%len(COLOR_LIST)] for i in all_conditions]
# 	transform = MDS(n_components=2, random_state=42, metric=metric, \
# 			dissimilarity='precomputed')
# 	embed = transform.fit_transform(mmd)
# 	if ax is None:
# 		ax = plt.gca()
# 	poly_colors = []
# 	poly_vals = []
# 	for i in range(len(conditions)-1):
# 		for j in range(i+1, len(conditions)):
# 			if conditions[i] == conditions[j]:
# 				color = to_rgba(colors[i], alpha=0.7)
# 				ax.plot([embed[i,0],embed[j,0]], [embed[i,1],embed[j,1]], \
# 					c=color, lw=0.5)
# 				for k in range(j+1, len(conditions)):
# 					if conditions[k] == conditions[j]:
# 						arr = np.stack([embed[i], embed[j], embed[k]])
# 						poly_colors.append(to_rgba(colors[i], alpha=0.2))
# 						poly_vals.append(arr)
# 	pc = PolyCollection(poly_vals, color=poly_colors)
# 	ax.add_collection(pc)
# 	ax.scatter(embed[:,0], embed[:,1], color=colors, s=s, alpha=alpha)
# 	if label_func is not None:
# 		for i in range(len(embed)):
# 			ax.annotate(label_func(conditions[i]), embed[i])
# 	plt.axis('off')
# 	if save_and_close:
# 		plt.savefig(os.path.join(dc.plots_dir, filename))
# 		plt.close('all')


def _estimate_mmd2(latent, i1, i2, sigma=None, max_n=None):
	"""From Gretton et. al. 2012"""
	if sigma is None:
		sigma = estimate_median_sigma(latent)
	A = -0.5 / sigma
	m, n = len(i1), len(i2)
	if max_n is not None:
		m, n = min(max_n,m), min(max_n,n)
		if m < len(i1):
			np.random.shuffle(i1)
			i1 = i1[:m]
		if n < len(i2):
			np.random.shuffle(i2)
			i2 = i2[:n]
	term_1 = 0.0
	for i in range(m):
		for j in range(m):
			if j == i:
				continue
			dist = np.sum(np.power(latent[i1[i]] - latent[i1[j]], 2))
			term_1 += np.exp(A * dist)
	term_1 *= 1/(m*(m-1))
	term_2 = 0.0
	for i in range(n):
		for j in range(n):
			if j == i:
				continue
			dist = np.sum(np.power(latent[i2[i]] - latent[i2[j]], 2))
			term_2 += np.exp(A * dist)
	term_2 *= 1/(n*(n-1))
	term_3 = 0.0
	for i in range(m):
		for j in range(n):
			dist = np.sum(np.power(latent[i1[i]] - latent[i2[j]], 2))
			term_3 += np.exp(A * dist)
	term_3 *= 2/(m*n)
	return term_1 + term_2 - term_3


def _estimate_mmd2_linear_time(latent, i1, i2, sigma=None):
	"""From Gretton et. al. 2012"""
	if sigma is None:
		sigma = estimate_median_sigma(latent)
	A = -0.5 / sigma
	n = min(len(i1), len(i2))
	m = n // 2
	assert m > 0
	k = lambda x,y: np.exp(A * np.sum(np.power(x-y,2)))
	h = lambda x1,y1,x2,y2: k(x1,x2)+k(y1,y2)-k(x1,y2)-k(x2,y1)
	term = 0.0
	for i in range(m):
		term += h(latent[i1[2*i]], latent[i2[2*i]], latent[i1[2*i+1]], \
			latent[i2[2*i+1]])
	return term / m


def _cluster_matrix(matrix, index=None):
	"""Order entries by a clustering dendrogram."""
	if index is None:
		index = len(matrix) // 2
	flat_dist1 = squareform(matrix[:index,:index])
	Z1 = linkage(flat_dist1, optimal_ordering=True)
	leaves1 = leaves_list(Z1)

	flat_dist2 = squareform(matrix[index:,index:])
	Z2 = linkage(flat_dist2, optimal_ordering=True)
	leaves2 = leaves_list(Z2) + index

	leaves = np.concatenate([leaves1, leaves2])
	new_matrix = np.zeros_like(matrix)
	for i in range(len(matrix)-1):
		for j in range(i,len(matrix)):
			temp = matrix[leaves[i],leaves[j]]
			new_matrix[i,j] = temp
			new_matrix[j,i] = temp
	return new_matrix


def _calculate_mmd(dc, condition_from_fn, parallel=False, alg='linear', \
	max_n=None, sigma=None, mmd_fn=None, condition_fn=None):
	assert alg in ['linear', 'quadratic']
	# Collect.
	latent = dc.request('latent_means')
	audio_fns = dc.request('audio_filenames')
	condition = np.array([condition_from_fn(str(i)) for i in audio_fns], \
			dtype='int')
	all_conditions = np.unique(condition) # np.unique sorts things
	n = len(all_conditions)
	result = np.zeros((n,n))
	print("Number of conditions:", n)
	if sigma is None:
		sigma = estimate_median_sigma(latent)
	print("Sigma squared:", sigma)
	if parallel:
		i_vals, j_vals = [], []
		for i in range(n-1):
			for j in range(i+1,n):
				i_vals.append(i)
				j_vals.append(j)
		gen = zip(i_vals, j_vals, repeat(condition), repeat(all_conditions), \
			repeat(alg), repeat(latent), repeat(sigma), \
			repeat(max_n))
		n_jobs = os.cpu_count()
		# Calculate.
		temp_results = Parallel(n_jobs=n_jobs)(delayed(_mmd_helper)(*args) for \
				args in gen)
		for i, j, mmd in temp_results:
			result[i,j] = mmd
			result[j,i] = mmd
	else:
		for i in range(n-1):
			for j in range(i+1, n):
				i1 = np.argwhere(condition == all_conditions[i]).flatten()
				i2 = np.argwhere(condition == all_conditions[j]).flatten()
				if alg == 'linear':
					temp = _estimate_mmd2_linear_time(latent, i1, i2, \
								sigma=sigma)
				elif alg == 'quadratic':
					temp = _estimate_mmd2(latent, i1, i2, \
								sigma=sigma, max_n=max_n)
				result[i,j] = temp
				result[j,i] = temp
	# Save.
	np.save(mmd_fn, result)
	np.save(condition_fn, all_conditions)
	return result, all_conditions


def _mmd_helper(i, j, condition, all_conditions, alg, latent, sigma, \
	max_n):
	"""Helper to make this parallelized."""
	i1 = np.argwhere(condition == all_conditions[i]).flatten()
	i2 = np.argwhere(condition == all_conditions[j]).flatten()
	if alg == 'linear':
		mmd = _estimate_mmd2_linear_time(latent, i1, i2, \
				sigma=sigma)
	else:
		mmd = _estimate_mmd2(latent, i1, i2, \
				sigma=sigma, max_n=max_n)
	print(i, j, mmd, flush=True)
	return i, j, mmd


def estimate_median_sigma(latent, n=10000):
	"""
	Estimate the median pairwise distance for use as a kernel bandwidth.

	Parameters
	----------
	latent : numpy.ndarray
		Latent means.
	n : int, optional
		Number of random pairs to draw. Defaults to `10000`.

	Returns
	-------
	sigma : float
		Median pairwise euclidean distance between sampled latent means.
	"""
	arr = np.zeros(n)
	for i in range(n):
		i1, i2 = np.random.randint(len(latent)), np.random.randint(len(latent))
		arr[i] = np.sum(np.power(latent[i1]-latent[i2],2))
	return np.median(arr)


def _matrix_from_txt(text_fn):
	"""Read a text file into an MMD matrix."""
	i_s, j_s, mmds = np.loadtxt(text_fn, delimiter=' ', unpack=True)
	n = int(round(max(np.max(i_s), np.max(j_s)))) + 1
	mmd_matrix = np.zeros((n,n))
	for i, j, mmd in zip(i_s, j_s, mmds):
		mmd_matrix[int(i), int(j)] = mmd
		mmd_matrix[int(j), int(i)] = mmd
	return mmd_matrix



if __name__ == '__main__':
	pass



###
