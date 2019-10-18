"""
MMD plots.

http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf

"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"


from itertools import repeat
from joblib import Parallel, delayed
import numpy as np
import os
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE


# For MUPET sample recordings
C57 = [3079, 3081, 3085, 3087, 3157, 3158, 3160, 3166, 3251, 3252, 3253, 3254, \
	3255, 3257, 3258, 3259, 3532, 3535, 9870, 9874]
DBA = [3070, 3074, 3168, 3170, 3171, 3172, 3240, 3241, 3243, 3244, 3245, 3246, \
	3247, 3248, 3249, 9856, 9857, 9858, 9859, 9863]
ALL_RECORDINGS = C57 + DBA

BAD_COLORS = ['silver', 'whitesmoke', 'floralwhite', 'aliceblue', \
	'lightgoldenrodyellow', 'lightgray', 'w', 'seashell', 'ivory', \
	'lemonchiffon','ghostwhite', 'white', 'beige', 'honeydew', 'azure', \
	'lavender', 'snow', 'linen', 'antiquewhite', 'papayawhip', 'oldlace', \
	'cornsilk', 'lightyellow', 'mintcream', 'lightcyan', 'lavenderblush', \
	'blanchedalmond', 'lightcoral']

from matplotlib.colors import cnames
color_list = []
for name, hex in cnames.items():
	if name not in BAD_COLORS:
		color_list.append(name)
color_list = np.array(color_list)
np.random.seed(42)
np.random.shuffle(color_list)
np.random.seed(None)


def mmd_matrix_DC(dc, condition_from_fn, load_data=False, ax=None, \
	save_and_close=True, divider=None, cluster=True, alg='linear', max_n=None, \
	cmap='viridis', filename='mmd_matrix.pdf', divider_color='white', \
	save_load_fns=['result_mmd_matrix.npy', 'all_conditions.npy'], \
	colorbar=True, cax=None):
	"""
	Parameters
	----------
	dc : ...
		...
	alg : {``'linear'``, ``'quadratic'``}, optional
		Deafults to ``'linear'``.

	"""
	loaded = False
	if load_data:
		try:
			result = np.load('temp_data/'+save_load_fns[0])
			loaded = True
		except:
			print("Unable to load data!")
	if not loaded:
		result, _ = _calculate_mmd(dc, condition_from_fn, alg=alg, max_n=max_n,\
				save_fns=save_load_fns)
	result = np.clip(result, 0, None)
	if cluster:
		result = _cluster_matrix(result)
	if ax is None:
		ax = plt.gca()
	im = ax.imshow(result, cmap=cmap)
	ax.axis('off')
	if divider is None:
		divider = len(result) // 2
	if divider > 0:
		ax.axhline(y=divider-0.5, c=divider_color, lw=0.9)
		ax.axvline(x=divider-0.5, c=divider_color, lw=0.9)
	if colorbar:
		min_val, max_val = 0, np.max(result)
		print("min/max", np.min(result), np.max(result))
		ticks = [min_val, max_val]
		fig = plt.gcf()
		cbar = fig.colorbar(im, cax=cax, fraction=0.046, \
			orientation="horizontal", ticks=[0,0.3])
		cbar.solids.set_edgecolor("face")
		cbar.solids.set_rasterized(True)
		cbar.ax.set_xticklabels(['0.0', '0.3'])
	if save_and_close:
		plt.savefig(os.path.join(dc.plots_dir, filename))
		plt.close('all')


def mmd_tsne_DC(dc, condition_from_fn, load_data=False, ax=None, alg='linear',\
	max_n=None, s=4.0, alpha=0.8, save_and_close=True, filename='mmd_tsne.pdf', \
	save_load_fns=['result_mmd_tsne.npy', 'all_conditions.npy']):
	"""
	Compute and plot a t-SNE layout from an MMD matrix.

	Parameters
	----------
	dc : ...
		...

	"""
	loaded = False
	if load_data:
		try:
			result = np.load('temp_data/'+save_load_fns[0])
			conditions = np.load('temp_data/'+save_load_fns[1])
			loaded = True
		except:
			print("Unable to load data!")
	if not loaded:
		result, conditions = _calculate_mmd(dc, condition_from_fn, alg=alg, \
				max_n=max_n, save_fns=save_load_fns)
	result = np.clip(result, 0, None)
	conditions = list(np.unique(conditions)) # np.unique sorts things
	identities = np.array([c//100 for c in conditions])
	colors = [color_list[i%len(color_list)] for i in identities]
	transform = TSNE(n_components=2, random_state=42, metric='precomputed')
	embed = transform.fit_transform(result)

	if ax is None:
		ax = plt.gca()

	poly_colors = []
	poly_vals = []
	for i in range(len(identities)-1):
		for j in range(i+1, len(identities)):
			if identities[i] == identities[j]:
				color = to_rgba(colors[i], alpha=0.7)
				ax.plot([embed[i,0],embed[j,0]], [embed[i,1],embed[j,1]], \
					c=color, lw=0.5)
				for k in range(j+1, len(identities)):
					if identities[k] == identities[j]:
						arr = np.stack([embed[i], embed[j], embed[k]])
						poly_colors.append(to_rgba(colors[i], alpha=0.2))
						poly_vals.append(arr)
	pc = PolyCollection(poly_vals, color=poly_colors)
	ax.add_collection(pc)
	ax.scatter(embed[:,0], embed[:,1], color=colors, s=s, alpha=alpha)
	plt.axis('off')
	if save_and_close:
		plt.savefig(os.path.join(dc.plots_dir, filename))
		plt.close('all')


def _estimate_mmd2(latent, i1, i2, sigma_squared=None, max_n=None):
	"""
	From Gretton et. al. 2012
	"""
	if sigma_squared is None:
		sigma_squared = _estimate_median_sigma_squared(latent)
	A = -0.5 / sigma_squared
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


def _estimate_mmd2_linear_time(latent, i1, i2, sigma_squared=None):
	"""
	From Gretton et. al. 2012
	"""
	if sigma_squared is None:
		sigma_squared = _estimate_median_sigma_squared(latent)
	A = -0.5 / sigma_squared
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
	print(leaves)
	new_matrix = np.zeros_like(matrix)
	for i in range(len(matrix)-1):
		for j in range(i,len(matrix)):
			temp = matrix[leaves[i],leaves[j]]
			new_matrix[i,j] = temp
			new_matrix[j,i] = temp
	return new_matrix


def _calculate_mmd(dc, condition_from_fn, alg='linear', max_n=None, \
	save_fns=['result.npy', 'conditions.npy']):
	# Collect
	latent = dc.request('latent_means')
	audio_fns = dc.request('audio_filenames')
	condition = np.array([condition_from_fn(str(i)) for i in audio_fns], \
			dtype='int')
	# Calculate.
	all_conditions = np.unique(condition) # np.unique sorts things
	n = len(all_conditions)
	result = np.zeros((n,n))
	print("n=", n)
	sigma_squared = _estimate_median_sigma_squared(latent)

	# NOTE: HERE!
	i_vals, j_vals = [], []
	for i in range(n-1):
		for j in range(i+1,n):
			i_vals.append(i)
			j_vals.append(j)

	gen = zip(i_vals, j_vals, repeat(condition), repeat(all_conditions), \
		repeat(result), repeat(alg), repeat(latent), repeat(sigma_squared), \
		repeat(max_n), list(range(len(i_vals))))
	n_jobs = os.cpu_count()
	Parallel(n_jobs=n_jobs)(delayed(_mmd_helper)(*args) for args in gen)

	np.save('temp_data/'+save_fns[0], result)
	np.save('temp_data/'+save_fns[1], all_conditions)
	return result, all_conditions


def _mmd_helper(i, j, condition, all_conditions, result, alg, latent, \
	sigma_squared, max_n, iteration):
	"""Helper to make this parallelized."""
	i1 = np.argwhere(condition == all_conditions[i]).flatten()
	i2 = np.argwhere(condition == all_conditions[j]).flatten()
	if alg == 'linear':
		mmd = _estimate_mmd2_linear_time(latent, i1, i2, \
				sigma_squared=sigma_squared)
	else:
		mmd = _estimate_mmd2(latent, i1, i2, \
				sigma_squared=sigma_squared, max_n=max_n)
	print(i, j, mmd, flush=True)
	result[i,j] = mmd
	result[j,i] = mmd


def _estimate_median_sigma_squared(latent, n=4000):
	arr = np.zeros(n)
	for i in range(n):
		i1, i2 = np.random.randint(len(latent)), np.random.randint(len(latent))
		arr[i] = np.sum(np.power(latent[i1]-latent[i2],2))
	return np.median(arr)


def _matrix_from_txt(text_fn):
	"""

	"""
	i_s, j_s, mmds = np.loadtxt(text_fn, delimiter=' ', unpack=True)
	n = int(round(max(np.max(i_s), np.max(j_s)))) + 1
	result = np.zeros((n,n))
	for i, j, mmd in zip(i_s, j_s, mmds):
		result[int(i), int(j)] = mmd
		result[int(j), int(i)] = mmd
	return result


# # For MUPET sample recordings
# def _condition_from_fn(fn):
# 	return ALL_RECORDINGS.index(int(fn.split('/')[-1].split('.')[0]))



if __name__ == '__main__':
	pass



###
