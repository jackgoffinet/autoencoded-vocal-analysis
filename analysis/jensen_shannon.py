"""
Estimate Jensen-Shannon distances.

D_KL(P||Q) = E_P log(P/Q)
D_JS(P||Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M)
where M = 0.5 * (P + Q)


TO DO:
	- Color code by strain.
"""
__author__ = "Jack Goffinet"
__date__ = "January - February 2019"

import numpy as np
from scipy.io import loadmat
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
plt.switch_backend('agg')


def helper(kde_1, kde_2, n=10**4):
	"""Helper function"""
	samples = kde_1.resample(size=n)
	log_p_1 = kde_1.logpdf(samples)
	log_p_2 = kde_2.logpdf(samples)
	log_sum = np.logaddexp(log_p_1, log_p_2)
	return np.mean(log_p_1) - np.mean(log_sum) + np.log(2.0)


def estimate_jsd(cloud_1, cloud_2, num_dims=5, n=3*10**4, pca=None):
	assert(num_dims <= min(cloud_1.shape[1], cloud_2.shape[1]))
	# First take the first <num_dims> principal components.
	if pca is None:
		pca = PCA(n_components=num_dims, random_state=0)
		pca.fit(np.concatenate((cloud_1, cloud_2), axis=0))
	cloud_1 = pca.transform(cloud_1)
	cloud_2 = pca.transform(cloud_2)
	# Make KDEs.
	kde_1 = gaussian_kde(cloud_1.T, bw_method='scott')
	kde_2 = gaussian_kde(cloud_2.T, bw_method='scott')
	# Calculate KL-divergences.
	jsd = helper(kde_1, kde_2, n=n)
	jsd += helper(kde_2, kde_1, n=n)
	jsd /= 2.0
	# jsd = np.sqrt(jsd) # divergence to distance.
	return jsd


def plot_jsd_matrix(d):
	# First fit PCA on all the data.
	pca = PCA(n_components=10, random_state=0)
	all_points = tuple(np.array(d[i]) for i in d)
	pca.fit(np.concatenate(all_points, axis=0))
	individuals = sorted(list(d.keys()))
	# result = np.zeros((len(individuals), len(individuals)))
	# for i in range(len(individuals)):
	# 	print(i)
	# 	cloud_1 = np.array(d[individuals[i]])
	# 	for j in range(i+1,len(individuals),1):
	# 		cloud_2 = np.array(d[individuals[j]])
	# 		jsd = estimate_jsd(cloud_1, cloud_2, pca=pca)
	# 		result[i,j] = jsd
	# 		result[j,i] = jsd
	# np.save('jsds.npy', result)
	result = np.load('jsds.npy')
	plt.imshow(result)
	plt.title('Estimated Jensen-Shannon Divergences')
	plt.xticks(list(range(5)), individuals)
	plt.colorbar()
	plt.yticks(list(range(5)), individuals)
	plt.savefig('temp.pdf')
	plt.close('all')


def split_latent_by_individual(load_filename, split_func):
	d = loadmat(load_filename)
	latent = d['latent']
	results = {}
	for i, filename in enumerate(d['filenames']):
		individual = split_func(filename)
		if individual in results:
			results[individual].append(latent[i])
		else:
			results[individual] = [latent[i]]
	return results


def mouse_filename_to_individual(filename):
	return filename.split('/')[-2]

def marmoset_filename_to_individual(filename):
	return filename.split('/')[-2]

if __name__ == '__main__':
	d = split_latent_by_individual('marmoset.mat', marmoset_filename_to_individual)
	plot_jsd_matrix(d)
	quit()
	# # MICE STUFF
	# Collect all the latent data.
	d = {}
	for fn in ['all_mice.mat', 'retest_mice.mat', 'female_mice.mat']:
		temp_d = split_latent_by_individual(fn, mouse_filename_to_individual)
		d = {**d, **temp_d}
	to_del = []
	for key in d:
		# print(key, len(d[key]))
		if len(d[key]) < 500:
			to_del.append(key)
			print("deleting", key)
	for key in to_del:
		del d[key]
	# quit()
	# # Fit the PCA.
	# pca = PCA(n_components=10, random_state=42)
	# pca.fit(loadmat('all_mice.mat')['latent'])

	# Define pairs to compare.

	females = ['Ai_14_female', 'RAm_2_fd', 'RAm_6_fd', 'RAm_female_2', 'VM_8_fd']
	males = [key for key in d.keys() if key not in females and 'retest' not in key]
	from itertools import combinations, product
	# pairs = list(combinations(females, 2))
	pairs = list(product(males, females))
	from random import shuffle
	shuffle(pairs)
	pairs = pairs[:20]

	# pairs = [
	# 	['TVA_17_28d_retest', 'TVA_17_fd'],
	# 	['TVA_18_29d_retest', 'TVA_18_fd'],
	# 	['TVA_19_35d_retest', 'TVA_19_fd'],
	# 	['TVA_27_7d_retest', 'TVA_27_fd'],
	# 	['TVA_27_14d_retest', 'TVA_27_fd'],
	# 	['TVA_27_28d_retest', 'TVA_27_fd'],
	# 	['TVA_27_7d_retest', 'TVA_27_14d_retest'],
	# 	['TVA_27_7d_retest', 'TVA_27_28d_retest'],
	# 	['TVA_27_14d_retest', 'TVA_27_28d_retest'],
	# 	['TVA_28_7d_retest', 'TVA_28_fd'],
	# 	['TVA_28_14d_retest', 'TVA_28_fd'],
	# 	['TVA_28_7d_retest', 'TVA_28_14d_retest'],
	# ]
	for pair in pairs:
		cloud_0 = np.array(d[pair[0]])
		cloud_1 = np.array(d[pair[1]])
		jsd = estimate_jsd(cloud_0, cloud_1, num_dims=10)
		print(jsd)

# Day/day pairs, 3d:
[0.036140903267919744,
0.07829119117198646,
0.06355225747408888,
0.18526231676745974,
0.17321206580723436,
0.22535376215286218,
0.018219258260628757,
0.029186389573726434,
0.03200255891959192,
0.071275093060292,
0.050751016095311186,
0.03881374134119964]

# Day/day pairs, 10d:
[0.2390934227723721,
0.3335377212457925,
0.3233390194808442,
0.42999947210827816,
0.4189072240154811,
0.4460354639534144,
0.22296848548795978,
0.2387838102515255,
0.2544091227538331,
0.29886403745471435,
0.2896319054820097,
0.22200991497795453]

# Male/male pairs, 3d:
[0.05334153811795017,
0.08124592308913747,
0.1319505184996449,
0.1294030776319074,
0.045404154347895465,
0.3094258353490166,
0.09024039688651275,
0.14048311858020102,
0.18790276375618575,
0.15208708491305323,
0.12358950367506927,
0.13139585182560187,
0.09427177407965315,
0.11514686180256384,
0.10011589934585918,
0.15053847871014925,
0.06088049711899657,
0.11416429984008925,
0.13467431773465954,
0.2578934433961525]

# Male/male pairs, 10d:
[0.35453717177432587,
0.6477139617934043,
0.43827705291990016,
0.262413410971582,
0.5749172418369163,
0.39379647689345354,
0.38286711686383856,
0.2746894837198516,
0.2876949804235832,
0.27520445284974515,
0.26460373879439414,
0.2797559424400945,
0.43758950799768737,
0.41821118714850136,
0.3041071053035854,
0.6241578149612895,
0.3595496661427905,
0.614520705143958,
0.5504382986966054,
0.5389990948465976]

# Female/female pairs, 10d:
[0.20027845001234573,
0.3666748913268417,
0.535135685302747,
0.5066794613332966,
0.3841650912424178,
0.4189464613650359,
0.4010543860545325,
0.46904497668421163,
0.4404849260582814,
0.55393408840857]

# Female/female pairs, 3d:
[0.20385252054007386,
0.030414578334260867,
0.23420284938391078,
0.22777705077094013,
0.14782747311693434,
0.21047612101554092,
0.16701974579211198,
0.15553898607485328,
0.26453240504066056,
0.1453484132044217]

# Male/Female, 3d:
[0.2792583810303947,
0.1426687172483706,
0.18429989887387765,
0.44950927357416837,
0.15059618139237385,
0.49299689898198495,
0.451944243191142,
0.49397180127251195,
0.17541952810560912,
0.12819316346852483,
0.3471932653728881,
0.3879517361676167,
0.43888696446309017,
0.1650695941210253,
0.40920368511648364,
0.31404870535342333,
0.46347269024183524,
0.36425010864893315,
0.47412162252695733,
0.23783338027953171]

# Male/Female, 10d:
[0.6678778431063036,
0.6230205673623789,
0.47509153363428147,
0.4957744617269809,
0.5539554277487769,
0.48694673981008696,
0.5135588075697676,
0.6216311503566573,
0.4390283232054698,
0.5114688572427556,
0.6606974360864786,
0.5887936647740543,
0.46494461660885544,
0.5801151838345583,
0.6648272018671109,
0.4390211550825932,
0.5555106100025277,
0.5964527396980881,
0.5142226887201139,
0.4606756667194193]


###
