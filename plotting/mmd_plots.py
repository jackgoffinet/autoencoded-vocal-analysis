"""
MMD plots.

"""
import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')


C57 = [3079, 3081, 3085, 3087, 3157, 3158, 3160, 3166, 3251, 3252, 3253, 3254, \
	3255, 3257, 3258, 3259, 3532, 3535, 9870, 9874]

DBA = [3070, 3074, 3168, 3170, 3171, 3172, 3240, 3241, 3243, 3244, 3245, 3246, \
	3247, 3248, 3249, 9856, 9857, 9858, 9859, 9863]

ALL_RECORDINGS = C57 + DBA


def mmd_matrix_DC(dc, filename='mmd_matrix.png'):
	"""
	Paramters
	---------

	"""
	# Collect
	latent = dc.request('latent_means')
	print("read latent")
	audio_fns = dc.request('audio_filenames')
	print("read filenames")
	condition = np.array([condition_from_fn(str(i)) for i in audio_fns], dtype='int')
	# Calculate.
	all_conditions = np.unique(condition)
	n = len(all_conditions)
	result = np.zeros((n,n))
	print("n=", n)
	for i in range(n-1):
		print(i)
		for j in range(i+1,n):
			i1 = np.argwhere(condition == all_conditions[i]).flatten()
			i2 = np.argwhere(condition == all_conditions[j]).flatten()
			mmd = get_mmd(latent, i1, i2)
			result[i,j] = mmd
			result[j,i] = mmd
			np.save('result.npy', result)
	np.save('result.npy', result)
	plt.imshow(result)
	plt.savefig(os.path.join(dc.plots_dir, filename))
	plt.close('all')


def get_mmd(latent, i1, i2, bandwidth=1.5):
	"""

	"""
	k = -0.5 * bandwidth
	m, n = len(i1), len(i2)
	term_1 = 0.0
	for i in range(m):
		for j in range(m):
			if j == i:
				continue
			dist = np.sum(np.power(latent[i1[i]] - latent[i1[j]], 2))
			term_1 += np.exp(k * dist)
	term_1 *= 1/(m*(m-1))
	term_2 = 0.0
	for i in range(n):
		for j in range(n):
			if j == i:
				continue
			dist = np.sum(np.power(latent[i2[i]] - latent[i2[j]], 2))
			term_2 += np.exp(k * dist)
	term_2 *= 1/(n*(n-1))
	term_3 = 0.0
	for i in range(m):
		for j in range(n):
			dist = np.sum(np.power(latent[i1[i]] - latent[i2[j]], 2))
			term_3 += np.exp(k * dist)
	term_3 *= 2/(m*n)
	return term_1 + term_2 - term_3



def condition_from_fn(fn):
	return ALL_RECORDINGS.index(int(fn.split('/')[-1].split('.')[0]))



def make_g():
	d = np.load('result.npy')
	plt.imshow(d)
	plt.axvline(x=19.5, c='darkorange', lw=1)
	plt.axhline(y=19.5, c='darkorange', lw=1)
	plt.text(6,-1,'C57BL/6')
	plt.text(26.5,-1,'DBA/2')
	plt.text(-8,10,'C57BL/6')
	plt.text(-6,30,'DBA/2')
	plt.axis('off')
	plt.savefig('temp.pdf')
	plt.close('all')


def make_g_2():
	d = np.load('result.npy')
	# import umap
	# transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
	# 	metric='precomputed', random_state=42)
	from sklearn.manifold import TSNE
	transform = TSNE(n_components=2, metric='precomputed')
	embed = transform.fit_transform(d)
	c = ['r']*20 + ['b']*20
	plt.scatter(embed[:,0], embed[:,1], c=c, s=25.0)
	plt.axis('off')
	plt.savefig('temp_scatter.pdf')
	plt.close('all')


# def condition_from_fn(fn):
# 	"""
#
# 	"""
# 	mouse_num = int(fn.split('_')[0][2:])
# 	session_num = fn.split('_')[1]
# 	if 'day' in session_num:
# 		session_num = int(session_num[3:])
# 	elif 's' in session_num:
# 		session_num = int(session_num[1:])
# 	else:
# 		raise NotImplementedError
# 	return 100*mouse_num + session_num

# def get_label_from_condition(condition):
# 	return "M"+str()


if __name__ == '__main__':
	make_g()
	make_g_2()



###
