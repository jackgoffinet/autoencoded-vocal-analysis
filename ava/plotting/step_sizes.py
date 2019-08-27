"""
Look at step sizes, conditioned on stuff.

"""
__author__ = "Jack Goffinet"
__date__ = "November 2018"

import numpy as np
import umap
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
plt.switch_backend('agg')



def get_embeddings_times(loader, model, alg='umap'):
	# First get latent representations.
	latent = model.get_latent(loader, n=10**9)
	# Then collect times.
	assert(len(latent) == len(loader.dataset))
	times = np.zeros(len(latent))
	i = 0
	for temp in loader:
		batch_times = temp['time'].detach().numpy()
		a  = np.min(batch_times)
		b  = np.max(batch_times)
		if b > 26:
			print('here', a, b)
			quit()
		times[i:i+len(batch_times)] = batch_times
		i += len(batch_times)
	perm = np.random.permutation(len(latent))
	latent = latent[perm]
	times = times[perm]
	# Fit UMAP on a random subset, get embedding.
	if alg == 'umap':
		transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean')
	elif alg == 'pca':
		transform = PCA(n_components=10)
	else:
		print('unidentified algorithm: ', alg)
		quit()
	print("fitting gif")
	transform.fit(latent[:9000])
	print("done")
	embeddings = transform.transform(latent)
	return embeddings, times



def make_step_size_plot(loader, model):
	xmin, xmax, ymin, ymax, tmin, tmax = -7, 15, -7, 15, 15, 26
	embeddings, times = get_embeddings_times(loader, model, alg='umap')
	pca_embeddings, _ = get_embeddings_times(loader, model, alg='pca')
	p = np.argsort(times)
	embeddings = embeddings[p]
	pca_embeddings = pca_embeddings[p]
	temp_embed = []
	for i, embedding in enumerate(embeddings):
		if embedding[1] > 6.:
			temp_embed.append(np.copy(pca_embeddings[i]))
	embeddings = np.array(temp_embed)
	distances = np.sqrt(np.sum(np.power(np.diff(embeddings, axis=0), 2), axis=1))
	X = np.array(range(len(distances))).reshape(-1,1)
	plt.scatter(range(len(distances)), distances, c='k', s=1, alpha=0.08)
	reg = LinearRegression().fit(X, distances)
	plt.plot([[0],[len(distances)]],reg.predict([[0],[len(distances)]]))
	plt.savefig('temp.pdf')
	plt.close('all')
