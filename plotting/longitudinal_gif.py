from __future__ import print_function, division
"""
Make a gif of birdsong trajectories.

"""
__author__ = "Jack Goffinet"
__date__ = "November 2018 - February 2019"


import numpy as np
import umap
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
plt.switch_backend('agg')


def get_conditions(loader, query='condition', n=10**10):
	n = min(n, len(loader.dataset))
	conditions = np.zeros(len(loader.dataset))
	for i, temp in enumerate(loader):
		i1 = min(n, i*len(temp[query]))
		i2 = min(n, (i+1)*len(temp[query]))
		conditions[i1:i2] = temp[query]
	return conditions


def get_mean_freqs(loader, n=10**4):
	results = []
	for temp in loader:
		specs = temp['image'].detach().cpu().numpy()
		freq_weights = np.arange(128)
		time_weights = np.sum(specs, axis=1)
		results += np.einsum('j,ik,ijk->i',freq_weights, time_weights, specs).tolist()
		if len(results) >= n:
			break
	results = results[:n]
	# results -= np.percentile(results, 5.0)
	# results /= np.percentile(results, 95.0)
	# results[results<0.0] = 0.0
	# results[results>1.0] = 1.0
	# results = 1.0 - results
	results -= np.mean(results)
	results /= -1.0*np.std(results)
	return results


def make_projection(loader, model, latent=None, title="", save_filename='temp.pdf'):
	# First get latent representations.
	if latent is None:
		latent, filenames = model.get_latent(loader, n=30000, random_subset=True, return_fields=['filename'])
	# cmap = plt.get_cmap('tab20')
	# print(filenames[:5])
	# quit()
	# colors = [cmap((hash(f.split('/')[-2])%256) / 256.0) for f in filenames]

	transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
	embedding = transform.fit_transform(latent)
	X, Y = embedding[:,0], embedding[:,1]
	plt.scatter(X, Y, c='b', alpha=0.1, s=0.7) #rgba_colors
	if len(title) > 0:
		plt.title(title)
	plt.axis('off')
	plt.savefig(save_filename)
	plt.close('all')


def generate_syllables(loader, model):
	"""Plot a projection of the latent space and generated syllables."""
	height, width = 4, 6
	# First get the latent projection used to define the transform.
	latent, times = model.get_latent(loader, n=30000, random_subset=True, return_times=True)
	transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
	print("fitting transform")
	transform.fit(latent)
	print("done")
	# Then get more latent transforms, this time to collect images.
	latent = model.get_latent(loader, n=10000)
	embedding = transform.transform(latent)
	# cmap = plt.cm.get_cmap('Set1', 3)
	# patches = [mpatches.Patch(color=cmap(i), label='BM00'+str(i+3)) for i in range(3)]
	# colors = []
	# individuals = []
	# for temp in loader:
	# 	individuals += temp['individual'].detach().cpu().numpy().tolist()
	# 	colors += cmap(temp['individual'].detach().cpu().numpy() - 3).tolist()
	x = embedding[:,0]
	y = embedding[:,1]
	plt.scatter(x, y, alpha=0.1, s=1)
	np.random.seed(42)
	indices = np.random.permutation(len(embedding))[:height*width]
	for i, index in enumerate(indices):
		plt.text(x[index], y[index], str(i))
	# plt.legend(handles=patches, loc='best')
	# plt.xlim((5,25))
	# plt.ylim((0,14))
	plt.axis('off')
	plt.savefig('temp.pdf')
	plt.close('all')
	# Plot spectrograms.
	gap = 10
	big_img = np.zeros((height*128+(height-1)*gap, width*128+(width-1)*gap))
	for i in range(height):
		for j in range(width):
			temp_im = loader.dataset[indices[i*width+j]]['image'].detach().cpu().numpy().reshape(128,128)
			big_img[i*(128+gap):i*(128+gap)+128,j*(128+gap):j*(128+gap)+128] = temp_im
	plt.imshow(big_img, aspect='auto', origin='lower', interpolation='none')
	for i in range(height):
		for j in range(width):
			plt.text(j*(128+gap), i*(128+gap), str(i*width+j), color='white')
	plt.axis('off')
	plt.savefig('temp1.pdf')
	plt.close('all')


def get_embeddings_times(loader, model, return_latent=False, return_images=False):
	# First get latent representations.
	if return_images:
		latent, times, images = model.get_latent(loader, n=30000, random_subset=True, return_times=True, return_images=True)
	else:
		latent, times = model.get_latent(loader, n=30000, random_subset=True, return_times=True, return_images=False)
	# perm = np.random.permutation(len(latent))
	# latent = latent[perm]
	# times = times[perm]
	# Fit UMAP on a random subset, get embedding.
	transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
	print("fitting transform")
	transform.fit(latent)
	print("done")
	embeddings = transform.transform(latent)
	if return_latent:
		if return_images:
			return embeddings, times, latent, images
		return embeddings, times, latent
	if return_images:
		return embeddings, times, images
	return embeddings, times


def make_time_heatmap(loader, model):
	xmin, xmax, ymin, ymax = -10, 15, -14, 12
	tmin, tmax = 13, 55
	embeddings, times = get_embeddings_times(loader, model)
	points = np.zeros((len(embeddings), 3))
	points[:,:2] = embeddings
	times -= 8*13
	points[:,2] = times
	scott = 0.5 * (len(points)) ** (-1.0/7.0) # half scott's rule
	kernel = gaussian_kde(points.T, bw_method=scott)
	hmap = np.zeros((100,100))
	xs, ys = np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100)
	ts = np.linspace(tmin,tmax,20)
	for i, x in tqdm(enumerate(xs)):
		for j, y in enumerate(ys):
			temp = np.array([[x,y,t] for t in ts])
			temp = kernel(temp.T)
			hmap[i,j] = np.dot(ts-tmin, temp) / (np.sum(temp) + 1e-10)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(np.rot90(hmap), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
	ax.plot(embeddings[:,0], embeddings[:,1], 'k.', alpha=0.07, markersize=0.5)
	plt.savefig('temp.pdf')
	plt.close('all')


# Make a gif w/ query times, not orderings
def make_kde_gif(loader, model):
	embeddings, times = get_embeddings_times(loader, model)
	times -= 290
	xmin, xmax, ymin, ymax = np.min(embeddings[:,0]), np.max(embeddings[:,0]), np.min(embeddings[:,1]), np.max(embeddings[:,1])
	gap = 4
	xmin, xmax, ymin, ymax = xmin-gap, xmax+gap, ymin-gap,ymax+gap
	# Write a bunch of jpgs.
	num_segs = (99-59+1) * 2 + 1
	query_times = np.linspace(59.0, 99.0, num_segs, endpoint=True)
	delta = query_times[1] - query_times[0]
	for i, query_time in enumerate(query_times):
		m1, m2 = [], []
		for time, embedding in zip(times, embeddings):
			if abs(time - query_time) < 3*delta:
				m1.append(embedding[0])
				m2.append(embedding[1])
		X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
		positions = np.vstack([X.ravel(), Y.ravel()])
		if len(m1) == 0:
			Z = np.zeros(X.shape)
		else:
			values = np.vstack([np.array(m1), np.array(m2)])
			scott = 0.5 * len(m1) ** (-1./6.)
			kernel = gaussian_kde(values, bw_method=scott)
			Z = np.reshape(kernel(positions).T, X.shape)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
				extent=[xmin, xmax, ymin, ymax])
		# rgba_colors = np.zeros((len(embeddings),4))
		# rgba_colors[:,3] = 0.04
		ax.plot(embeddings[:,0], embeddings[:,1], 'k.', alpha=0.07, markersize=0.5)
		# (embedding[0], embedding[1], color=rgba_colors, s=0.5)
		# ax.plot(m1, m2, 'k.', markersize=2)
		ax.set_xlim([xmin, xmax])
		ax.set_ylim([ymin, ymax])
		# if query_time < 32:
		# 	plt.text(-14, 6, "August "+str(int(np.floor(query_time))))
		# else:
		# 	plt.text(-14, 6, "September "+str(int(np.floor(query_time - 31.0))))
		plt.text(-8, 9, str(int(np.floor(query_time)))+' dph')
		plt.title("blu258 Development")
		plt.savefig(str(i).zfill(2)+'.jpg')
		plt.close('all')
	# Turn them into a gif.
	import imageio
	images = []
	for i in range(num_segs):
		image = imageio.imread(str(i).zfill(2) + '.jpg')
		images.append(image)
	print('saving gif')
	imageio.mimsave('temp.gif', images, duration=0.15)


# Make a gif w/ query times, not orderings
def make_dot_gif(loader, model):
	embeddings, times = get_embeddings_times(loader, model)
	print("times:", np.min(times), np.max(times))
	xmin, xmax, ymin, ymax = np.min(embeddings[:,0]), np.max(embeddings[:,0]), np.min(embeddings[:,1]), np.max(embeddings[:,1])
	gap = 4
	xmin, xmax, ymin, ymax = xmin-gap, xmax+gap, ymin-gap,ymax+gap
	# Write a bunch of jpgs.
	num_segs = (69-47) * 4 + 1
	query_times = np.linspace(47.0, 69.0, num_segs, endpoint=False)
	delta = query_times[1] - query_times[0]
	window = 4*delta
	for i, query_time in enumerate(query_times):
		m1, m2 = [], []
		ds = []
		for time, embedding in zip(times, embeddings):
			if time > query_time and time < query_time + window:
				m1.append(embedding[0])
				m2.append(embedding[1])
				ds.append(1.0 - (time - query_time)/window)
		plt.scatter(embeddings[:,0], embeddings[:,1], c='k', alpha=0.04, s=0.5)
		rgba_colors = np.zeros((len(ds),4))
		rgba_colors[:,0] = 1.0
		rgba_colors[:,3] = np.array(ds) ** 2
		plt.scatter(m1, m2, color=rgba_colors, s=0.8)
		plt.xlim([xmin, xmax])
		plt.ylim([ymin, ymax])
		plt.text(-8, 9, str(int(np.floor(query_time)))+' dph')
		plt.title("grn288 Development")
		plt.savefig(str(i).zfill(2)+'.jpg')
		plt.close('all')
	# Turn them into a gif.
	import imageio
	images = []
	for i in range(num_segs):
		image = imageio.imread(str(i).zfill(2) + '.jpg')
		images.append(image)
	print('saving gif')
	imageio.mimsave('temp.gif', images, duration=0.1)
