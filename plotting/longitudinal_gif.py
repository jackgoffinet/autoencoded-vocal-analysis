from __future__ import print_function, division
"""
Make a .gif of birdsong trajectories.

"""
__author__ = "Jack Goffinet"
__date__ = "November 2018"


import numpy as np
import umap
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
plt.switch_backend('agg')


def make_projection(loader, model):
	# First get latent representations.
	latent = model.get_latent(loader, n=9000)
	transform = TSNE(n_components=2, n_iter=1000)
	# transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean')
	print("fitting")
	embedding = transform.fit_transform(latent)
	print("done")
	np.save('embedding.npy', embedding)
	rgba_colors = np.zeros((len(embedding),4))
	rgba_colors[:,2] = 1.0
	rgba_colors[:,3] = 0.2
	plt.scatter(embedding[:,0], embedding[:,1], c=rgba_colors, s=0.5)
	plt.savefig('temp.pdf')
	plt.close('all')


def generate_syllables(loader, model):
	"""Plot a projection of the latent space and generated syllables."""
	# First get latent representations.
	latent = model.get_latent(loader, n=9000)
	transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean')
	embedding = transform.fit_transform(latent)
	np.save('embedding.npy', embedding)
	cmap = plt.cm.get_cmap('Set1', 5)
	patches = [mpatches.Patch(color=cmap(i), label='S'+str(i+1)) for i in range(5)]
	colors = []
	individuals = []
	for temp in loader:
		individuals += temp['individual'].detach().cpu().numpy().tolist()
		colors += cmap(temp['individual'].detach().cpu().numpy() - 1).tolist()
	x = embedding[:,0]
	y = embedding[:,1]
	plt.scatter(x, y, c=colors, alpha=0.8, s=0.5)
	np.random.seed(42)
	p = np.random.permutation(len(embedding))
	points = embedding[p][:20]
	for i, point in enumerate(points):
		plt.text(point[0], point[1], str(i))
	plt.legend(handles=patches, loc='best')
	plt.savefig('temp.pdf')
	plt.close('all')
	# Plot spectrograms.
	gap = 10
	big_img = np.zeros((4*128+3*gap, 5*128+4*gap))
	for i in range(4):
		for j in range(5):
			temp_im = loader.dataset[p[i*5+j]]['image'].detach().cpu().numpy().reshape(128,128)
			big_img[i*(128+gap):i*(128+gap)+128,j*(128+gap):j*(128+gap)+128] = temp_im
	plt.imshow(big_img, aspect='auto', origin='lower', interpolation='none')
	for i in range(4):
		for j in range(5):
			plt.text(j*(128+gap), i*(128+gap), str(i*5+j), color='white')
	plt.axis('off')
	plt.savefig('temp1.pdf')
	plt.close('all')





def get_embeddings_times(loader, model):
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
	transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean')
	print("fitting gif")
	transform.fit(latent[:9000])
	print("done")
	embeddings = transform.transform(latent)
	return embeddings, times



def make_time_heatmap(loader, model):
	xmin, xmax, ymin, ymax, tmin, tmax = -7, 15, -7, 15, 15, 26
	embeddings, times = get_embeddings_times(loader, model)
	points = np.zeros((len(embeddings), 3))
	points[:,:2] = embeddings
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



	# # embeddings = embeddings[times.argsort()]
	# # times = np.linspace(0,1,len(embeddings))
	# xmin, xmax, ymin, ymax = -7, 15, -7, 15
	# hmap = np.zeros((100,100))
	# points = embeddings[:1000]
	# values = np.array([heatmap_func(embedding) for embedding in embeddings])
	#
	# for i, x in tqdm(enumerate(np.linspace(xmin, xmax, 100, endpoint=True))):
	# 	for j, y in enumerate(np.linspace(ymin, ymax, 100, endpoint=True)):
	# 		weights, values = [], []
	# 		for k, embedding in enumerate(embeddings):
	# 			x_e, y_e = embedding
	# 			r2 = (x_e - x)**2 + (y_e - y)**2
	# 			if r2 < 1.0:
	# 				continue
	# 			weights.append((r2+1.0)**(-2))
	# 			values.append(times[k])
	# 		if len(weights) < 10:
	# 			hmap[j,i] = -0.1
	# 		else:
	# 			weights = np.array(weights)
	# 			weights /= np.sum(weights)
	# 			values = np.array(values)
	# 			hmap[j,i] = np.dot(weights, values)
	# np.save('time_hmap.npy', hmap)
	# # hmap = np.load('time_hmap.npy')
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.imshow(hmap, cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
	# ax.plot(embeddings[:,0], embeddings[:,1], 'k.', alpha=0.07, markersize=0.5) # NOTE: TEMP
	# ax.set_xlim([xmin, xmax])
	# ax.set_ylim([ymin, ymax])
	# # ax.colorbar()
	# plt.savefig('temp.pdf')
	# plt.close('all')


# # Make a gif w/ query times, not orderings
# def make_kde_gif(loader, model):
# 	embeddings, times = get_embeddings_times(loader, model)
# 	xmin, xmax, ymin, ymax = -7, 15, -7, 15
# 	# Write a bunch of jpgs.
# 	num_segs = 100
# 	query_times = np.linspace(15.0, 27.0, num_segs, endpoint=False)
# 	delta = query_times[1] - query_times[0]
# 	min_time = np.min(times)
# 	max_time = np.max(times)
# 	for i, query_time in enumerate(query_times):
# 		m1, m2 = [], []
# 		for time, embedding in zip(times, embeddings):
# 			if abs(time - query_time) < 6*delta:
# 				m1.append(embedding[0])
# 				m2.append(embedding[1])
# 		X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# 		positions = np.vstack([X.ravel(), Y.ravel()])
# 		if len(m1) == 0:
# 			Z = np.zeros(X.shape)
# 		else:
# 			values = np.vstack([np.array(m1), np.array(m2)])
# 			scott = 0.5 * len(m1) ** (-1./6.)
# 			kernel = gaussian_kde(values, bw_method=scott)
# 			Z = np.reshape(kernel(positions).T, X.shape)
# 		fig = plt.figure()
# 		ax = fig.add_subplot(111)
# 		ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
# 				extent=[xmin, xmax, ymin, ymax])
# 		# rgba_colors = np.zeros((len(embeddings),4))
# 		# rgba_colors[:,3] = 0.04
# 		ax.plot(embeddings[:,0], embeddings[:,1], 'k.', alpha=0.07, markersize=0.5)
# 		# (embedding[0], embedding[1], color=rgba_colors, s=0.5)
# 		# ax.plot(m1, m2, 'k.', markersize=2)
# 		ax.set_xlim([xmin, xmax])
# 		ax.set_ylim([ymin, ymax])
# 		plt.text(-5, 10, "August "+str(int(np.floor(query_time))))
# 		plt.text(-5, 12, str(int(np.floor(query_time)+35))+' dph')
# 		plt.savefig(str(i).zfill(2)+'.jpg')
# 		plt.close('all')
# 	# Turn them into a gif.
# 	import imageio
# 	images = []
# 	for i in range(num_segs):
# 		image = imageio.imread(str(i).zfill(2) + '.jpg')
# 		images.append(image)
# 	print('saving gif')
# 	imageio.mimsave('temp.gif', images, duration=0.15)


def make_kde_gif(loader, model):
	embeddings, times = get_embeddings_times(loader, model)
	xmin, xmax, ymin, ymax = -7, 15, -7, 15
	# Write a bunch of jpgs.
	num_points = 1800
	delta = 200
	p = times.argsort()
	times = times[p]
	print("num points", len(times))
	embeddings = embeddings[p]
	fig_num, j = 0, 0
	while j + num_points < len(times):
		X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
		positions = np.vstack([X.ravel(), Y.ravel()])
		m1 = embeddings[j:j+num_points,0]
		m2 = embeddings[j:j+num_points,1]
		values = np.vstack([m1, m2])
		scott = 0.5 * len(m1) ** (-1./6.) # half scott
		kernel = gaussian_kde(values, bw_method=scott)
		Z = np.reshape(kernel(positions).T, X.shape)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
				extent=[xmin, xmax, ymin, ymax])
		ax.plot(embeddings[:,0], embeddings[:,1], 'k.', alpha=0.07, markersize=0.5)
		ax.set_xlim([xmin, xmax])
		ax.set_ylim([ymin, ymax])
		med_time = times[j + delta // 2]
		plt.text(-5, 10, "August "+str(int(np.floor(med_time))))
		plt.text(-5, 12, str(int(np.floor(med_time)+35))+' dph')
		plt.savefig(str(fig_num).zfill(2)+'.jpg')
		plt.close('all')
		fig_num += 1
		j += delta
	# Turn them into a gif.
	import imageio
	images = []
	for j in range(fig_num):
		image = imageio.imread(str(j).zfill(2) + '.jpg')
		images.append(image)
	print('saving gif')
	imageio.mimsave('temp.gif', images, duration=0.15)


def make_gif(loader, model):
	embeddings, times = get_latent_times(loader, model)

	# Write a bunch of jpgs.
	num_segs = 27-15
	query_times = np.linspace(15.0, 27.0, num_segs, endpoint=False)
	delta = query_times[1] - query_times[0]
	print((np.max(times)-np.min(times)))
	min_time = np.min(times)
	max_time = np.max(times)

	for i, query_time in enumerate(query_times):
		X = []
		Y = []
		rgba_colors = np.zeros((len(embeddings),4))
		for j, time, embedding in zip(range(len(times)), times, embeddings):
			X.append(embedding[0])
			Y.append(embedding[1])
			if time >= query_time and time < query_time + delta:
				rgba_colors[j,0] = (time - min_time) / (max_time - min_time)
				rgba_colors[j,2] = 1 - (time - min_time) / (max_time - min_time)
				rgba_colors[j,3] = 0.7
			else:
				rgba_colors[j,3] = 0.04


		plt.scatter(X, Y, color=rgba_colors, s=0.5)
		axes = plt.gca()
		# axes.set_ylim([-6,15])
		# axes.set_xlim([-7,13])
		axes.set_ylim([-6,15])
		axes.set_xlim([-6,15])
		plt.text(-5, 10, "August "+str(int(np.floor(query_time))))
		plt.text(-5, 12, str(int(np.floor(query_time)+35))+' dph')
		plt.savefig(str(i).zfill(2)+'.jpg')
		plt.close('all')

	# Turn them into a gif.
	import imageio
	images = []
	for i in range(num_segs):
		image = imageio.imread(str(i).zfill(2) + '.jpg')
		images.append(image)
	print('saving gif')
	imageio.mimsave('temp.gif', images, duration=1)
