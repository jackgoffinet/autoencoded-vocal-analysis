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


def make_projection(loader, model):
	# First get latent representations.
	latent, times = model.get_latent(loader, n=30000, random_subset=True, return_times=True)
	# indices = np.where(times > 8*31)
	# print("total:", len(latent))
	# latent = latent[indices]
	# print("kept:", len(latent))
	# times = times[indices]
	# conditions = get_conditions(loader, n=10000)
	# individuals = get_conditions(loader, n=10000, query='individual')
	# sessions = get_conditions(loader, n=10000, query='session')
	# mean_freqs = get_mean_freqs(loader, n=10000)

	# transform = TSNE(n_components=2, n_iter=1000)
	transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
	print("fitting transform")
	embedding = transform.fit_transform(latent)
	print("done")
	np.save('embedding.npy', embedding)
	# rgba_colors = np.zeros((len(embedding),4))
	# rgba_colors[:,2] = conditions
	# rgba_colors[:,3] = 0.2
	patches = [mpatches.Patch(color=[0.0,0.0,0.0,0.2], label='air'),
		mpatches.Patch(color=[0.0,0.0,1.0,0.2], label='heliox')]
	# cmap = plt.cm.get_cmap('Set1', 3)
	# patches = [mpatches.Patch(color=cmap(i), label='BM00'+str(i+4)) for i in range(3)]
	# colors = cmap(individuals - 3)
	colors = []
	X, Y = embedding[:,0], embedding[:,1]
	# X, Y = [], []
	temp_cmap = plt.get_cmap('viridis')
	# colors = temp_cmap(mean_freqs)
	# patches = []
	latent_by_session = [[],[],[],[],[],[],[],[],[],[]]
	# for i in range(5):
		# patches.append(mpatches.Patch(color=temp_cmap(i/4.0), label='Session '+str(i+1)))
	# patches = [mpatches.Patch(color='r', label='Session 1 (air)'),
			# mpatches.Patch(color='b', label='Sessions 2-9 (heliox)'),
			# mpatches.Patch(color='k', label='Session 10 (air)')]
	# for i, point in enumerate(embedding):
	# 	if individuals[i] == 5:
	# 		X.append(point[0])
	# 		Y.append(point[1])
	# 		temp = 'b'
	# 		if sessions[i] == 1:
	# 			temp = 'r'
	# 		elif sessions[i] == 10:
	# 			temp = 'k'
	# 		colors.append(temp)
	# 		latent_by_session[int(round(sessions[i]-1))].append(latent[i])
	# np.save('BM005_latent_by_session.npy', latent_by_session)
	# print("found: ", len(X))
	plt.scatter(X, Y, c='b', alpha=0.1, s=1) #rgba_colors
	# plt.legend(handles=patches, loc='best')
	# plt.ylim((0,14))
	# plt.xlim((5,25))
	plt.title('pur224 projection, 90 dph')
	# plt.title('USV Mean Frequencies')
	plt.axis('off')
	plt.savefig('temp.pdf')
	plt.close('all')


def generate_syllables(loader, model):
	"""Plot a projection of the latent space and generated syllables."""
	height, width = 4, 6
	# First get the latent projection used to define the transform.
	latent, times = model.get_latent(loader, n=30000, random_subset=True, return_times=True)
	indices = np.where(times > 8*31)
	print("total:", len(latent))
	latent = latent[indices]
	print("kept:", len(latent))
	times = times[indices]
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


def get_embeddings_times(loader, model):
	# First get latent representations.
	latent, times = model.get_latent(loader, n=30000, random_subset=True, return_times=True)
	indices = np.where(times > 8*31)
	print("total:", len(latent))
	latent = latent[indices]
	print("kept:", len(latent))
	times = times[indices]
	# perm = np.random.permutation(len(latent))
	# latent = latent[perm]
	# times = times[perm]
	# Fit UMAP on a random subset, get embedding.
	transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
	print("fitting transform")
	transform.fit(latent)
	print("done")
	embeddings = transform.transform(latent)
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
	times -= 8*13
	xmin, xmax, ymin, ymax = -7, 15, -7, 15
	# Write a bunch of jpgs.
	num_segs = (56-14) * 2 + 1
	query_times = np.linspace(14.0, 56.0, num_segs, endpoint=True)
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
		if query_time < 32:
			plt.text(-14, 6, "August "+str(int(np.floor(query_time))))
		else:
			plt.text(-14, 6, "September "+str(int(np.floor(query_time - 31.0))))
		plt.text(-14, 9, str(int(np.floor(query_time)+35))+' dph')
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


# def make_kde_gif(loader, model):
	embeddings, times = get_embeddings_times(loader, model)
	xmin, xmax, ymin, ymax = -10, 15, -14, 12
	# Write a bunch of jpgs.
	num_points = 1800
	delta = 200
	p = times.argsort()
	times = times[p]
	embeddings = embeddings[p]
	print("num points", len(times))
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
		plt.axis('off')
		med_time = times[j + delta // 2] - 8*31
		if med_time < 32:
			plt.text(-14, 6, "August "+str(int(np.floor(med_time))))
		else:
			plt.text(-14, 6, "September "+str(int(np.floor(med_time - 31.0))))
		plt.text(-14, 9, str(int(np.floor(med_time)+35))+' dph')
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



# def make_gif(loader, model):
# 	embeddings, times = get_latent_times(loader, model)
#
# 	# Write a bunch of jpgs.
# 	num_segs = 27-15
# 	query_times = np.linspace(15.0, 27.0, num_segs, endpoint=False)
# 	delta = query_times[1] - query_times[0]
# 	print((np.max(times)-np.min(times)))
# 	min_time = np.min(times)
# 	max_time = np.max(times)
#
# 	for i, query_time in enumerate(query_times):
# 		X = []
# 		Y = []
# 		rgba_colors = np.zeros((len(embeddings),4))
# 		for j, time, embedding in zip(range(len(times)), times, embeddings):
# 			X.append(embedding[0])
# 			Y.append(embedding[1])
# 			if time >= query_time and time < query_time + delta:
# 				rgba_colors[j,0] = (time - min_time) / (max_time - min_time)
# 				rgba_colors[j,2] = 1 - (time - min_time) / (max_time - min_time)
# 				rgba_colors[j,3] = 0.7
# 			else:
# 				rgba_colors[j,3] = 0.04
#
#
# 		plt.scatter(X, Y, color=rgba_colors, s=0.5)
# 		plt.title("pur224 projection")
# 		axes = plt.gca()
# 		axes.set_xlim([-14,9])
# 		axes.set_ylim([-11,13])
# 		plt.text(-14, 6, "August "+str(int(np.floor(query_time))))
# 		plt.text(-14, 9, str(int(np.floor(query_time)+35))+' dph')
# 		plt.savefig(str(i).zfill(2)+'.jpg')
# 		plt.close('all')
#
# 	# Turn them into a gif.
# 	import imageio
# 	images = []
# 	for i in range(num_segs):
# 		image = imageio.imread(str(i).zfill(2) + '.jpg')
# 		images.append(image)
# 	print('saving gif')
# 	imageio.mimsave('temp.gif', images, duration=1)
