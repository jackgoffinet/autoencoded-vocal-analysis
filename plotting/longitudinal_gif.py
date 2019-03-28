from __future__ import print_function, division
"""
Make a gif of birdsong trajectories.

"""
__author__ = "Jack Goffinet"
__date__ = "November 2018 - March 2019"


import time
import numpy as np
import umap
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
plt.switch_backend('agg')



def make_projection(loader, model, latent=None, title="", save_filename='temp.pdf', n=30000, axis=False):
	# # First get latent representations.
	# if latent is None:
	# 	latent, filenames, images = model.get_latent(loader, n=n, random_subset=True, return_fields=['filename', 'image']) # image is TEMP!
	# transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
	# embedding = transform.fit_transform(latent)
	# X, Y = embedding[:,0], embedding[:,1]
	#
	# # TEMP!
	# indices = []
	# for i, embed in enumerate(embedding):
	# 	if embed[0] > 4.0:
	# 		indices.append(i)
	# indices = np.array(indices, dtype='int')
	# transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
	# embedding = transform.fit_transform(latent[indices])
	#
	# d = {'latent': latent[indices], 'embedding':embedding, 'filenames':filenames[indices], 'images':images[indices]}
	# np.save('data.npy', d)

	d = np.load('data.npy').item()
	latent = d['latent']
	embedding = d['embedding']
	filenames = d['filenames']
	images = d['images']

	X, Y = embedding[:,0], embedding[:,1]

	# # TEMP!
	temp = np.sum(images, axis=2)
	mean_freqs = []
	for i in range(len(temp)):
		temp[i] /= np.sum(temp[i])
		mean_freqs.append(np.dot(temp[i], np.arange(128)))
	mean_freqs -= np.percentile(mean_freqs, 5)
	mean_freqs /= np.percentile(mean_freqs, 95)
	mean_freqs[mean_freqs < 0.0] = 0.0
	mean_freqs[mean_freqs > 1.0] = 1.0

	# durations = []
	# # from scipy.special import softmax
	# filter = np.zeros(128)
	# filter[:64] = np.linspace(1.0,0.0,64)
	# filter[64:] = np.linspace(0.0,1.0,64)
	# for image in images:
	# 	temp = np.max(image, axis=0)
	# 	temp /= np.sum(temp)
	# 	durations.append(np.dot(filter, temp))
	# durations -= np.percentile(durations, 5)
	# durations /= np.percentile(durations, 95)
	# durations[durations < 0.0] = 0.0
	# durations[durations > 1.0] = 1.0


	# sizes = np.zeros(len(filenames))
	# rgba_colors = np.zeros((len(filenames),4))
	# rgba_colors[:,3] = 0.4
	# for i, filename in enumerate(filenames):
	#
	# 	# Male/female
	# 	if 'Ai_14' in filename or 'RAm_2' in filename or 'RAm_6' in filename or 'RAm_female' in filename or 'VM_8' in filename:
	# 		rgba_colors[i,:] = [70/255,240/255,240/255,0.4]
	# 	elif 'opto' not in filename and '7d' not in filename and '14d' not in filename:
	# 		rgba_colors[i,:] = [245/255,130/255,48/255,0.3]
	#
	# 	else:
	# 		rgba_colors[i,3] = 0.01
	# 	sizes[i] = 1.2


		# if 'TVA_28_fd' in filename:
		# 	rgba_colors[i,:3] = [128/255,0.0,0.0]
		# 	sizes[i] = 4.0
		# if 'VM_31_fd' in filename:
		# 	rgba_colors[i,:3] = [170/255,110/255,40/255]
		# 	sizes[i] = 4.0
		# if 'Ai_14' in filename:
		# 	rgba_colors[i,:3] = [128/255,128/255,0.0]
		# 	sizes[i] = 4.0
		# if 'VM_47_fd' in filename:
		# 	rgba_colors[i,:3] = [210/255,245/255,60/255]
		# 	sizes[i] = 4.0
		# if 'VM_75_fd' in filename:
		# 	rgba_colors[i,:3] = [60/255,180/255,75/255]
		# 	sizes[i] = 4.0
		# if 'VM_8_fd' in filename:
		# 	rgba_colors[i,:3] = [70/255,240/255,240/255]
		# 	sizes[i] = 4.0
		# if 'RAm_2_fd' in filename:
		# 	rgba_colors[i,:3] = [0/255,130/255,200/255]
		# 	sizes[i] = 4.0
		# if 'RAm_female' in filename:
		# 	rgba_colors[i,:3] = [145/255,30/255,180/255]
		# 	sizes[i] = 4.0
		# if 'RAm_6_fd' in filename:
		# 	rgba_colors[i,:3] = [240/255,50/255,230/255]
		# 	sizes[i] = 4.0
		# else:
		# 	rgba_colors[i,3] = 0.02
		# 	sizes[i] = 1.2

		# # Opto
		# sizes[i] = 1.2
		# if 'opto' in filename:
		# 	rgba_colors[i,:] = [0.0,0.0,1.0,0.2]
		# else:
		# 	rgba_colors[i,:] = [1.0,0.0,0.0,0.2]

		# # TVA
		# if 'TVA' in filename:
		# 	rgba_colors[i,3] = 0.25
		# 	if '28_fd' in filename:
		# 		rgba_colors[i,0] = 1.0
		# 	elif '28_7d' in filename:
		# 		rgba_colors[i,1] = 0.6
		# 	elif '28_14d' in filename:
		# 		rgba_colors[i,2] = 1.0

	# patches = [mpatches.Patch(color=[1.0,0.0,0.0], label='female-directed'), mpatches.Patch(color=[0.0,0.0,1.0], label='opto-elicited')]
	plt.scatter(X, Y, c=mean_freqs, cmap='viridis', s=1.2, alpha=0.5)
	if len(title) > 0:
		plt.title(title)
	if not axis:
		plt.axis('off')
	# plt.legend(handles=patches, loc='best')
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


def get_embeddings_times(loader, model, return_latent=False, return_images=False, n=30000):
	# First get latent representations.
	if return_images:
		return_fields = ['time', 'image']
		latent, times, images = model.get_latent(loader, n=n, random_subset=True, return_fields=return_fields)
	else:
		return_fields = ['time']
		latent, times = model.get_latent(loader, n=n, random_subset=True, return_fields=return_fields)
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



# Make a gif w/ query times.
def make_dot_gif(loader, model, title="", n=30000):
	embeddings, times = get_embeddings_times(loader, model, n=n)
	tmax, tmin = np.max(times), np.min(times)
	print(tmin, tmax)
	print(time.gmtime(tmin))
	print(time.gmtime(tmax))
	xmin, xmax, ymin, ymax = np.min(embeddings[:,0]), np.max(embeddings[:,0]), np.min(embeddings[:,1]), np.max(embeddings[:,1])
	gap = 4
	xmin, xmax, ymin, ymax = xmin-gap, xmax+gap, ymin-gap,ymax+gap
	# Write a bunch of jpgs.
	num_segs = int(round(4.0 * (tmax-tmin) / (60*60*24)))
	query_times = np.linspace(tmin, tmax, num_segs, endpoint=False)
	delta = query_times[1] - query_times[0]
	window = 4*delta
	for i, query_time in enumerate(query_times):
		m1, m2 = [], []
		ds = []
		for t, embedding in zip(times, embeddings):
			if t > query_time and t < query_time + window:
				m1.append(embedding[0])
				m2.append(embedding[1])
				ds.append(1.0 - (t - query_time)/window)
		plt.scatter(embeddings[:,0], embeddings[:,1], c='k', alpha=0.04, s=0.5)
		rgba_colors = np.zeros((len(ds),4))
		rgba_colors[:,0] = 1.0
		rgba_colors[:,3] = np.array(ds) ** 2
		plt.scatter(m1, m2, color=rgba_colors, s=0.8)
		plt.xlim([xmin, xmax])
		plt.ylim([ymin, ymax])
		plt.text(xmin+2, ymin-5, time.strftime('%b %d', time.gmtime(query_time)))
		plt.title(title)
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
