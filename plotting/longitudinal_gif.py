from __future__ import print_function, division
"""
Make a gif of birdsong trajectories.

"""
__author__ = "Jack Goffinet"
__date__ = "November 2018 - March 2019"


import os
import time
from tqdm import tqdm
import joblib

import numpy as np
import umap
import hdbscan
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.switch_backend('agg')

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models.glyphs import ImageURL



def make_projection(d, title="", save_filename='temp.pdf', n=30000, axis=False):
	required_fields = ['latent', 'embedding']
	d = update_data(d, required_fields, n=n)
	latents = d['latent']
	embedding = d['embedding']

	X, Y = embedding[:,0], embedding[:,1]

	# rgba_colors = np.zeros((len(embedding),4))
	# rgba_colors[:,3] = 0.6
	# patches = [ \
	# 		mpatches.Patch(color='tab:blue', label='S1'), \
	# 		mpatches.Patch(color='tab:orange', label='S2'), \
	# 		mpatches.Patch(color='tab:green', label='S3'), \
	# 		mpatches.Patch(color='tab:red', label='S4'), \
	# 		mpatches.Patch(color='tab:purple', label='S5'), \
	# 		]

	plt.scatter(X, Y, c='b', alpha=0.1, s=0.25)

	if len(title) > 0:
		plt.title(title)
	if not axis:
		plt.axis('off')
	# plt.legend(handles=patches, loc='best')
	plt.savefig(save_filename)
	plt.close('all')
	return d


def update_data(data, required_fields, n=30000):
	print('required_fields', required_fields)
	required_fields = [i for i in required_fields if i not in list(data.keys())]
	# Get stuff from the model.
	all_return_fields = ['image', 'duration', 'time', 'file_time', 'filename']
	return_fields = [i for i in required_fields if i in all_return_fields]
	if 'day' in required_fields and 'time' not in data and 'time' not in return_fields:
		return_fields += ['time']
	if len(return_fields) > 0 or 'latent' not in data:
		temp = data['model'].get_latent(data['loader'], n=n, random_subset=True, return_fields=return_fields)
		return_fields = ['latent'] + return_fields
		for i, j in zip(return_fields, temp):
			data[i] = j
	required_fields = [i for i in required_fields if i not in return_fields + ['latent']]
	if 'label' in required_fields and 'embedding' not in data:
		required_fields = ['embedding'] + required_fields
	# Compute stuff.
	if 'embedding' in required_fields and 'embedding' not in data:
		transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
		embedding = transform.fit_transform(data['latent'])
		joblib.dump(transform, 'temp_reducer.sav')
		data['embedding'] = embedding
	if 'label' in required_fields and 'label' not in data:
		# pca = PCA(n_components=4)
		# points = pca.fit_transform(data['latent'])
		# data['pca_projection'] = points
		clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=200)
		clusterer.fit(data['embedding'])
		data['label'] = clusterer.labels_
	if 'day' in required_fields and 'day' not in data:
		days = day_from_time(d['time'])
		data['day'] = days
	return data


def plot_generated_cluster_means(d, title="", n=30000):
	# Update the data.
	required_fields = ['latent', 'time', 'label']
	d = update_data(d, required_fields, n=n)
	latents = d['latent']
	times = d['time']
	labels = d['label']

	d_latent = {}
	for i in range(len(labels)):
		label = labels[i]
		if labels[i] == -1: # unclustered
			continue
		day = day_from_time(times[i])
		if (label, day) in d_latent:
			d_latent[(label, day)].append(latents[i])
		else:
			d_latent[(label, day)] = [latents[i]]
	min_time, max_time = np.min(times), np.max(times)
	for label in range(np.max(labels)+1):
		label_latent = latents[np.where(labels==label)]
		pca = PCA(n_components=2)
		label_p = pca.fit_transform(label_latent)
		min_x, max_x = np.min(label_p[:,0]), np.max(label_p[:,0])
		min_y, max_y = np.min(label_p[:,1]), np.max(label_p[:,1])
		current_time = min_time
		i = 0
		x_t = []
		y_t = []
		while current_time <= max_time:
			day = day_from_time(times[i])
			if (label, day) in d_latent:
				n_labeled = len(d_latent[(label, day)])
				mean_latent = np.mean(np.array(d_latent[(label, day)]), axis=0).reshape(1,-1)
				temp = pca.transform(mean_latent.reshape(1,-1)).flatten()
				x_t.append(temp[0])
				y_t.append(temp[1])
				generated_spec = d['model'].generate_from_latent(mean_latent).reshape(128,128)
				fig, axarr = plt.subplots(1,2)
				axarr[0].imshow(generated_spec, origin='lower')
				axarr[0].set_title("Generated Spectrogram")
				axarr[0].set_axis_off()
				axarr[1].scatter(label_p[:,0], label_p[:,1], c='b', alpha=0.1)
				axarr[1].plot(x_t, y_t, c='r')
				axarr[1].set_xlim(min_x-1, max_x+1)
				axarr[1].set_ylim(min_y-1, max_y+1)
				axarr[1].set_axis_off()
				axarr[1].set_title("Latent Trajectory")
				axarr[1].scatter(x_t[-1], y_t[-1], c='r', s=30, marker='*')
			else:
				plt.plot([],[])
				plt.title('...')
			plt.savefig(str(i).zfill(2)+'.jpg')
			plt.close('all')
			current_time += 60*60*24 # a day, in seconds
			i += 1
		import imageio
		images = []
		for j in range(i):
			image = imageio.imread(str(j).zfill(2) + '.jpg')
			images.append(image)
		print('saving gif')
		imageio.mimsave('temp_'+str(label)+'.gif', images, duration=0.2)
	return d


def day_from_time(query_times):
	try:
		iterator = iter(query_times)
	except TypeError:
		return time.strftime('%b %d', time.gmtime(query_times))
	result = [time.strftime('%b %d', time.gmtime(i)) for i in query_times]
	return np.array(result)



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


# def make_time_heatmap(loader, model):
# 	xmin, xmax, ymin, ymax = -10, 15, -14, 12
# 	tmin, tmax = 13, 55
# 	embeddings, times = get_embeddings_times(loader, model)
# 	points = np.zeros((len(embeddings), 3))
# 	points[:,:2] = embeddings
# 	times -= 8*13
# 	points[:,2] = times
# 	scott = 0.5 * (len(points)) ** (-1.0/7.0) # half scott's rule
# 	kernel = gaussian_kde(points.T, bw_method=scott)
# 	hmap = np.zeros((100,100))
# 	xs, ys = np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100)
# 	ts = np.linspace(tmin,tmax,20)
# 	for i, x in tqdm(enumerate(xs)):
# 		for j, y in enumerate(ys):
# 			temp = np.array([[x,y,t] for t in ts])
# 			temp = kernel(temp.T)
# 			hmap[i,j] = np.dot(ts-tmin, temp) / (np.sum(temp) + 1e-10)
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111)
# 	ax.imshow(np.rot90(hmap), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
# 	ax.plot(embeddings[:,0], embeddings[:,1], 'k.', alpha=0.07, markersize=0.5)
# 	plt.savefig('temp.pdf')
# 	plt.close('all')



# Make a gif w/ query times.
def make_dot_gif(d, title="", n=30000, axis=False):
	# Update the data.
	required_fields = ['embedding', 'time']
	d = update_data(d, required_fields, n=n)
	embeddings = d['embedding']
	times = d['time']

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
		if not axis:
			plt.axis('off')
		plt.text(xmin+2, ymax-2, time.strftime('%b %d', time.gmtime(query_time)))
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
	return d



def make_html_plot(d, output_dir='temp', n=30000, num_imgs=2000, title=""):
	"""Make an HTML tooltip mouse-over plot."""
	required_fields = ['image', 'embedding']
	d = update_data(d, required_fields, n=n)
	embedding = d['embedding']
	specs = d['image']

	# Write the tooltip images.
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	num_imgs = min(len(specs), num_imgs)
	for i in tqdm(range(num_imgs)):
		save_image(specs[i], os.path.join(output_dir, str(i)+'.jpg'))
	output_file(os.path.join(output_dir,"main.html"))
	source = ColumnDataSource(
			data=dict(
				x=embedding[:num_imgs,0],
				y=embedding[:num_imgs,1],
				imgs = ['./'+str(i)+'.jpg' for i in range(num_imgs)],
			)
		)
	source2 = ColumnDataSource(
			data=dict(
				x=embedding[num_imgs:,0],
				y=embedding[num_imgs:,1],
			)
		)
	p = figure(plot_width=800, plot_height=600, title=title)
	p.scatter('x', 'y', size=3, fill_color='blue', fill_alpha=0.1, source=source2)
	tooltip_points = p.scatter('x', 'y', size=5, fill_color='red', source=source)
	hover = HoverTool(
			renderers=[tooltip_points],
			tooltips="""
			<div>
				<div>
					<img
						src="@imgs" height="128" alt="@imgs" width="128"
						style="float: left; margin: 0px 0px 0px 0px;"
						border="1"
					></img>
				</div>
			</div>
			"""
		)
	p.add_tools(hover)
	p.title.align = "center"
	p.title.text_font_size = "25px"
	p.axis.visible = False
	p.xgrid.visible = False
	p.ygrid.visible = False
	show(p)


def save_image(data, filename):
	"""https://fengl.org/2014/07/09/matplotlib-savefig-without-borderframe/"""
	sizes = np.shape(data)
	height = float(sizes[0])
	width = float(sizes[1])
	fig = plt.figure()
	fig.set_size_inches(width/height, 1, forward=False)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(data, cmap='viridis', origin='lower')
	plt.savefig(filename, dpi=height)
	plt.close('all')
