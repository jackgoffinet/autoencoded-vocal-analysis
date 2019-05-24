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

from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.switch_backend('agg')

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models.glyphs import ImageURL


def mean_freq(image):
	f = np.linspace(30,135,128)
	temp = np.sum(image, axis=1).flatten()
	thresh = np.quantile(temp, 0.7)
	temp -= thresh
	temp[temp < 0.0] = 0.0
	temp /= np.sum(temp)
	return np.dot(temp, f)


def make_projection(d, title="", save_filename='temp.pdf', n=30000, axis=False):
	# required_fields = ['embedding', 'filename', 'image'] # filename, image is temporary
	# d = update_data(d, required_fields, n=n)
	# embedding = d['embedding']
	#
	# np.save('d.npy', d)
	d = np.load('d.npy').item()

	X, Y = d['embedding'][:,0], d['embedding'][:,1]

	rgba_colors = np.zeros((len(d['embedding']),4))
	rgba_colors[:,3] = 0.6
	cmap = get_cmap('viridis')
	# color='tab:blue'
	patches = [ \
			mpatches.Patch(color=cmap(0/8), label='0%'), \
			mpatches.Patch(color=cmap(2/8), label='20%'), \
			mpatches.Patch(color=cmap(3/8), label='30%'), \
			mpatches.Patch(color=cmap(4/8), label='40%'), \
			mpatches.Patch(color=cmap(5/8), label='50%'), \
			mpatches.Patch(color=cmap(6/8), label='60%'), \
			mpatches.Patch(color=cmap(8/8), label='80%'), \
			]

	values = []
	for image in d['image']:
		values.append(mean_freq(image))
	values = np.array(values)
	min_val, max_val = np.min(values), np.max(values)
	values = (values - min_val) / (max_val - min_val)

	print("values", np.min(values), np.max(values))

	fig, ax = plt.subplots()
	cax = ax.scatter(X, Y, c=values, cmap='viridis', alpha=0.5, s=0.45)
	ax.set_aspect('equal')
	if len(title) > 0:
		ax.set_title(title)
	if not axis:
		ax.axis('off')
	# plt.legend(handles=patches, loc='lower center', ncol=7)
	ticks = [(i - min_val)/(max_val-min_val) for i in [50,80,110]]
	cbar = fig.colorbar(cax, fraction=0.046, orientation="horizontal", ticks=ticks)
	cbar.solids.set_edgecolor("face")
	cbar.solids.set_rasterized(True)
	cbar.ax.set_xticklabels(['50 kHz', '80 kHz', '110 kHz'])
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

		# NOTE: TEMP
		indices = np.argwhere(embedding[:,0] > -3).flatten()
		transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
		embedding = transform.fit_transform(data['latent'][indices])
		for field in list(data.keys()):
			if field not in ['embedding', 'model', 'loader']:
				try:
					data[field] = data[field][indices]
				except:
					print("caught on", field)
					quit()

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



if __name__ == '__main__':
	pass




###
