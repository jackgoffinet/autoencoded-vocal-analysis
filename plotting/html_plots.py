"""
Plot a syllable projection with spectrograms appearing as tooltips.

"""
import numpy as np
import matplotlib.pyplot as plt
import umap
import os

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models.glyphs import ImageURL


plt.switch_backend('agg')


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


def write_images(loader, model, output_dir='temp/', num_imgs=100, n=30000):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	# First get latent representations.
	latent, images = model.get_latent(loader, n=n, random_subset=True, return_fields=['image'])
	transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
	embedding = transform.fit_transform(latent)
	X = embedding[:,0]
	Y = embedding[:,1]
	Y[X < -10] += 5.0
	X[X < -10] += 10.0
	embedding[:,0] = X
	embedding[:,1] = Y


	# indices = []
	# for i, embed in enumerate(embedding):
	# 	if embed[0] > 4.0:
	# 		indices.append(i)
	# indices = np.array(indices, dtype='int')
	# transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
	# embedding = transform.fit_transform(latent[indices])
	# images = images[indices]
	# num_imgs = min(num_imgs, len(images))

	np.save('embedding.npy', embedding)
	for i in range(num_imgs):
		save_image(images[i], output_dir + str(i) + '.jpg')
	return embedding


def make_html_plot(loader, model, output_dir='temp/', num_imgs=1000, title="", n=30000):
	embedding = write_images(loader, model, output_dir=output_dir, num_imgs=num_imgs, n=n)
	output_file(output_dir + "main.html")
	source = ColumnDataSource(
			data=dict(
				x=np.load('embedding.npy')[:num_imgs,0],
				y=np.load('embedding.npy')[:num_imgs,1],
				imgs = ['./'+str(i)+'.jpg' for i in range(num_imgs)],
			)
		)

	source2 = ColumnDataSource(
			data=dict(
				x=np.load('embedding.npy')[num_imgs:,0],
				y=np.load('embedding.npy')[num_imgs:,1],
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


if __name__ == '__main__':
	pass
