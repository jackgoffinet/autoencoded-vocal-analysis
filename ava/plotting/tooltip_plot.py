"""
Plot a syllable projection with spectrograms appearing as tooltips.


TO DO:
- Stop reading and writing the embedding.
"""
__date__ = "March 2019 - July 2019"

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models.glyphs import ImageURL
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import numpy as np
import umap



def tooltip_plot_DC(d, num_imgs=5000, output_dir='html', title="", n=30000):
	"""
	DataContainer version of tooltip_plot.

	Parameters
	----------
	d : plotting.DataContainer
		See plotting.DataContainer class for details.

	Other Parameters
	----------------
	See plotting.tooltip_plot.tooltip_plot

	"""
	embedding = d.request('latent_mean_umap')
	images = d.request('specs')
	output_dir = os.path.join(d.plots_dir, output_dir)
	print("writing tooltip plot to", output_dir)
	tooltip_plot(embedding, images, output_dir=output_dir, num_imgs=num_imgs, \
		title=title, n=n)


def tooltip_plot(embedding, images, output_dir='temp', num_imgs=10000, title="",
	n=30000):
	"""
	Create a scatterplot of the embedding with spectrogram tooltips.

	Parameters
	----------
	embedding : numpy.ndarray
		The scatterplot coordinates. Shape: (num_points, 2)

	images : numpy.ndarray
		A spectrogram image for each scatter point. Shape:
		(num_points, height, width)

	output_dir : str, optional
		Directory where html and jpegs are written. Deafaults to "temp".

	num_imgs : int, optional
		Number of points with tooltip images. Defaults to 10000.

	title : str, optional
		Title of plot. Defaults to ''.

	n : int, optional
		Total number of scatterpoints to plot. Defaults to 30000.
	"""
	# Shuffle the embedding and images.
	np.random.seed(42)
	perm = np.random.permutation(len(embedding))
	np.random.seed(None)
	embedding = embedding[perm]
	images = images[perm]

	n = min(len(embedding), n)
	num_imgs = min(len(images), num_imgs)
	write_images(embedding, images, output_dir=output_dir, num_imgs=num_imgs, n=n)
	output_file(os.path.join(output_dir, "main.html"))
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


def write_images(embedding, images, output_dir='temp/', num_imgs=100, n=30000):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	X = embedding[:,0]
	Y = embedding[:,1]
	for i in range(num_imgs):
		save_image(images[i], os.path.join(output_dir, str(i) + '.jpg'))
	return embedding



if __name__ == '__main__':
	pass


###
