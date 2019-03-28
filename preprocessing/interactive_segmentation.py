"""
Interactive syllable segmentation

1) Relevant audio?
2) Boundary?
3) Good clusters?
"""

__author__ = "Jack Goffinet"
__date__ = "March 2019"

import os
import numpy as np
import h5py
import joblib
from tqdm import tqdm
from skimage.transform import resize
from scipy.io import wavfile, loadmat
from scipy.signal import stft, welch
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
# plt.switch_backend('agg')

import umap

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models.glyphs import ImageURL

FS = 44100
SPEC_THRESH = 0.0
EPSILON = 1e-9

X1, Y1, X2, Y2 = 0, 0, 0, 0
LINES = []

def segment(load_dirs):
	global X1, Y1, X2, Y2
	# Collect filenames.
	filenames = []
	for load_dir in load_dirs:
		filenames += [os.path.join(load_dir,i) for i in os.listdir(load_dir) if i[-4:] in ['.wav', '.mat']]
	filenames = [i for i in filenames if wavfile.read(i)[0] == FS]

	# Collect PSDs and spectrograms.
	psds, specs = [], []
	np.random.seed(42)
	psd_f_ind, spec_f_ind = None, None
	print("Collecting data...")
	for i in tqdm(range(10000)):
		filename = filenames[np.random.randint(len(filenames))]
		psd, spec, psd_f_ind, spec_f_ind = get_rand_psd_spec(filename, psd_dur=0.12, spec_dur=0.15, psd_f_ind=psd_f_ind, spec_f_ind=spec_f_ind)
		psds.append(psd)
		specs.append(spec)
	# UMAP the PSDs.
	print("Dimensionality reduction...")
	transform_1 = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
	embedding = transform_1.fit_transform(np.array(psds))
	joblib.dump(transform_1, 'transform_1.sav')
	perm = np.random.permutation(len(embedding))
	embedding, specs = embedding[perm], np.array(specs)[perm]
	np.save('embedding.npy', embedding)
	# Make an HTML mouse-over plot.
	print("Making HTML...")
	make_html_plot(embedding, specs, output_dir='temp2')
	embedding = np.load('embedding.npy')
	# Plot an interactive selcting tool.
	print("Interactive plot...")
	x1s, y1s, x2s, y2s = [], [], [], []
	colors = np.array(['b'] * len(embedding))
	while X1 is not None:
		X1 = None
		for i, embed in enumerate(embedding):
			if colors[i] == 'r':
				continue
			for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
				if embed[0] < max(x1, x2) and embed[0] > min(x1, x2) and \
						embed[1] < max(y1, y2) and embed[1] > min(y1, y2):
					colors[i] = 'r'
		fig, current_ax = plt.subplots()
		toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
				drawtype='box', useblit=True,
				button=[1, 3],  # don't use middle button
				minspanx=5, minspany=5,
				spancoords='pixels',
				interactive=True)
		plt.connect('key_press_event', toggle_selector)
		plt.scatter(embedding[:,0], embedding[:,1], c=colors, alpha=0.1, s=0.7)
		plt.show()
		if X1 is not None:
			x1s.append(X1)
			y1s.append(Y1)
			x2s.append(X2)
			y2s.append(Y2)
	boundaries = {'x1s':x1s, 'y1s':y1s, 'x2s':x2s, 'y2s':y2s}
	np.save('boundaries.npy', boundaries)


	transform_1 = joblib.load('transform_1.sav')
	boundaries = np.load('boundaries.npy').item()
	# Find segmentation boundaries.
	specs = []
	np.random.seed(42)
	psd_f_ind, spec_f_ind = None, None
	print("Collecting data...")
	for i in tqdm(range(10000)):
		flag = True
		while flag:
			filename = filenames[np.random.randint(len(filenames))]
			psd, spec, psd_f_ind, spec_f_ind = get_rand_psd_spec(filename, psd_dur=0.12, spec_dur=0.15, psd_f_ind=psd_f_ind, spec_f_ind=spec_f_ind)
			flag = not psd_in_region(psd, transform_1, boundaries)
		specs.append(spec)
	specs = np.array(specs)
	transform_2 = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
	embedding = transform_2.fit_transform(specs.reshape(len(specs), -1))
	np.save('transform_2.npy', transform_2)
	perm = np.random.permutation(len(embedding))
	embedding, specs = embedding[perm], specs[perm]
	np.save('embedding_2.npy', embedding)
	# Make an HTML mouse-over plot.
	print("Making HTML...")
	marked_specs = get_marked_specs(specs)
	make_html_plot(embedding, marked_specs, output_dir='temp3')
	# Plot an interactive selcting tool.
	print("Interactive plot...")
	plt.scatter(embedding[:,0], embedding[:,1], c='b', alpha=0.1, s=0.7)
	ax = plt.gca()
	fig = plt.gcf()



	"""
	theta = np.arctan()
	center = (0.5(x+) , 0.5(y+))
	lines = ...
	# Other bookeeping for regions... some way to visualize regions

	# Label segmented clusters.
	specs = []
	np.random.seed(42)
	psd_f_ind, spec_f_ind = None, None
	print("Collecting data...")
	for i in range(10000):
		spec = None
		while spec is None:
			filename = filenames[np.random.randint(len(filenames))]
			spec, spec_f_ind = get_rand_segmented_spec(filename, lines, f_ind=spec_f_ind)
		specs.append(spec)

	# Review segmenting decisions.
	"""


def get_rand_psd_spec(filename, psd_dur=0.2, spec_dur=0.3, psd_f_ind=None, \
		spec_f_ind=None, return_middle_sample=False):
	audio = get_audio(filename)
	psd_samples, spec_samples = int(psd_dur * FS), int(spec_dur * FS)
	max_samples = max(psd_samples, spec_samples)
	start = np.random.randint(len(audio) - max(psd_samples, spec_samples))
	delta = (max_samples - psd_samples) // 2
	psd, psd_f_ind = get_psd(audio[start+delta:start+delta+psd_samples], f_ind=psd_f_ind)
	delta = (max_samples - spec_samples) // 2
	spec, spec_f_ind = get_spec(audio[start+delta:start+delta+spec_samples], f_ind=spec_f_ind)
	if return_middle_sample:
		return psd, spec, psd_f_ind, spec_f_ind, start+psd_samples//2
	return psd, spec, psd_f_ind, spec_f_ind


def get_rand_segmented_spec(filename, lines, transform, boundaries, \
			psd_dur=0.2, spec_dur=0.3, psd_spec_ind=None, spec_f_ind=None, \
			max_syll_dur=1.5):
	"""

	Returns
	-------
	spec : numpy array or None
		spectrogram of a single syllable or None if none is found.
	"""
	# Get a psd
	psd, _, psd_f_ind, spec_d_ind, mid = get_rand_psd_spec(filename, \
			psd_dur=psd_dur, spec_dur=spec_dur, psd_spec_ind=psd_spec_ind, \
			spec_f_ind=spec_f_ind, return_middle_sample=True)
	# If there's no nearby birdsong, give up.
	if not psd_in_region(psd, transform, boundaries):
		return None
	# Otherwise, find the nearest syllable.
	max_syll_samples = int(max_syll_dur * FS)
	# First search left, looking for local minima in distances to segmenting lines.
	distances = {} # NOTE: TEMP to see how stable these are.
	for i in range(max(0,mid-max_syll_samples), min(len(audio)-1, mid+max_syll_samples), 1):
			distances[i] = get_distance(filename) # NOTE: HERE



def get_spec(audio, min_freq=300, max_freq=12e3, f_ind=None):
	"""Get a spectrogram."""
	# Get the spectrogram.
	f, t, Zxx = stft(audio, fs=FS)
	# Cut out the frequencies we don't care about.
	if f_ind is None:
		f_ind = np.searchsorted(f, [min_freq, max_freq])
	f = f[f_ind[0]:f_ind[1]]
	Zxx = Zxx[f_ind[0]:f_ind[1]]
	# Set a threshold & normalize.
	Zxx = np.log((np.abs(Zxx) + EPSILON))
	Zxx[Zxx < SPEC_THRESH] = 0.0
	Zxx /= np.max(Zxx) + EPSILON
	return Zxx, f_ind


def get_psd(audio, min_freq=300, max_freq=12e3, f_ind=None):
	"""Get a power spectral density."""
	# Get the PSD.
	f, Pxx = welch(audio, fs=FS)
	# Cut out the frequencies we don't care about.
	if f_ind is None:
		f_ind = np.searchsorted(f, [min_freq, max_freq])
	f = f[f_ind[0]:f_ind[1]]
	Pxx = Pxx[f_ind[0]:f_ind[1]]
	return Pxx, f_ind


def make_html_plot(embedding, specs, output_dir='temp', num_imgs=2000, title=""):
	"""Make an HTML tooltip mouse-over plot."""
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


def line_select_callback(eclick, erelease):
	"""https://matplotlib.org/examples/widgets/rectangle_selector.html"""
	# eclick and erelease are the press and release events
	global X1, Y1, X2, Y2
	X1, Y1 = eclick.xdata, eclick.ydata
	X2, Y2 = erelease.xdata, erelease.ydata


def toggle_selector(event):
	"""https://matplotlib.org/examples/widgets/rectangle_selector.html"""
	print(' Key pressed.')
	if event.key in ['Q', 'q'] and toggle_selector.RS.active:
		print(' RectangleSelector deactivated.')
		toggle_selector.RS.set_active(False)
	if event.key in ['A', 'a'] and not toggle_selector.RS.active:
		print(' RectangleSelector activated.')
		toggle_selector.RS.set_active(True)


def get_audio(filename):
	"""Retrieve audio from a given file."""
	if filename[-4:] == '.wav':
		fs, audio = wavfile.read(filename)
	elif filename[-4:] == '.mat':
		temp = loadmat(filename)
		audio = temp['spike2Chunk'].reshape(-1)
		fs = temp['fs']
	else:
		raise NotImplementedError
	assert fs == FS, "found "+str(fs)+", required " + str(FS)
	return audio


def psd_in_region(psd, transform, b):
	"""Does this PSD contain relevant audio?"""
	embed = transform.transform(psd.reshape(1,-1)).flatten()
	for x1, y1, x2, y2 in zip(b['x1s'], b['y1s'], b['x2s'], b['y2s']):
		if embed[0]>min(x1,x2) and embed[0]<max(x1,x2) and embed[1]>min(y1,y2) and embed[1]<max(y1,y2):
			return True
	return False


def get_marked_specs(specs):
	"""Mark the center in time."""
	marked_specs = np.zeros(shape=(specs.shape[0], specs.shape[1]+4, specs.shape[2]))
	marked_specs[:,2:-2,:] = specs[:,:,:]
	marked_specs[:,:2,specs.shape[2]//2] = 1.2
	marked_specs[:,-2:,specs.shape[2]//2] = 1.2
	return marked_specs


def onclick(event):
	if event.button == 3:
		global LINES
		x = event.xdata
		y = event.ydata
		if len(LINES) == 0 or len(LINES[-1]) % 4 == 0:
			LINES.append([x,y])
		else:
			LINES[-1].append(x,y)
			x1, y1, x2, y2 = LINES[-1]
			plt.plot([x1,x1], [y1,y2], '-')
			radius = 0.5 * ((x1-x2)**2 + (y1-y2)**2)**0.5
			circle = plt.Circle((0.5*(x1+x2), 0.5*(y1+y2)), radius, color='r', alpha=0.3)
			plt.gca().add_artist(circle)
			plt.show()
			# fig.canvas.draw()?


if __name__ == '__main__':
	load_dirs = ['data/raw/bird_data/BF'+str(i).zfill(2) for i in range(1,44)]
	# load_dirs = ['data/raw/bird_data/79']
	segment(load_dirs)



###
