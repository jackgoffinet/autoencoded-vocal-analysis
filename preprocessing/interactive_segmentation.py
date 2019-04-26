"""
Interactive syllable segmentation

1) Relevant audio?
2) Boundary?
3) Good clusters?
"""

__author__ = "Jack Goffinet"
__date__ = "March-April 2019"

import os
import numpy as np
import h5py
import joblib
from tqdm import tqdm
from skimage.transform import resize, downscale_local_mean
from scipy.io import wavfile, loadmat
from scipy.signal import stft, welch

from time import sleep

# import matplotlib
# matplotlib.use('qt4agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
# plt.switch_backend('agg')

import umap

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models.glyphs import ImageURL


from preprocessing import time_from_filename


FS = 44100
SPEC_THRESH = 1.0 # -8.5 homer, 4.0 mouse, -9 marmoset, -3.5 zebra finch
EPSILON = 1e-9

DOWNSAMPLE_TUPLE = (1,1) # marmoset (1,2)

MIN_FREQ, MAX_FREQ = 300, 12e3 # 40k-110k mouse, 350-13k marmoset

PSD_DUR, SPEC_DUR = 0.5, 0.5 # 0.1 mouse, 0.5 marmoset, 0.5 zebra finch

CENTER_T = 0.5 # how far into the song are we looking?
SONG_DURATION = 2.0 # how long is the song?

X1, Y1, X2, Y2 = 0, 0, 0, 0
LINES = []


def segment(load_dirs):
	""" Main method """
	global X1, Y1, X2, Y2
	# Collect filenames.
	filenames = []
	for load_dir in load_dirs:
		filenames += [os.path.join(load_dir,i) for i in os.listdir(load_dir) if i[-4:] in ['.wav', '.mat']]

	"""
	run_num = 0
	while True:
		run_num += 1
		# Collect PSDs and spectrograms.
		psds, specs = [], []
		np.random.seed(42)
		psd_f_ind, spec_f_ind = None, None
		print("Collecting data...")

		# from multiprocessing import Pool
		# from itertools import repeat
		# with Pool(min(3, os.cpu_count()-1)) as pool:
		# 	pool.starmap(get_rand_psds_specs, zip())
		# # quit()

		for i in tqdm(range(5000)): #10000

			flag = True
			while flag:
				filename = filenames[np.random.randint(len(filenames))]
				psd, spec, psd_f_ind, spec_f_ind = get_rand_psd_spec(filename, psd_dur=PSD_DUR, spec_dur=SPEC_DUR, psd_f_ind=psd_f_ind, spec_f_ind=spec_f_ind)
				if run_num > 1:
					flag = not psd_in_region(spec, transform, boundaries)
				else:
					flag = False
			# print("psd", psd.shape)
			# print("spec", spec.shape)
			# print(np.min(spec), np.max(spec))
			# plt.imshow(spec, aspect='auto', origin='lower')
			# plt.savefig('temp.pdf')
			# plt.close('all')
			# sleep(2)
			# if i == 10:
			# 	quit()
			psds.append(psd)
			specs.append(spec)
		psds = np.array(psds)
		np.save('psds.npy', psds)
		specs = np.array(specs)
		np.save('specs.npy', specs)
		# UMAP the PSDs.
		print("Dimensionality reduction...")
		transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
		embedding = transform.fit_transform(specs.reshape(len(specs), -1))
		joblib.dump(transform, 'transform_'+str(run_num)+'.sav')
		perm = np.random.permutation(len(embedding))
		embedding, specs = embedding[perm], specs[perm]
		np.save('embedding.npy', embedding)

		# Make an HTML mouse-over plot.
		print("Making HTML...")
		make_html_plot(embedding, specs, output_dir='temp2', num_imgs=400)
		embedding = np.load('embedding.npy')

		# Non-interactive:
		plt.title("Select relevant audio")
		plt.scatter(embedding[:,0], embedding[:,1], c='b', alpha=0.1, s=0.7)
		plt.savefig('temp.pdf')
		plt.close('all')
		x1 = float(input("x1: "))
		x2 = float(input("x2: "))
		y1 = float(input("y1: "))
		y2 = float(input("y2: "))
		x1s, x2s, y1s, y2s = [x1], [x2], [y1], [y2]

		boundaries = {'x1s':x1s, 'y1s':y1s, 'x2s':x2s, 'y2s':y2s}
		np.save('boundaries.npy', boundaries)


		if input("q for quit ") == 'q':
			quit()

	"""

	"""
	# Plot an interactive selcting tool.
	print("Interactive plot...")
	x1s, y1s, x2s, y2s = [], [], [], []
	colors = np.array(['b'] * len(embedding))
	X1 = 0
	while X1 is not None:
		print("here")
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
		plt.title("Select relevant audio")
		plt.connect('key_press_event', toggle_selector)
		print("making plot 1")
		plt.scatter(embedding[:,0], embedding[:,1], c=colors, alpha=0.1, s=0.7)
		plt.show()
		sleep(20)
		print("making plot 2")
		if X1 is not None:
			x1s.append(X1)
			y1s.append(Y1)
			x2s.append(X2)
			y2s.append(Y2)
		"""

	"""
	boundaries = {'x1s':x1s, 'y1s':y1s, 'x2s':x2s, 'y2s':y2s}
	np.save('boundaries.npy', boundaries)
	"""


	"""
	PART II
	-------
	"""

	"""
	transform = joblib.load('transform_'+str(run_num)+'.sav')
	boundaries = np.load('boundaries.npy').item()

	# Find segmentation boundaries.
	specs = []
	np.random.seed(42)
	psd_f_ind, spec_f_ind = None, None
	print("Collecting data...")

	# # Collect non-random spectrograms.
	# delta_t = 0.025
	# fs, audio = wavfile.read('001.wav')
	# t = 0.0
	# f_ind = None
	# specs = []
	# print("len(audio)", len(audio)/fs)
	# while t + SPEC_DUR + 0.01 < min(len(audio)/fs, 2000):
	# 	i1, i2 = int(fs*t), int(fs*(t+SPEC_DUR))
	# 	spec, f_ind = get_spec(audio[i1:i2], f_ind=f_ind, min_freq=MIN_FREQ, max_freq=MAX_FREQ)
	# 	specs.append(spec)
	# 	t += delta_t
	# print("f_ind", f_ind)

	# Collect random spectrograms.
	for i in tqdm(range(5000)): # 100000
		flag = True
		while flag:
			filename = filenames[np.random.randint(len(filenames))]
			psd, spec, psd_f_ind, spec_f_ind = get_rand_psd_spec(filename, psd_dur=PSD_DUR, spec_dur=SPEC_DUR, psd_f_ind=psd_f_ind, spec_f_ind=spec_f_ind)
			flag = not psd_in_region(spec, transform, boundaries)
		specs.append(spec)
		if (i + 1) % 500 == 0:
			np.save('specs.npy', np.array(specs))
	print("spec_f_ind:", spec_f_ind)

	specs = np.array(specs)
	np.save('specs.npy', specs)

	print('specs', specs.shape)
	# weighted_specs = np.copy(specs)
	# for i in range(len(specs)):
	# 	for j in range(len(specs[i])):
	# 		weighted_specs[i,j,:specs.shape[2]//2] *= np.linspace(0,1,specs.shape[2]//2)
	# 		weighted_specs[i,j,specs.shape[2]//2:] *= np.linspace(1,0,specs.shape[2]-specs.shape[2]//2)
	print("Dimensionality reduction...")
	transform_final = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean', random_state=42)
	print("specs shape:", specs.shape)
	embedding = transform_final.fit_transform(specs.reshape(len(specs), -1))
	joblib.dump(transform_final, 'transform_2.sav')
	perm = np.random.permutation(len(embedding))
	embedding, specs = embedding[perm], specs[perm]
	np.save('embedding_2.npy', embedding)

	# Make an HTML mouse-over plot.
	print("Making HTML...")
	marked_specs = get_marked_specs(specs)
	make_html_plot(embedding, marked_specs, output_dir='temp3')

	# Non-interactive:
	plt.scatter(embedding[:,0], embedding[:,1], c='b', alpha=0.1, s=0.7)
	plt.savefig('temp.pdf')
	plt.close('all')
	center_x = float(input("center x: "))
	center_y = float(input("center y: "))
	radius = float(input("radius: "))
	radii = [radius]
	centers = [[center_x, center_y]]
	"""


	"""
	# Plot an interactive selcting tool.
	print("Interactive plot...")
	x1s, y1s, x2s, y2s = [], [], [], []
	radii = []
	centers = []
	X1 = 0
	while X1 is not None:
		X1 = None
		fig, current_ax = plt.subplots()
		toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
				drawtype='line', useblit=True,
				button=[1, 3],  # don't use middle button
				minspanx=0, minspany=0,
				spancoords='pixels',
				interactive=True)
		plt.title("Select relevant audio")
		plt.connect('key_press_event', toggle_selector)
		plt.scatter(embedding[:,0], embedding[:,1], c='b', alpha=0.1, s=0.7)
		for x1, y1, x2, y2, center, radius in zip(x1s, y1s, x2s, y2s, centers, radii):
			plt.plot([x1,x2], [y1,y2], c='r')
			circle = plt.Circle(tuple(center), radius, color='r', fill=False)
			current_ax.add_artist(circle)
		plt.show()
		if X1 is not None:
			x1s.append(X1)
			y1s.append(Y1)
			x2s.append(X2)
			y2s.append(Y2)
			centers.append([(X1+X2)/2,(Y1+Y2)/2])
			radii.append(0.5 * ((X1-X2)**2+(Y1-Y2)**2)**0.5)
	"""

	# TEMP!
	centers = [[12.5,7]]
	radii = [2]


	seg_stuff = {'radii':radii, 'centers':np.array(centers)}
	np.save('seg_stuff.npy', seg_stuff)


	# Parse through all the files, collecting
	seg_stuff = np.load('seg_stuff.npy').item()
	radius = seg_stuff['radii'][0]
	center = seg_stuff['centers'][0]

	transform = joblib.load('transform_2.sav')

	# Segment a file.
	segment_audio('data/raw/bird_data/79', 'temp_save', transform, center, radius)
	quit()
	fs, audio = wavfile.read('data/raw/bird_data/79/red291_43566.27672737_4_11_7_41_12.wav')
	print(filenames[0])
	temp = segment_audio_chunk(audio, transform, center, radius)
	print(temp)
	quit()


def segment_audio(load_dir, save_dir, transform, center, radius):
	"""Save all the song bouts."""
	load_filenames = [os.path.join(load_dir,i) for i in os.listdir(load_dir) if i[-4:] in ['.wav', '.mat']]
	# load_filenames = ['data/raw/bird_data/79/red291_43566.27672737_4_11_7_41_12.wav']
	audio_segs, filenames, file_times, times = [], [], [], []
	num_audio_frames = int(SONG_DURATION*FS)
	write_file_num = 0
	songs_per_file = 10
	max_num_songs = 2000
	for load_filename in load_filenames:
		print(load_filename, len(audio_segs))
		filename_time = time_from_filename(load_filename)
		start_t = 0.0
		center_ts = []
		audio = get_audio(load_filename)
		while True:
			result = segment_audio_chunk(audio[int(FS*start_t):], transform, center, radius)
			if result == -1:
				continue
			if result is None:
				break
			center_ts.append(result + start_t)
			start_t += result + SONG_DURATION / 2
		for center_t in center_ts:
			t1 = int((center_t - CENTER_T)*FS)
			t2 = t1 + num_audio_frames
			audio_segs.append(audio[t1:t2])
			filenames.append(load_filename)
			file_times.append(center_t - CENTER_T)
			times.append(filename_time + center_t - CENTER_T)
		while len(audio_segs) >= songs_per_file:
			save_filename = os.path.join(save_dir, str(write_file_num).zfill(2)+'.hdf5')
			wavfile.write(os.path.join(save_dir, 'temp.wav'), FS, np.concatenate(audio_segs, axis=0))
			with h5py.File(save_filename, "w") as f:
				f.create_dataset('audio', data=np.array(audio_segs[:songs_per_file]))
				f.create_dataset('filename', data=np.array(filenames[:songs_per_file]).astype('S'))
				f.create_dataset('file_time', data=np.array(file_times[:songs_per_file]))
				f.create_dataset('time', data=np.array(times[:songs_per_file]))
			audio_segs = audio_segs[songs_per_file:]
			filenames = filenames[songs_per_file:]
			file_times = file_times[songs_per_file:]
			times = times[songs_per_file:]
			write_file_num += 1
			if write_file_num * songs_per_file >= max_num_songs:
				return



def segment_audio_chunk(audio, transform, center, radius):
	r2 = radius*radius
	f_ind = None
	STEP = 0.01
	SMALL_STEP = 0.002
	t = CENTER_T
	while t + SPEC_DUR + 0.01 < len(audio)/FS:
		embed, f_ind = get_embed_from_time(audio, t, f_ind, transform)
		current_r2 = np.sum(np.power(embed-center,2))
		# print(t, current_r2)
		if current_r2 > r2:
			t += STEP
			continue
		# Step until radius increases.
		prev_r2 = current_r2
		while True:
			t += SMALL_STEP
			embed, f_ind = get_embed_from_time(audio, t, f_ind, transform)
			current_r2 = np.sum(np.power(embed-center,2))
			# print(t, current_r2, "inside loop")
			if current_r2 > prev_r2:
				t -= SMALL_STEP
				# print(t)
				# print(CENTER_T)
				# print(t-CENTER_T+SONG_DURATION)
				# print(len(audio)/FS)
				if t - CENTER_T + SONG_DURATION < len(audio)/FS:
					return t
				return None
			prev_r2 = current_r2
	return None


def get_embed_from_time(audio, t, f_ind, transform):
	i1, i2 = int(t*FS), int((t+SPEC_DUR)*FS)
	spec, f_ind = get_spec(audio[i1:i2], f_ind=f_ind)
	return transform.transform(spec.reshape(1,-1)).flatten(), f_ind


def get_rand_psd_spec(filename, psd_dur=0.2, spec_dur=0.3, psd_f_ind=None, \
		spec_f_ind=None, return_middle_sample=False):
	audio = get_audio(filename)
	psd_samples, spec_samples = int(psd_dur * FS), int(spec_dur * FS)
	max_samples = max(psd_samples, spec_samples)
	start = np.random.randint(len(audio) - max_samples)
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



def get_spec(audio, min_freq=MIN_FREQ, max_freq=MAX_FREQ, f_ind=None, fs=FS):
	"""Get a spectrogram."""
	# Get the spectrogram.
	f, t, Zxx = stft(audio, fs=fs)
	# Cut out the frequencies we don't care about.
	if f_ind is None:
		f_ind = np.searchsorted(f, [min_freq, max_freq])
	f = f[f_ind[0]:f_ind[1]]
	Zxx = Zxx[f_ind[0]:f_ind[1]]
	# Set a threshold & normalize.
	Zxx = np.log((np.abs(Zxx) + EPSILON))
	Zxx -= SPEC_THRESH
	Zxx[Zxx < 0.0] = 0.0
	Zxx /= np.max(Zxx) + EPSILON
	Zxx = downscale_local_mean(Zxx, DOWNSAMPLE_TUPLE)
	return Zxx, f_ind


def get_psd(audio, min_freq=MIN_FREQ, max_freq=MAX_FREQ, f_ind=None):
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


if __name__ == '__main__':
	# load_dirs = ['human/']
	load_dirs = ['data/raw/bird_data/79']
	segment(load_dirs)



###
