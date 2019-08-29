"""
Remove noise from segmenting files.

NOTE: needs to be tested - 8/29/19
"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import umap

from ava.plotting.tooltip_plot import tooltip_plot
from ava.plotting.latent_projection import latent_projection
from ava.segmenting.utils import get_spec, get_audio_seg_filenames, \
		get_onsets_offsets_from_file


def clean_segments(seg_dirs, audio_dirs, out_seg_dirs, p, n_samples=None, \
	num_imgs=2000):
	"""
	Manually remove noise by selecting regions of a UMAP projection.

	Parameters
	----------
	seg_dirs :
		...

	out_seg_dirs :
		....

	p :
		...

	n_samples :
		...
	"""
	specs, max_len, _ = get_specs(audio_dirs, seg_dirs, p, n_samples=n_samples)
	specs = np.stack(specs)
	transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
		metric='euclidean', random_state=42)
	embed = transform.fit_transform(specs.reshape(len(specs), -1))
	bounds = {'x1': [], 'x2': [], 'y1': [], 'y2': []}
	colors = ['b'] * len(embed)
	first_iteration = True
	# Keep drawing boxes around noise.
	while True:
		draw_plot(embed, colors)
		if first_iteration:
			first_iteration = False
			title = "Select unwanted sounds:"
			tooltip_plot(embed, specs, num_imgs=num_imgs, title=title)
		if input("Press [q] to quit drawing rectangles: ") == 'q':
			break
		print("Select a rectangle containing noise:")
		x1 = get_input("x1: ")
		x2 = get_input("x2: ")
		y1 = get_input("y1: ")
		y2 = get_input("y2: ")
		bounds['x1'].append(x1)
		bounds['x2'].append(x2)
		bounds['y1'].append(y1)
		bounds['y2'].append(y2)
		# Update scatter colors.
		colors = update_colors(colors, embed, bounds)
	# Write files to out_seg_dirs.
	for seg_dir, audio_dir, out_seg_dir in \
				zip(seg_dirs, audio_dirs, out_seg_dirs):
		specs, _, all_fns = \
				get_specs([audio_dir], [seg_dir], p, max_len=max_len)
		specs = np.stack(specs)
		embed = transform.transform(specs.reshape(len(specs), -1))
		out_segs = []
		prev_fn, prev_segs = None, None
		for i in range(len(all_fns)):
			if all_fns[i] != prev_fn:
				if len(out_segs) > 0:
					audio_fn = os.path.join(audio_dir, prev_fn)
					out_seg_fn = os.path.join(out_seg_dir, prev_fn)
					write_segs(out_segs, out_seg_fn, audio_fn)
					out_segs = []
				prev_fn = all_fns[i]
				prev_segs = np.loadtxt(os.path.join(seg_dir, prev_fn))
			if not in_bounds(embeds[i], bounds):
				out_segs.append()
		if len(out_segs) > 0:
			audio_fn = os.path.join(audio_dir, prev_fn)
			out_seg_fn = os.path.join(out_seg_dir, prev_fn)
			write_segs(out_segs, out_seg_fn, audio_fn)


def get_specs(audio_dirs, seg_dirs, p, n_samples=None, max_len=None):
	"""
	Make a bunch of spectrograms.

	Parameters
	----------

	Returns
	-------

	"""
	# Get the filenames.
	audio_fns, seg_fns = get_audio_seg_filenames(audio_dirs, seg_dirs)
	# Collect spectrograms.
	specs, all_fns = [], []
	for audio_fn, seg_fn in zip(audio_fns, seg_fns):
		onsets, offsets = get_onsets_offsets_from_file(seg_fn)
		fs, audio = wavfile.read(audio_fn)
		for onset, offset in zip(onsets, offsets):
			i1, i2 = int(onset * fs), int(offset * fs)
			assert i1 >= 0 and i2 <= len(audio), audio_fn + ", " + seg_fn
			spec, _, _ = get_spec(audio[i1:i2], p)
			specs.append(spec)
			all_fns.append(os.path.split(seg_fn)[-1])
			if len(specs) == n_samples:
				break
	# Zero-pad.
	assert len(specs) > 0, "Found no spectrograms!"
	n_freq_bins = specs
	if max_len is None:
		max_len = max(spec.shape[1] for spec in specs)
	for i in range(len(specs)):
		spec = np.zeros((n_freq_bins, max_len))
		spec[:,:len(specs[i])] = specs[i][:,:max_len]
		specs[i] = spec
	return specs, max_len, all_fns


def draw_plot(embed, colors, title=""):
	"""Helper function to plot a UMAP projection with grids."""
	plt.scatter(embed[:,0], embed[:,1], c=colors, s=0.9, alpha=0.7)
	delta = 1
	if np.max(embed) - np.min(embed) > 20:
		delta = 5
	min_xval = int(np.floor(np.min(embed[:,0])))
	if min_xval % delta != 0:
		min_xval -= min_xval % delta
	max_xval = int(np.ciel(np.max(embed[:,0])))
	if max_xval % delta != 0:
		max_xval -= (max_xval % delta) - delta
	min_yval = int(np.floor(np.min(embed[:,1])))
	if min_yval % delta != 0:
		min_yval -= min_yval % delta
	max_yval = int(np.ciel(np.max(embed[:,1])))
	if max_yval % delta != 0:
		max_yval -= (max_yval % delta) - delta
	for x_val in range(min_xval, max_xval+1):
		plt.axvline(x=x_val, lw=0.5, alpha=0.7)
	for y_val in range(min_yval, max_yval+1):
		plt.axhline(y=y_val, lw=0.5, alpha=0.7)
	plt.title(title)
	plt.savefig('temp.pdf')
	plt.close('all')


def write_segs(segs, out_fn, header_fn):
	"""
	Write onstes/offsets to a text file.

	Parameters
	----------
	segs :
		...

	out_fn :
		...

	header_fn :
		...
	"""
	segs = np.stack([np.array(seg) for seg in segs])
	header = "Cleaned onsets/offsets for " + header_fn
	np.savetxt(fn, segs, fmt='%.5f', header=header)


def get_input(query_str):
	"""Get float-valued input."""
	while True:
		try:
			temp = float(input(query_str))
			return temp
		except:
			print("Unrecognized input!")
			pass


def update_colors(colors, embed, bounds):
	"""Color red if embed is in the bounds, blue otherwise."""
	for i in range(len(colors)):
		if colors[i] == 'b' and in_bounds(embed[i], bounds):
			colors[i] = 'r'
	return colors


def in_bounds(point, bounds):
	"""Is the point in the given rectangular bounds?"""
	for i in range(len(bounds['x1'])):
		if point[0] > bounds['x1'] and point[0] < bounds['x2'] and \
				point[1] > bounds['y1'] and point[1] < bounds['y2']:
			return True
	return False


if __name__ == '__main__':
	pass


###
