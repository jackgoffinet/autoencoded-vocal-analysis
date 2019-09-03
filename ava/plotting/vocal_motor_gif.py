"""
Motor vocal gif

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
plt.switch_backend('agg')

from scipy.io import wavfile
from scipy.signal import stft
from scipy.interpolate import interp1d

import umap

EPS = 1e-12


def get_onsets_offsets_from_file(filename, dt):
	# filename = audio_filename.split('.')[0] + '.txt'
	d = np.loadtxt(filename)
	onsets = []
	offsets = []
	for i in range(len(d)):
		onsets.append(int(np.floor(d[i,1]/dt)))
		offsets.append(int(np.ceil(d[i,2]/dt)))
		if offsets[-1] - onsets[-1] >= 128:
			onsets = onsets[:-1]
			offsets = offsets[:-1]
	return onsets, offsets

def mel(a):
	return 1127 * np.log(1 + a / 700)


def inv_mel(a):
	return 700 * (np.exp(a / 1127) - 1)


if __name__ == '__main__':
	# filenames = ['../data/raw/marmosets/S'+str(j)+'/S'+str(j)+'_D'+str(i) for i in range(1,6,1) for j in range(1,6,1)]
	# all_freqs = []
	# rand_trajs = []
	# for filename in filenames:
	# 	fs, audio = wavfile.read(filename+'.wav')
	# 	f, t, Zxx = stft(audio, fs=fs, nperseg=1024, noverlap=512)
	# 	i1 = np.searchsorted(f, 1e3)
	# 	i2 = np.searchsorted(f, 22e3)
	# 	f = f[i1:i2]
	# 	spec = np.log(np.abs(Zxx[i1:i2,:]) + EPS)
	#
	# 	# Denoise.
	# 	spec_thresh = np.percentile(spec, 80)
	# 	spec -= spec_thresh
	# 	spec[spec<0.0] = 0.0
	#
	# 	# Switch to mel frequency spacing.
	# 	mel_f = np.linspace(mel(f[0]), mel(f[-1]), 128, endpoint=True)
	# 	mel_f = inv_mel(mel_f)
	# 	mel_f[0] = f[0] # Correct for numerical errors.
	# 	mel_f[-1] = f[-1]
	# 	mel_f_spec = np.zeros((128, spec.shape[1]), dtype='float')
	# 	for j in range(spec.shape[1]):
	# 		interp = interp1d(f, spec[:,j], kind='cubic')
	# 		mel_f_spec[:,j] = interp(mel_f)
	# 	spec = mel_f_spec
	#
	# 	dt = t[1] - t[0]
	# 	onsets, offsets = get_onsets_offsets_from_file(filename+'.txt', dt)
	#
	# 	n = 10000
	# 	freqs = np.zeros((n,128))
	# 	i = 0
	# 	for onset, offset in zip(onsets, offsets):
	# 		i1, i2 = onset, offset
	# 		if np.random.rand() < 0.01:
	# 			rand_trajs.append(spec[:,i1:i2].T)
	# 		for j in range(i1, i2, 1):
	# 			freqs[i] = spec[:,j]
	# 			i += 1
	# 			if i == n:
	# 				break
	# 		if i == n:
	# 			break
	# 	print("i = ", i)
	#
	# 	freqs = freqs[:i]
	# 	all_freqs.append(freqs)
	#
	#
	# all_freqs = np.concatenate(all_freqs, axis=0)
	# perm = np.random.permutation(len(all_freqs))
	# all_freqs = all_freqs[perm]
	# transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean')
	# embedding = transform.fit_transform(all_freqs[:10000])
	# np.save('embedding.npy', embedding)
	# np.save('rand_trajs.npy', rand_trajs)
	#
	# temp_embeds = []
	# for rand_traj in rand_trajs:
	# 	temp_embeds.append(transform.transform(rand_traj))
	# np.save('temp_embeds.npy', temp_embeds)
	# print('all saved')
	embedding, rand_trajs, temp_embeds = np.load('embedding.npy'), np.load('rand_trajs.npy'), np.load('temp_embeds.npy')

	# perm = np.random.permutation(len(onsets))[:10]
	# onsets = np.array(onsets)[perm]
	# offsets = np.array(offsets)[perm]

	plt.scatter(embedding[:,0], embedding[:,1], c='b', alpha='0.05', s=0.5)
	# cmap = cm.get_cmap('viridis',128)
	#
	# from matplotlib.collections import LineCollection
	# from matplotlib.colors import ListedColormap, BoundaryNorm
	#
	# temp_embeds = []
	# for rand_traj in rand_trajs:
	# 	temp_embeds.append(transform.transform(rand_traj))
	colors = ['b','g','r','k','y', 'c', 'm']
	for i, points_temp in enumerate(temp_embeds[:len(colors)]):
		x = points_temp[:,0]
		y = points_temp[:,1]
		print(points_temp.shape)
		plt.plot(x,y,lw=0.5,c=colors[i])

	print('saving')
	plt.savefig('temp.pdf')







###
