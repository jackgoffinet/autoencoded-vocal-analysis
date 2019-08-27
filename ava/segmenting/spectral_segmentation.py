"""
Segment audio based on spectral properties.


"""
__author__ = "Jack Goffinet"
__date__ = "June 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from scipy.io import wavfile
from scipy.signal import stft
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC





def train_classifier(audio_fns, syll_fns, save_dir):
	# Collect training data.
	Xs, ys = [], []
	for audio_fn, syll_fn in zip(audio_fns, syll_fns):
		fs, audio = wavfile.read(audio_fn)
		f, t, spec = stft(audio, fs=fs)
		spec = spec[:spec.shape[0]//2]
		dt = t[1] - t[0]
		spec = np.abs(spec)
		Xs.append(spec.T)
		syll_times = np.loadtxt(syll_fn)
		y = np.zeros(spec.shape[1], dtype='int')
		for ts in syll_times:
			i1, i2 = int(round(ts[0]/dt)), int(round(ts[1]/dt))
			y[i1:i2+1] = 1
		ys.append(y)
	X = np.concatenate(tuple(Xs))
	X /= np.max(X)
	y = np.concatenate(tuple(ys))
	# Train a model.
	# clf = LogisticRegression(random_state=0, solver='lbfgs', \
	# 		class_weight=None).fit(X, y)
	print("fit")
	clf = SVC(max_iter=7000).fit(X,y)

	i1, i2 = -8000, -4000
	fig = plt.figure()
	ax = fig.add_subplot(2, 1, 1)
	ax.imshow(X[i1:i2].T, origin='lower', aspect='auto')
	ax = fig.add_subplot(2, 1, 2)
	print("predict")
	temp = clf.predict(X[i1:i2])
	print(temp.shape)
	ax.plot(temp)
	plt.savefig('temp.pdf')




if __name__ == '__main__':
	dir = 'data/classifier_training/grn288_'
	audio_fns = [dir+str(i)+'.wav' for i in [1,2,3]]
	syll_fns = [dir+str(i)+'.txt' for i in [1,2,3]]
	train_classifier(audio_fns, syll_fns, None)




###
