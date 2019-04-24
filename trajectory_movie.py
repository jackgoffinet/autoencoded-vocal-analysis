"""
Make a birdsong trajectory video.
"""
__author__ = "Jack Goffinet"
__date__ = "April 2019"

import numpy as np
import imageio
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import umap
import joblib
from scipy.io import wavfile
from scipy.interpolate import interp1d

from preprocessing.interactive_segmentation import get_spec, FS


start, stop = 342.0, 356.0
spec_dur = 0.5
delta_t = 0.01
skip_num = 5

slow_factor = 1

print("real frame rate:", 1.0/delta_t)
print("skip real frame rate:", 1.0/delta_t/skip_num)
print("slowed frame_rate", 1.0/delta_t/slow_factor)
print("skip frame_rate", 1.0/delta_t/slow_factor/skip_num)

MIN_FREQ, MAX_FREQ = 350, 13e3

transform = joblib.load('transform_2.sav')

fs, audio = wavfile.read('001.wav')
assert fs == FS
wavfile.write('out.wav', fs, audio[int(fs*(start+spec_dur)):int(fs*stop)] / np.max(np.abs(audio)))
audio = audio[int(fs*start):int(fs*stop)]

print("getting specs")
t = 0.0
f_ind = None
specs = []
while t + spec_dur + 0.01 < len(audio)/fs:
	i1, i2 = int(fs*t), int(fs*(t+spec_dur))
	spec, f_ind = get_spec(audio[i1:i2], f_ind=f_ind, min_freq=MIN_FREQ, max_freq=MAX_FREQ)
	specs.append(spec)
	t += delta_t
print("f_ind", f_ind)

specs = np.array(specs)
print("specs shape", specs.shape)
specs = specs.reshape(len(specs), -1)

print("transforming specs")
embedding = transform.transform(specs)


X = embedding[:,0]
Y = embedding[:,1]


embedding = np.load('embedding_2.npy')
eX, eY = embedding[:,0], embedding[:,1]


xmin = np.min(eX) - 1.0
xmax = np.max(eX) + 1.0
ymin = np.min(eY) - 1.0
ymax = np.max(eY) + 1.0

# fX = interp1d(np.arange(len(X)), X)
# fY = interp1d(np.arange(len(X)), Y)
print("plotting")
for i in range(0,len(X),skip_num):
	plt.scatter(eX,eY,c='k', alpha=0.05, s=0.9)
	t_vals = (np.arange(3*i)/3.0)[-100:]
	# plt.plot(fX(t_vals), fY(t_vals), c='b', alpha=0.2)
	plt.plot(X[:i+1], Y[:i+1], c='r', alpha=0.6, lw=0.5)
	plt.scatter([X[i]], [Y[i]], c='r', s=60.0, marker='*')
	plt.axis('off')
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	plt.savefig('movie_imgs/'+str(i//skip_num)+'.jpg')
	plt.close('all')

# images = []
# for i in range(0,10**6,skip_num):
# 	try:
# 		image = imageio.imread('movie_imgs/' + str(i) + '.jpg')
# 		images.append(image)
# 	except:
# 		break
# imageio.mimsave('temp.avi', images, duration=skip_num*delta_t)

"""
ffmpeg -r 13.3 -i movie_imgs/%d.jpg temp_slow.mp4
ffmpeg -i temp_slow.mp4 -i out_slow.wav -acodec aac -strict experimental temp_slow_audio.mp4
"""
