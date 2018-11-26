from __future__ import print_function, division
"""
Minimal working example for generative modeling of acoustic syllables.

0) Insert directories of .wav files in data/raw/ . Optionally include metadata
	saved as a python dictionary in <meta.npy>.
1) Tune segmenting parameters.
2) Segment audio into syllables.
3) Train a generative model on these syllables.
4) Use the model to get a latent representation of these syllables.
5) Visualize these latent representations.
6) Generate novel audio.

BEFORE RUNNING:
- change load and save directories below.
- look at time_from_filename in preprocessing/preprocessing.py

TO DO:
- Thresholds not being passed
- Random seed
- More efficient dataloader implementation
"""
__author__ = "Jack Goffinet"
__date__ = "November 2018"

import numpy as np

spec_shape = (128,128)

preprocess_params = {
	'fs':44100.0,
	'min_freq':300,
	'max_freq':12e3,
	'min_dur':0.05, # NOTE: translate to time bins later on
	'max_dur':0.3,
	'num_freq_bins':spec_shape[0],
	'num_time_bins':spec_shape[1],
	'meta': {},
}
# Set the dimension of the latent space.
latent_dim = 64
nf = 8
encoder_conv_layers =	[
		[1,1*nf,3,1,1],
		[1*nf,1*nf,3,2,1],
		[1*nf,2*nf,3,1,1],
		[2*nf,2*nf,3,2,1],
		[2*nf,3*nf,3,1,1],
		[3*nf,3*nf,3,2,1],
		[3*nf,4*nf,3,1,1],
]
# [in_features, out_features]
encoder_dense_layers =	[
		[2**13, 2**10],
		[2**10, 2**8],
		[2**8, 2**6],
		[2**6, latent_dim],
]

network_dims = {
		'input_shape':spec_shape,
		'input_dim':np.prod(spec_shape),
		'latent_dim':latent_dim,
		'post_conv_shape':(4*nf,16,16),
		'post_conv_dim':np.prod([4*nf,16,16]),
		'encoder_conv_layers':encoder_conv_layers,
		'encoder_fc_layers':encoder_dense_layers,
		'decoder_convt_layers':[[i[1],i[0]]+i[2:] for i in encoder_conv_layers[::-1]],
		'decoder_fc_layers':[i[::-1] for i in encoder_dense_layers[::-1]],
}


# Visualize some stuff.
from plotting.longitudinal_gif import make_gif, make_kde_gif, make_projection
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders

save_directories = ['data/processed/bird_data/'+str(i)+'/' for i in range(50,61,1)]

partition = get_partition(save_directories, split=1.0)
loader, _ = get_data_loaders(partition, shuffle=(False,False), time_shift=(False, False))
model = DLGM(network_dims, load_dir='data/models/sam/')

make_kde_gif(loader, model)
quit()


"""
# 1) Tune segmenting parameters.
load_dirs = ['data/raw/bird_data/'+str(i)+'/' for i in range(50,61,1)]
from preprocessing.preprocessing import tune_segmenting_params
seg_params = tune_segmenting_params(load_dirs, **preprocess_params)
preprocess_params['seg_params'] = seg_params

quit()
"""


seg_params = {'th_1':0.12, 'th_2':0.12, 'th_3':0.0, 'min_var':0.2}
preprocess_params['seg_params'] = seg_params
# 2) Segment audio into syllables.
from preprocessing.preprocessing import process_sylls
load_directories = ['data/raw/bird_data/'+str(i)+'/' for i in range(52,61,1)] # TEMP
save_directories = ['data/processed/bird_data/'+str(i)+'/' for i in range(52,61,1)]
for load_dir, save_dir in zip(load_directories, save_directories):
	try:
		preprocess_params['meta'] = np.load(load_dir + 'meta.npy').item()
	except FileNotFoundError:
		pass
	process_sylls(load_dir, save_dir, **preprocess_params)



load_directories = ['data/raw/bird_data/'+str(i)+'/' for i in range(50,61,1)]
save_directories = ['data/processed/bird_data/'+str(i)+'/' for i in range(50,61,1)]
quit()


# 3) Train a generative model on these syllables.
from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
partition = get_partition(save_directories, split=0.8)
model = DLGM(network_dims, partition=partition, save_dir='data/models/sam/')
model.train(epochs=100)

quit()

from models.dlgm import DLGM
from models.dataset import get_partition, get_data_loaders
model = DLGM(network_dims, load_dir='data/models/sam/')

# 4) Use the model to get a latent representation of these syllables.
save_directories = ['data/processed/bird_data/'+str(i)+'/' for i in range(59,61,1)]

partition = get_partition(save_directories, split=1.0)
loader, _ = get_data_loaders(partition, shuffle=(False,False), time_shift=(False, False))
latent = model.get_latent(loader, n=9000)
np.save('latent.npy', latent)


# 5) Visualize these latent representations.
import umap
from sklearn.manifold import TSNE
# transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean')
transform = TSNE(n_components=2, n_iter=1000)
embedding = transform.fit_transform(latent)
np.save('embedding.npy', embedding)


embedding = np.load('embedding.npy')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
x = embedding[:,0]
y = embedding[:,1]
plt.scatter(x, y, alpha=0.3, s=0.5)
axes = plt.gca()
plt.savefig('temp.pdf')
quit()

# 6) Generate novel audio.
from generate.inversion import invert_spec
interp = np.linspace(latent[0], latent[1], 10) # interpolate between two syllables.
specs = [model.decode(point) for point in interp]
audio = np.concatenate([invert_spec(i) for i in specs])
from scipy.io import wavfile
wavfile.write('interpolation.wav', params['fs'], audio)


if __name__ == '__main__':
	pass
