from __future__ import print_function, division
"""
Minimal working example for generative modeling of acoustic syllables.

0) Insert directories of .wav files in data/raw/ . Optionally include metadata
   as python dictionaries saved in .npy format.
   
1) Segment audio into syllables.
2) Train a generative model on these syllables.
3) Use the model to get a latent representation of these syllables.
4) Visualize these latent representations.
5) Generate novel audio.

"""
__author__ = "Jack Goffinet"
__date__ = "November 2018"

import numpy as np
from os import listdir


params = {
        'segmenting_timescale':0.1,
        'spectrogram_shape': (128,128),
        'latent_dimension': 16,
}

# 1) Segment audio into syllables.
from preprocessing.amplitude_segmentation import Sylls
load_directories = ['data/raw/Bird0']
save_directories = ['data/processed/Bird0']
for load_dir, save_dir in zip(load_directories, save_directories):
    syllables = Sylls(load_dir, *params)
    syllables.save(save_dir)

# 2) Train a generative model on these syllables.
from models.dlgm import Dlgm
model = Dlgm(*params) # or ConvVAE(*params)
train_loader, test_loader = models.get_data_loaders(save_directories, split=0.8)
model.train(epochs=100, save_dir='data/models/mwe/')

# 3) Use the model to get a latent representation of these syllables.
loader = models.get_single_loader(save_directories)
latent = model.get_latent(loader)

# 4) Visualize these latent representations.
import umap
transform = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='euclidean')
embedding = transform.fit_transform(latent)

# 5) Generate novel audio.
interp = np.linspace(latent[0], latent[1]) # interpolate between two syllables.
specs = model.generate(interp)
audio = np.concatenate([invert(i) for i in specs])
from scipy.io import wavfile
wavfile.write('interpolation.wav', params['fs'], audio)
