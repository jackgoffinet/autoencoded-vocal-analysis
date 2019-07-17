"""
DataContainer class for linking directories containing different sorts of data.

This is meant to make plotting and analysis easier.
"""
__author__ = "Jack Goffinet"
__date__ = "July 2019"


import h5py
import numpy as np
import os
from sklearn.decomposition import PCA
import torch
import umap

from ..models.vae import VAE
from ..models.vae_dataset import get_partition, get_data_loaders,
	get_hdf5s_from_dir


PROJECTION_FIELDS = ['latent_mean', 'latent_mean_pca', 'latent_mean_umap']
SPEC_FIELDS = ['specs', 'onsets', 'offsets', 'audio_filenames']
ALL_FIELDS = np.unique(np.concatenate([PROJECTION_FIELDS, SPEC_FIELDS]))



class DataContainer():
	"""
	Link directories containing different data sources for easy plotting.

	The idea is for plotting and analysis tools to get passed a DataContainer,
	from which they can request different types of data. Those requests can then
	be handled here without much redundant code and processing.

	Notes
	-----
	Intended directory structure notes...
	"""

	def __init__(self, audio_dirs=None, spec_dirs=None, feature_dirs=None, \
		projection_dirs=None, plots_dir=None, model_filename=None):
		"""
		Parameters
		----------
		audio_dirs : list of str, or None, optional
			Directories containing audio. Defaults to None.

		spec_dirs : list of str, or None, optional
			Directories containingDefaults to None.

		model_filename : str or None, optional
			The models.VAE checkpoint to load. Defaults to None.

		projection_dirs : list of str, or None, optional
			Directory containing different projections. Defaults to None.

		plots_dir : str or None, optional
			Directory containingDefaults to None.

		feature_dir : str or None, optional
			Directory containingDefaults to None.
		"""
		self.audio_dirs = audio_dirs
		self.spec_dirs = spec_dirs
		self.feature_dirs = feature_dirs
		self.projection_dirs = projection_dirs
		self.plots_dir = plots_dir
		self.model_filename = model_filename
		self.sylls_per_file = None # syllables in each hdf5 file in spec_dirs
		self.fields = {}
		if spec_dirs is not None:
			self.fields['specs'] = 1


	def request(self, field):
		"""Request some type of data."""
		assert field in ALL_FIELDS
		# If it's not here, make it and return it.
		if field not in self.fields:
			data = self.make_field(field)
			return data
		# Otherwise, read it and return it.
		return self.read_field(field)


	def make_field(self, field):
		"""Make a field."""
		if field == 'latent_means':
			data = self.make_latent_means()
		elif field == 'latent_mean_pca':
			data = self.make_latent_mean_pca_projection()
		elif field == 'latent_mean_umap':
			data = self.make_latent_mean_umap_projection()
		elif field == 'specs':
			raise NotImplementedError
		else:
			raise NotImplementedError
		# Add this field to the collection of fields that have been computed.
		self.fields[field] = 1
		return data


	def read_field(self, field):
		"""Read a field from memory."""
		if field in PROJECTION_FIELDS:
			load_dirs = self.projection_dirs
		elif field in SPEC_FIELDS:
			load_dirs = self.spec_dirs
		else:
			raise NotImplementedError
		to_return = []
		for i in range(len(self.spec_dirs)):
			spec_dir, load_dir = spec_dirs[i], load_dirs[i]
			hdf5s = get_hdf5s_from_dir(spec_dir)
			for j, hdf5 in enumerate(hdf5s):
				filename = os.path.join(load_dir, os.path.split(hdf5)[-1])
				with h5py.File(filename, 'r') as f:
					to_return.append(f[field])
		return np.concatenate(to_return)


	def make_latent_means(self):
		"""
		Write latent means for the syllables in self.spec_dirs.

		Returns
		-------
		latent_means : numpy.ndarray
			Latent means of shape (max_num_syllables, z_dim)

		NOTE
		----
		- Test this.
		- Duplicated code with <write_projection>?
		"""
		# First, see how many syllables are in each file.
		hdf5_file = get_hdf5s_from_dir(self.spec_dirs[0])[0]
		with h5py.File(hdf5_file, 'r') as f:
			self.sylls_per_file = len(f['specs'])
		# Load the model, making sure to get z_dim correct.
		z_dim = torch.load(self.model_filename)['z_dim']
		model = VAE(z_dim=z_dim)
		model.load_state(self.model_filename)
		# For each directory...
		all_latent = []
		for i in range(len(self.spec_dirs)):
			spec_dir, proj_dir = self.spec_dirs[i], self.projection_dirs[i]
			# Make a DataLoader for the syllables.
			partition = get_partition([spec_dir], 1, shuffle=False)
			loader = get_data_loaders(partition, shuffle=(False,False))['train']
			# Get the latent means from the model.
			latent_means = model.get_latent(loader)
			all_latent.append(latent_means)
			# Write them to the corresponding projection directory.
			hdf5s = get_hdf5s_from_dir(spec_dir)
			assert len(latent_means) // len(hdf5s) == self.sylls_per_file
			for j in range(len(hdf5s)):
				filename = os.path.join(proj_dir, os.path.split(hdf5s[j])[-1])
				data = latent_means[j*sylls_per_file:(j+1)*sylls_per_file]
				with h5py.File(filename, 'w') as f:
					f.create_dataset('latent_mean', data=data)
			# Leave a paper trail.
			self.document_parents(proj_dir, i)
		return np.concatenate(all_latent)


	def make_latent_mean_umap_projection(self):
		"""Project latent means to two dimensions with UMAP."""
		# Get latent means.
		latent_means = self.request('latent_means')
		# UMAP them.
		transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, \
			metric='euclidean', random_state=42)
		embedding = transform.transform(latent_means)
		# Write to files.
		self.write_projection("latent_mean_umap", embedding)
		return embedding


	def make_latent_pca_projection(self):
		"""Project latent means to two dimensions with PCA."""
		# Get latent means.
		latent_means = self.request('latent_means')
		# UMAP them.
		transform = pca.PCA(n_components=2, copy=False, random_state=42)
		embedding = transform.transform(latent_means)
		# Write to files.
		self.write_projection("latent_mean_pca", embedding)
		return embedding


	def write_projection(self, key, data):
		"""Write the given projection to self.projection_dirs."""
		sylls_per_file = self.sylls_per_file
		# For each directory...
		k = 0
		for i in range(len(self.projection_dirs)):
			spec_dir, proj_dir = self.spec_dirs[i], self.projection_dirs[i]
			hdf5s = get_hdf5s_from_dir(spec_dir)
			for j in range(len(hdf5s)):
				filename = os.path.join(proj_dir, os.path.split(hdf5s[j])[-1])
				to_write = data[k:k+sylls_per_file]
				with h5py.File(filename, 'w') as f:
					f.create_dataset(key, data=to_write)
				k += sylls_per_file


	def document_parents(self, dir, index):
		"""
		Write a small text file that documents how and when stuff was computed.

		Parameters
		----------
		dir : str

		index : int

		Notes
		-----

		"""
		pass


	def clean_projection_dir(self):
		"""Remove all the latent projections."""
		raise NotImplementedError


	def clean_plots_dir(self):
		"""Remove all the plots."""
		raise NotImplementedError



if __name__ == '__main__':
	pass


###
