"""
Vocal-Motor DLGM:

Unsupervised learning of motor control manifolds underlying vocal communication.

TO DO: add optimizer, partition, etc. to MotorDLGM
"""
__author__ = "Jack Goffinet"
__date__ = "December 2018"


import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import umap

from rankonenormal import RankOneNormal
from motor_dataset import get_partition, get_data_loaders


accel_filter = nn.Conv2d(1, 1, kernel_size=3, bias=False)
kernel = np.tile([1.0, -2.0, 1.0], (8,1))
kernel = torch.from_numpy(kernel).type(torch.FloatTensor).cuda().view(1,1,3,8)
accel_filter.weight.data = kernel
accel_filter.weight.requires_grad = False


class Encoder(nn.Module):
	"""Encoder network, a standard feedforward network."""

	def __init__(self, latent_dim, use_cuda=True, batch_size=64):
		super(Encoder, self).__init__()
		self.latent_dim = latent_dim
		self.use_cuda = use_cuda
		self.batch_size = batch_size
		# Make a bunch of convolutional & fully connected layers.
		layer_dims = [
		[128,	64],
		[64,	32],
		[32,	16],
		[16,	16],
		]
		self.fc_layers = []
		for in_dim, out_dim in layer_dims:
			self.fc_layers.append(nn.Linear(in_dim, out_dim))
		self.fc_layers = nn.ModuleList(self.fc_layers)
		# Make three last fully connected layers for <mu>, <log_d>, and <u>.
		self.fc1 = nn.Linear(16, self.latent_dim)
		self.fc2 = nn.Linear(16, self.latent_dim)
		self.fc3 = nn.Linear(16, self.latent_dim)
		if self.use_cuda:
			self.cuda()


	def forward(self, v):
		# Run through fully-connected layers.
		for layer in self.fc_layers:
			v = F.relu(layer(v))
		mu = self.fc1(v)
		log_d = self.fc2(v)
		u = self.fc3(v)
		return mu, log_d, u


class Decoder(nn.Module):
	"""Decoder network, throw around the latent samples a bit, then decode."""

	def __init__(self, latent_dim, use_cuda=True, batch_size=64):
		super(Decoder, self).__init__()
		self.latent_dim = latent_dim
		self.use_cuda = use_cuda
		self.batch_size = batch_size
		self.g_3 = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
		self.g_2 = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
		self.g_1 = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
		self.t_2 = nn.Linear(self.latent_dim, self.latent_dim)
		self.t_1 = nn.Linear(self.latent_dim, self.latent_dim)
		layer_dims = [
		[8,		16],
		[16,	32],
		[32,	64],
		[64,	128],
		]
		self.fc_layers = []
		for in_dim, out_dim in layer_dims:
			self.fc_layers.append(nn.Linear(in_dim, out_dim))
		self.fc_layers = nn.ModuleList(self.fc_layers)
		if self.use_cuda:
			self.cuda()


	def forward(self, xis):
		"""Generate given samples from latent space."""
		# Sequentially introduce samples from latent space.
		xi_1, xi_2, xi_3 = xis
		h_3 = self.g_3(xi_3)
		h_2 = F.relu(self.t_2(h_3)) + self.g_2(xi_2)
		h_1 = F.relu(self.t_1(h_2)) + self.g_1(xi_1)
		v = h_1
		# Run through the fully connected layers.
		for layer in self.fc_layers[:-1]:
			v = F.relu(layer(v))
		# Apply a sigmoid on the last layer.
		v = torch.sigmoid(self.fc_layers[-1](v))
		return v



class MotorDLGM(nn.Module):
	"""Vocal-Motor DLGM"""

	def __init__(self, latent_dim, use_cuda=True, aux_loss_multiplier=None, batch_size=64):
		 super(MotorDLGM, self).__init__()
		 self.use_cuda = use_cuda
		 self.aux_loss_multiplier = aux_loss_multiplier
		 self.batch_size = batch_size
		 self.latent_dim = latent_dim
		 self.encoder = Encoder(self.latent_dim, batch_size=self.batch_size)
		 self.decoder = Decoder(self.latent_dim, batch_size=self.batch_size)
		 if self.use_cuda:
			 self.cuda()


	def model(self, x):
		pyro.module("decoder", self.decoder)
		with pyro.iarange("data", x.size(0)):
			# setup hyperparameters for prior p(z)
			mu = x.new_zeros((x.shape[0], self.latent_dim))
			log_d = x.new_zeros((x.shape[0], self.latent_dim))
			u = x.new_zeros((x.shape[0], self.latent_dim))
			db = RankOneNormal(mu, log_d, u)
			# db = dist.Normal(mu, torch.exp(log_d)).independent(1)
			xi_1 = pyro.sample("latent_1", db)
			xi_2 = pyro.sample("latent_2", db)
			xi_3 = pyro.sample("latent_3", db)
			loc_img = self.decoder.forward((xi_1, xi_2, xi_3))
			# score against actual images
			pyro.sample("obs", dist.Normal(loc_img, scale=0.02).independent(1), obs=x.view(-1, 128))
			return loc_img


	def guide(self, x):
		pyro.module("encoder", self.encoder)
		with pyro.iarange("data", x.size(0)):
			mu, log_d, u = self.encoder.forward(x)
			db = RankOneNormal(mu, log_d, u)
			pyro.sample("latent_1", db)
			pyro.sample("latent_2", db)
			pyro.sample("latent_3", db)


	def aux_model(self, x):
		pyro.module("motor_dlgm", self)
		loc = x.new_zeros((x.shape[0], 126))
		with pyro.iarange("data", x.size(0)):
			mu, _, _ = self.encoder.forward(x)
			accels = accel_filter(mu).squeeze(3).squeeze(1)
			pyro.sample("y_aux", dist.Normal(loc=loc, scale=0.02).independent(1), obs=accels)


	def aux_guide(self, x):
		pass


	def get_latent(self, loader, n=9000):
		n = min(n, len(loader.dataset)*128)
		latent = np.zeros((n, self.latent_dim))
		i = 0
		for temp in loader:
			x = temp['image'].cuda().view(-1, 128)
			mu, _, _ = self.encoder.forward(x)
			mu = mu.detach().cpu().numpy()
			index = min(n-i,len(mu))
			latent[i:i+index] = mu[:index]
			i += index
		return latent


	def visualize(self, loader):
		latent = self.get_latent(loader)
		print("latent shape", latent.shape)
		transform = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric='euclidean')
		embeddings = transform.fit_transform(latent)
		plt.scatter(embeddings[:,0], embeddings[:,1], alpha=0.1, s=0.5)
		plt.savefig('temp.pdf')


	def run_inference_for_epoch(loader, objectives, shapes):
		num_objectives = len(objectives)
		epoch_losses = np.zeros(num_objectives)
		# For each batch...
		for i, temp in enumerate(loader):
			x = temp['image'].cuda()
			# For each loss...
			for j in range(num_objectives):
				# Perform a single optimization step.
				x_view = x.view(shapes[j])
				temp_loss = objectives[j].step(x_view)
				epoch_losses[j] += temp_loss
		return epoch_losses / len(loader.dataset)


	def train(self, loaders, num_epochs):
		optimizer = Adam({"lr": 3e-5})
		elbo = Trace_ELBO()
		main_obj = SVI(self.model, self.guide, optimizer, loss=elbo)
		aux_obj = SVI(self.aux_model, self.aux_guide, optimizer, loss=elbo)
		objectives = [main_obj, aux_obj]
		shapes = ((-1,128), (-1,1,128,128))

		losses = np.zeros((num_epochs,2))
		for epoch in range(num_epochs):
			losses[epoch,:] = dlgm.run_inference_for_epoch(loaders['train'], objectives, shapes)
			print(str(epoch).zfill(3), losses[epoch])

			if (epoch + 1) % tf == 0:
				self.visualize(loaders['test'], filename='temp.pdf')
				self.save_state(epoch)


	def save_state(self, epoch):
		if self.save_dir is not None:
			filename =  self.save_dir + 'checkpoint.tar'
			state = {
				'train_elbo': train_elbo,
				'test_elbo': test_elbo,
				'epoch': epoch,
				'encoder_state_dict': self.encoder.state_dict(),
				'decoder_state_dict': self.decoder.state_dict(),
				'optimizer_state': optimizer.get_state(),
				'partition': self.partition,
			}
			torch.save(state, filename)


if __name__ == '__main__':
	num_epochs = 100
	batch_size = 64
	tf = 2 # Test frequency

	# Set up the model.
	dlgm = MotorDLGM(8) # Deep Latent Gaussian Model

	# Set up the data loader.
	load_dirs = ['../data/processed/hage/S'+str(i)+'/' for i in [1,2,3,4,5]]
	partition = get_partition(load_dirs, split=0.8)
	loaders = get_data_loaders(partition, batch_size=batch_size, sylls_per_file=200)

	dlgm.train(loaders, num_epochs=num_epochs, tf=tf)




###
