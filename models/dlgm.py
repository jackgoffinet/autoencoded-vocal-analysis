from __future__ import print_function, division
"""
A Deep Latent Gaussian Model (DLGM) for image data implemented using Pyro &
PyTorch.

Introduced in:

Rezende, Danilo Jimenez, Shakir Mohamed, and Daan Wierstra. "Stochastic
backpropagation and approximate inference in deep generative models." arXiv
preprint arXiv:1401.4082 (2014).

https://arxiv.org/abs/1401.4082

TO DO: make GPU optional
TO DO: clean up <train>
TO DO: take a look at the 0.02 scale
"""
__author__ = "Jack Goffinet"
__date__ = "November 2018"


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
pyro.enable_validation(True)

import os
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from .rankonenormal import RankOneNormal
from .dataset import get_data_loaders



class Encoder(nn.Module):
	"""
	Encoder network.

	"""
	def __init__(self, params):
		super(Encoder, self).__init__()
		self.latent_dim = params['latent_dim']
		self.input_shape = params['input_shape']
		self.post_conv_dim = params['post_conv_dim']
		# Make a bunch of convolutional & fully connected layers.
		self.conv_layers, self.fc_layers, self.batch_norms = [], [], []
		for in_f, out_f, k_size, stride, pad in params['encoder_conv_layers']:
			self.conv_layers.append(nn.Conv2d(in_f, out_f, k_size, stride, padding=pad))
			self.batch_norms.append(nn.BatchNorm2d(in_f))
		for in_dim, out_dim in params['encoder_fc_layers']:
			self.fc_layers.append(nn.Linear(in_dim, out_dim))
		self.conv_layers = nn.ModuleList(self.conv_layers)
		self.fc_layers = nn.ModuleList(self.fc_layers)
		self.batch_norms = nn.ModuleList(self.batch_norms)
		# Make three last fully connected layers for <mu>, <log_d>, and <u>.
		self.fc1 = nn.Linear(self.latent_dim, self.latent_dim)
		self.fc2 = nn.Linear(self.latent_dim, self.latent_dim)
		self.fc3 = nn.Linear(self.latent_dim, self.latent_dim)


	def forward(self, v):
		# Reshape inputs.
		v = v.view(-1, 1, self.input_shape[0], self.input_shape[1])
		# Run through conv layers.
		for batch_norm, layer in zip(self.batch_norms, self.conv_layers):
			v = F.relu(layer(batch_norm(v)))
		# Reshape.
		v = v.view(-1, self.post_conv_dim)
		# Run through fully-connected layers.
		for layer in self.fc_layers:
			v = F.relu(layer(v))
		log_d = self.fc1(v)
		u = self.fc2(v)
		mu = self.fc3(v)
		return mu, log_d, u


class Decoder(nn.Module):
	"""
	Decoder network.
	"""
	def __init__(self, params):
		super(Decoder, self).__init__()
		self.latent_dim = params['latent_dim']
		self.input_shape = params['input_shape']
		self.input_dim = np.prod(self.input_shape)
		self.post_conv_shape = params['post_conv_shape']
		# Make the first few layers, where samples from the latent space are
		# provided as input to the network.
		self.g_3 = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
		self.g_2 = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
		self.g_1 = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
		self.t_2 = nn.Linear(self.latent_dim, self.latent_dim)
		self.t_1 = nn.Linear(self.latent_dim, self.latent_dim)

		# Make a bunch of fully connected & transposed convolutional layers.
		self.fc_layers, self.convt_layers, self.batch_norms = [], [], []
		for in_dim, out_dim in params['decoder_fc_layers']:
			self.fc_layers.append(nn.Linear(in_dim, out_dim))
		bit = (len(params['decoder_convt_layers']) + 1) % 2
		for i, (in_f, out_f, k_size, stride, pad) in enumerate(params['decoder_convt_layers']):
			self.convt_layers.append(nn.ConvTranspose2d(in_f, out_f, k_size,
					stride, padding=pad, output_padding=(i+bit)%2))
			if i != len(params['decoder_convt_layers']) - 1:
				self.batch_norms.append(nn.BatchNorm2d(out_f))
		self.fc_layers = nn.ModuleList(self.fc_layers)
		self.convt_layers = nn.ModuleList(self.convt_layers)
		self.batch_norms = nn.ModuleList(self.batch_norms)


	def forward(self, xis):
		"""Generate a spectrogram given samples from latent space."""
		# Sequentially introduce samples from latent space.
		xi_1, xi_2, xi_3 = xis
		h_3 = self.g_3(xi_3)
		h_2 = F.relu(self.t_2(h_3)) + self.g_2(xi_2)
		h_1 = F.relu(self.t_1(h_2)) + self.g_1(xi_1)
		v = h_1
		# Run through the fully connected layers.
		for layer in self.fc_layers:
			v = F.relu(layer(v))
		# Reshape.
		v = v.view(-1, self.post_conv_shape[0], self.post_conv_shape[1],
				self.post_conv_shape[2])
		# Run through the conv transpose layers.
		for batch_norm, layer in zip(self.batch_norms, self.convt_layers[:-1]):
			v = F.relu(batch_norm(layer(v)))
		# Apply a sigmoid on the last layer.
		v = torch.sigmoid(self.convt_layers[-1](v))
		# Reshape.
		v = v.view(-1, self.input_dim)
		return v



class DLGM(nn.Module):
	"""Deep Latent Gaussian Model"""

	def __init__(self, network_dims, partition=None, test_freq=5, save_dir=None, load_dir=None, sylls_per_file=1000):
		"""
		Construct a DLGM.

		Arguments
		---------
			- network_dims
		"""
		super(DLGM, self).__init__()
		assert(save_dir is not None or load_dir is not None)
		self.params = network_dims
		self.partition = partition
		self.input_shape = self.params['input_shape']
		self.input_dim = self.params['input_dim']
		self.latent_dim = self.params['latent_dim']
		self.test_freq = test_freq
		self.save_dir = save_dir
		self.sylls_per_file = sylls_per_file
		if self.save_dir is not None:
			if self.save_dir[-1] != '/':
				self.save_dir += '/'
			if not os.path.exists(self.save_dir):
			    os.makedirs(self.save_dir)
		else:
			self.save_dir = load_dir
		self.load_dir = load_dir
		if self.load_dir is not None:
			if self.load_dir[-1] != '/':
				self.load_dir += '/'
			filename = self.load_dir + 'checkpoint.tar'
			checkpoint = torch.load(filename)
			self.partition = checkpoint['partition']
		self.encoder = Encoder(self.params)
		self.decoder = Decoder(self.params)
		self.train_loader, self.test_loader = get_data_loaders(self.partition, \
				time_shift=(False,False), sylls_per_file=self.sylls_per_file)
		self.cuda()


	# define the model p(x|z)p(z)
	def model(self, x):
		# register PyTorch module <decoder> with Pyro
		pyro.module("decoder", self.decoder)
		with pyro.iarange("data", x.size(0)):
			# setup hyperparameters for prior p(z)
			mu = x.new_zeros((x.shape[0], self.latent_dim))
			log_d = x.new_zeros((x.shape[0], self.latent_dim))
			u = x.new_zeros((x.shape[0], self.latent_dim))
			try:
				db = RankOneNormal(mu, log_d, u)
				# db = dist.Normal(mu, torch.exp(log_d)).independent(1)
			except:
				print("caught in model")
				print(torch.sum(torch.isnan(mu)))
				print(torch.sum(torch.isnan(log_d)))
				quit()
			xi_1 = pyro.sample("latent_1", db)
			xi_2 = pyro.sample("latent_2", db)
			xi_3 = pyro.sample("latent_3", db)
			loc_img = self.decoder.forward((xi_1, xi_2, xi_3))
			# score against actual images
			pyro.sample("obs", dist.Normal(loc_img, scale=0.02).independent(1), obs=x.view(-1, self.input_dim))
			return loc_img


	# define the guide (variational distribution)
	def guide(self, x):
		# register PyTorch module `encoder` with Pyro
		pyro.module("encoder", self.encoder)
		with pyro.iarange("data", x.size(0)):
			mu, log_d, u = self.encoder.forward(x)

			try:
				db = RankOneNormal(mu, log_d, u)
			except:
				print("caught in guide")
				for group in [self.encoder.fc_layers]:
					for layer in group:
						print(torch.min(layer.weight), torch.max(layer.weight))
						print(torch.min(layer.bias), torch.max(layer.bias))
				print(torch.min(x), torch.max(x))
				print(torch.sum(torch.isnan(mu)))
				print(torch.sum(torch.isnan(log_d)))
				quit()
			pyro.sample("latent_1", db)
			pyro.sample("latent_2", db)
			pyro.sample("latent_3", db)


	def train(self, epochs, lr=3.0e-5, tf=2):
		"""Train the DLGM for some number of epochs."""
		# Set up the optimizer.
		optimizer = Adam({"lr": lr})
		train_elbo = {}
		test_elbo = {}
		start_epoch = 0
		# Load cached state, if given.
		if self.load_dir is not None:
			filename = self.load_dir + 'checkpoint.tar'
			checkpoint = torch.load(filename)
			self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
			self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
			optimizer.set_state(checkpoint['optimizer_state'])
			train_elbo = checkpoint['train_elbo']
			test_elbo = checkpoint['test_elbo']
			start_epoch = checkpoint['epoch'] + 1
			self.partition = checkpoint['partition']
			self.train_loader, self.test_loader = get_data_loaders(self.partition, \
					time_shift=(False,False), sylls_per_file=self.sylls_per_file)

		# Set up the inference algorithm.
		elbo = Trace_ELBO()
		svi = SVI(self.model, self.guide, optimizer, loss=elbo)

		print("dataset length: ", len(self.train_loader.dataset))
		for epoch in range(start_epoch, start_epoch+epochs+1, 1):
			train_loss = 0.0
			# Iterate over the training data.
			for i, temp in enumerate(self.train_loader):
				x = temp['image'].cuda().view(-1, self.input_dim)
				train_loss += svi.step(x)
			# Report training diagnostics.
			normalizer_train = len(self.train_loader.dataset)
			total_epoch_loss_train = train_loss / normalizer_train
			train_elbo[epoch] = total_epoch_loss_train
			print("[epoch %03d]  average train loss: %.4f" %
					(epoch, total_epoch_loss_train))

			if (epoch + 1) % tf == 0:
				test_loss = 0.0
				# Iterate over the test set.
				for i, temp in enumerate(self.test_loader):
					x = temp['image'].cuda().view(-1, self.input_dim)
					test_loss += svi.evaluate_loss(x)
				# Report test diagnostics.
				normalizer_test = len(self.test_loader.dataset)
				total_epoch_loss_test = test_loss / normalizer_test
				test_elbo[epoch] = total_epoch_loss_test
				print("[epoch %03d]  average test loss: %.4f" %
						(epoch, total_epoch_loss_test))
				self.visualize()

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


	def reconstruct_img(self, v, from_numpy=False):
		# Encode image <v>.
		if from_numpy:
			v = torch.from_numpy(v).type(torch.FloatTensor)
		mu, log_d, u = self.encoder.forward(v)
		db = RankOneNormal(mu, log_d, u)
		# db = dist.Normal(mu, torch.exp(log_d)).independent(1)
		xi_1 = pyro.sample("latent_1", db)
		xi_2 = pyro.sample("latent_2", db)
		xi_3 = pyro.sample("latent_3", db)
		loc_img = self.decoder.forward((xi_1, xi_2, xi_3))
		if from_numpy:
			return loc_img.detach().cpu().numpy()
		return loc_img


	def generate_from_latent(self, latent):
		latent = torch.from_numpy(latent).type(torch.FloatTensor)
		loc_img = self.decoder.forward((latent, latent, latent))
		return loc_img.detach().cpu().numpy()


	def get_latent(self, loader, n=3000):
		n = min(n, len(loader.dataset))
		print("len(loader.dataset)", len(loader.dataset))
		filename = self.load_dir + 'checkpoint.tar'
		checkpoint = torch.load(filename)
		self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
		latent = np.zeros((n, self.latent_dim))
		i = 0
		for temp in loader:
			x = temp['image'].cuda().view(-1, self.input_dim)
			mu, _, _ = self.encoder.forward(x)
			mu = mu.detach().cpu().numpy()
			index = min(n-i,len(mu))
			latent[i:i+index] = mu[:index]
			i += index
		return latent


	def visualize(self, loader=None, filename='temp.pdf'):
		"""Plot spectrograms and their reconstructions."""
		if loader is None:
			loader = self.test_loader
		gap = 10
		big_img = np.zeros((2*self.input_shape[0]+gap, 5*self.input_shape[1]+4*gap))
		for temp in loader:
			x = temp['image'].cuda().view(-1, self.input_dim)
			reconstructed = self.reconstruct_img(x)
			x = x.view((-1,) + self.input_shape)
			for im_num in range(5):
				temp_im = x[im_num].detach().cpu().numpy().reshape(self.input_shape)
				i1 = im_num*(self.input_shape[0] + gap)
				i2 = i1 + temp_im.shape[0]
				big_img[temp_im.shape[1]+gap:,i1:i2] = temp_im
				temp_im = reconstructed[im_num].detach().cpu().numpy().reshape(self.input_shape)
				big_img[:temp_im.shape[1],i1:i2] = temp_im
			break
		plt.imshow(big_img, aspect='equal', origin='lower', interpolation='none')
		plt.axis('off')
		plt.savefig(filename)
		plt.close('all')



if __name__ == '__main__':
	pass
