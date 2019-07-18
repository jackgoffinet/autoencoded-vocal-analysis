"""
A Variational Autoencoder (VAE) for spectrogram data implemented using PyTorch.

References
----------
[1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."
	arXiv preprint arXiv:1312.6114 (2013).

	https://arxiv.org/abs/1312.6114

[2] Rezende, Danilo Jimenez, Shakir Mohamed, and Daan Wierstra. "Stochastic
	backpropagation and approximate inference in deep generative models." arXiv
	preprint arXiv:1401.4082 (2014).

	https://arxiv.org/abs/1401.4082
"""
__author__ = "Jack Goffinet"
__date__ = "November 2018 - July 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
import torch
from torch.distributions import LowRankMultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .vae_dataset import SyllableDataset


X_SHAPE = (128,128)
X_DIM = np.prod(X_SHAPE)



class VAE(nn.Module):
	"""
	Variational Auto-Encoder class for single-channel images.

	Notes
	-----
	The prior p(z) is a unit normal distribution. The conditional distribution
	p(x|z) is set as a spherical normal distribution to prevent overfitting.
	The variational distribution, q(z|x) is a rank-1 multivariate normal
	distribution.

	The model is trained using the standard ELBO objective:

	ELBO = E_{q(z|x)} log p(x,z) + H[q(z|x)]

	where p(x,z) = p(z)p(x|z) and H is differential entropy. Here, q(z|x) and
	p(x|z) are parameterized by neural networks. Gradients are passed through
	stochastic layers via the reparameterization trick, implemented by the
	PyTorch rsample method.

	The dimensions of the network are hard-coded for use with 128x128
	spectrograms. While a desired latent dimension can be passed to __init__,
	the dimensions of the network limit the practical range of values roughly 8
	to 64 dimensions. Fiddling with the image dimensions will require updating
	the parameters of the layers defined in VAE.build_network.

	TO DO: numerical issues and learning rates
	"""

	def __init__(self, save_dir='', lr=1e-3, z_dim=32, model_precision=10.0,
		device_name="auto"):
		"""
		Construct the VAE.

		Parameters
		----------
		save_dir : str, optional
			Directory where the model is saved. Defaults to the current working
			directory.

		lr : float, optional
			Learning rate of the ADAM optimizer. Defaults to 1e-3.

		z_dim : int, optional
			Dimension of the latent space. Defaults to 32.

		model_precision : float, optional
			Precision of the noise model, p(x|z) = N(mu(z), \Lambda) where
			\Lambda = model_precision * I. Defaults to 10.0.

		device_name: str, optional
			Device to train the model on. Valid options are ["cpu", "cuda",
			"auto"]. "auto" will choose "cuda" if it is available. Defaults to
			"auto".

		Notes
		-----
		The model is built before it's parameters can be loaded from a file.
		This means self.z_dim must match z_dim of the model being loaded.
		"""
		super(VAE, self).__init__()
		self.save_dir = save_dir
		self.lr = lr
		self.z_dim = z_dim
		self.model_precision = model_precision
		assert device_name != "cuda" or torch.cuda.is_available()
		if device_name == "auto":
			device_name = "cuda" if torch.cuda.is_available() else "cpu"
		self.device = torch.device(device_name)
		if self.save_dir != '' and not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
		self.build_network()
		self.optimizer = Adam(self.parameters(), lr=self.lr)
		self.epoch = 0
		self.loss = {'train':{}, 'test':{}}
		self.to(self.device)


	def build_network(self):
		"""Define all the network layers."""
		# Encoder
		self.conv1 = nn.Conv2d(1, 8, 3,1,padding=1)
		self.conv2 = nn.Conv2d(8, 8, 3,2,padding=1)
		self.conv3 = nn.Conv2d(8, 16,3,1,padding=1)
		self.conv4 = nn.Conv2d(16,16,3,2,padding=1)
		self.conv5 = nn.Conv2d(16,24,3,1,padding=1)
		self.conv6 = nn.Conv2d(24,24,3,2,padding=1)
		self.conv7 = nn.Conv2d(24,32,3,1,padding=1)
		self.bn1 = nn.BatchNorm2d(1)
		self.bn2 = nn.BatchNorm2d(8)
		self.bn3 = nn.BatchNorm2d(8)
		self.bn4 = nn.BatchNorm2d(16)
		self.bn5 = nn.BatchNorm2d(16)
		self.bn6 = nn.BatchNorm2d(24)
		self.bn7 = nn.BatchNorm2d(24)
		self.fc1 = nn.Linear(8192,1024)
		self.fc2 = nn.Linear(1024,256)
		self.fc31 = nn.Linear(256,64)
		self.fc32 = nn.Linear(256,64)
		self.fc33 = nn.Linear(256,64)
		self.fc41 = nn.Linear(64,self.z_dim)
		self.fc42 = nn.Linear(64,self.z_dim)
		self.fc43 = nn.Linear(64,self.z_dim)
		# Decoder
		self.fc5 = nn.Linear(self.z_dim,64)
		self.fc6 = nn.Linear(64,256)
		self.fc7 = nn.Linear(256,1024)
		self.fc8 = nn.Linear(1024,8192)
		self.convt1 = nn.ConvTranspose2d(32,24,3,1,padding=1)
		self.convt2 = nn.ConvTranspose2d(24,24,3,2,padding=1,output_padding=1)
		self.convt3 = nn.ConvTranspose2d(24,16,3,1,padding=1)
		self.convt4 = nn.ConvTranspose2d(16,16,3,2,padding=1,output_padding=1)
		self.convt5 = nn.ConvTranspose2d(16,8,3,1,padding=1)
		self.convt6 = nn.ConvTranspose2d(8,8,3,2,padding=1,output_padding=1)
		self.convt7 = nn.ConvTranspose2d(8,1,3,1,padding=1)
		self.bn8 = nn.BatchNorm2d(32)
		self.bn9 = nn.BatchNorm2d(24)
		self.bn10 = nn.BatchNorm2d(24)
		self.bn11 = nn.BatchNorm2d(16)
		self.bn12 = nn.BatchNorm2d(16)
		self.bn13 = nn.BatchNorm2d(8)
		self.bn14 = nn.BatchNorm2d(8)


	def get_layers(self):
		"""Return a dictionary mapping names to network layers."""
		return {'fc1':self.fc1, 'fc2':self.fc2, 'fc31':self.fc31,
				'fc32':self.fc32, 'fc33':self.fc33, 'fc41':self.fc41,
				'fc42':self.fc42, 'fc43':self.fc43, 'fc5':self.fc5,
				'fc6':self.fc6, 'fc7':self.fc7, 'fc8':self.fc8, 'bn1':self.bn1,
				'bn2':self.bn2, 'bn3':self.bn3, 'bn4':self.bn4, 'bn5':self.bn5,
				'bn6':self.bn6, 'bn7':self.bn7, 'bn8':self.bn8, 'bn9':self.bn9,
				'bn10':self.bn10, 'bn11':self.bn11, 'bn12':self.bn12,
				'bn13':self.bn13, 'bn14':self.bn14, 'conv1':self.conv1,
				'conv2':self.conv2, 'conv3':self.conv3, 'conv4':self.conv4,
				'conv5':self.conv5, 'conv6':self.conv6, 'conv7':self.conv7,
				'convt1':self.convt1, 'convt2':self.convt2,
				'convt3':self.convt3, 'convt4':self.convt4,
				'convt5':self.convt5, 'convt6':self.convt6,
				'convt7':self.convt7}


	def encode(self, x):
		"""
		Compute q(z|x).

		q(z|x) = N(mu, \Sigma), \Sigma = u @ u.T + d
		where mu, u, and d are deterministic functions of x, @ denotes matrix
		multiplication, and \Sigma denotes a covariance matrix.

		Parameters
		----------
		x : torch.Tensor
			The input images, with shape [batch_size, height=128, width=128]

		Returns
		-------
		mu : torch.Tensor
			Posterior mean, with shape [batch_size, self.z_dim]

		u : torch.Tensor
			Posterior covariance factor, as defined above.
			Shape: [batch_size, self.z_dim]

		d : torch.Tensor
			Posterior diagonal factor, as defined above.
			Shape: [batch_size, self.z_dim]
		"""
		x = x.unsqueeze(1)
		x = F.relu(self.conv1(self.bn1(x)))
		x = F.relu(self.conv2(self.bn2(x)))
		x = F.relu(self.conv3(self.bn3(x)))
		x = F.relu(self.conv4(self.bn4(x)))
		x = F.relu(self.conv5(self.bn5(x)))
		x = F.relu(self.conv6(self.bn6(x)))
		x = F.relu(self.conv7(self.bn7(x)))
		x = x.view(-1, 8192)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		mu = F.relu(self.fc31(x))
		mu = self.fc41(mu)
		u = F.relu(self.fc32(x))
		u = self.fc42(u).unsqueeze(-1) # Last dimension is rank of \Sigma = 1.
		d = F.relu(self.fc33(x))
		d = torch.exp(self.fc43(d)) # d must be positive.
		return mu, u, d


	def decode(self, z):
		"""
		Compute p(x|z).

		p(x|z) = N(mu, \Lambda), \Lambda = <model_precision> * I
		where mu is a deterministic function of z, \Lambda is a precision
		matrix, and I is an identity matrix.

		Parameters
		----------
		z : torch.Tensor
			Batch of latent samples with shape [batch_size, self.z_dim]

		Returns
		-------
		x : torch.Tensor
			Batch of means mu, described above. Shape: [batch_size,
			X_DIM=128*128]
		"""
		z = F.relu(self.fc5(z))
		z = F.relu(self.fc6(z))
		z = F.relu(self.fc7(z))
		z = F.relu(self.fc8(z))
		z = z.view(-1,32,16,16)
		z = F.relu(self.convt1(self.bn8(z)))
		z = F.relu(self.convt2(self.bn9(z)))
		z = F.relu(self.convt3(self.bn10(z)))
		z = F.relu(self.convt4(self.bn11(z)))
		z = F.relu(self.convt5(self.bn12(z)))
		z = F.relu(self.convt6(self.bn13(z)))
		z = self.convt7(self.bn14(z))
		return z.view(-1, X_DIM)


	def forward(self, x, return_latent_rec=False):
		"""
		Send x round trip and compute a loss.

		In more detail: Given x, compute q(z|x) and sample: z ~ q(z|x). Then
		compute p(x|z) to get the log-likelihood of x, the input, given z, the
		sampled latent variable. We will also need the likelihood of z under the
		model's prior, p(z), and the entropy of the latent conditional
		distribution, H[q(z|x)]. ELBO can then be estimated:

		ELBO = E_{q(z|x)}[log p(z) + log p(x|z)] + H[q(z|x)]

		Parameters
		----------
		x : torch.Tensor
			A batch of samples from the data distribution (spectrograms).
			Shape: [batch_size, height=128, width=128]

		return_latent_rec : bool, optional
			Whether to return latent means and reconstructions. Defaults to
			False.

		Returns
		-------
		loss : torch.Tensor
			Negative ELBO times the batch size. Shape: []

		latent : numpy.ndarray, if return_latent_rec
			Latent means. Shape: [batch_size, self.z_dim]

		reconstructions : numpy.ndarray, if return_latent_rec
			Reconstructed means. Shape: [batch_size, height=128, width=128]
		"""
		mu, u, d = self.encode(x)
		latent_dist = LowRankMultivariateNormal(mu, u, d)
		z = latent_dist.rsample()
		x_rec = self.decode(z)
		elbo = -0.5 * (torch.sum(torch.pow(z,2)) + self.z_dim * \
			np.log(2*np.pi)) # B * p(z)
		elbo = elbo + -0.5 * (self.model_precision * \
			torch.sum(torch.pow(x.view(-1,X_DIM) - x_rec, 2)) + self.z_dim * \
			np.log(2*np.pi)) # ~ B * E_{q} p(x|z)
		elbo = elbo + torch.sum(latent_dist.entropy()) # ~ B * H[q(z|x)]
		if return_latent_rec:
			return -elbo, z.detach().cpu().numpy(), \
				x_rec.view(-1, X_SHAPE[0], X_SHAPE[1]).detach().cpu().numpy()
		return -elbo


	def train_epoch(self, train_loader):
		"""
		Train the model for a single epoch.

		Parameters
		----------
		train_loader : torch.utils.data.DataLoader
			Dataloader for training set spectrograms

		Returns
		-------
		elbo : float
			A biased estimate of the Evidence Lower BOund, estimated using
			samples from train_loader.
		"""
		self.train()
		train_loss = 0.0
		# print(len(train_loader.dataset))
		# print(len(train_loader.dataset.filenames))
		# print(train_loader.dataset.sylls_per_file)
		# quit()
		for batch_idx, data in enumerate(train_loader):
			data = data.to(self.device)
			self.optimizer.zero_grad()
			loss = self.forward(data)
			loss.backward()
			train_loss += loss.item()
			self.optimizer.step()
		train_loss /= len(train_loader.dataset)
		print('Epoch: {} Average loss: {:.4f}'.format(self.epoch, \
				train_loss))
		self.epoch += 1
		return train_loss


	def test_epoch(self, test_loader):
		"""
		Test the model on a held-out validation set, return an ELBO estimate.

		Paramters
		---------
		test_loader : torch.utils.data.Dataloader
			Dataloader for test set spectrograms

		Returns
		-------
		elbo : float
			An unbiased estimate of the Evidence Lower BOund, estimated using
			samples from test_loader.
		"""
		self.eval()
		test_loss = 0.0
		with torch.no_grad():
			for i, data in enumerate(test_loader):
				data = data.to(self.device)
				loss = self.forward(data)
				test_loss += loss.item()
		test_loss /= len(test_loader.dataset)
		print('Test loss: {:.4f}'.format(test_loss))
		return test_loss


	def train_loop(self, loaders, epochs=100, test_freq=2, save_freq=10,
		vis_freq=1):
		"""
		Train the model for multiple epochs, testing and saving along the way.

		Parameters
		----------
		loaders : dictionary
			Dictionary mapping the keys 'test' and 'train' to respective
			torch.utils.data.Dataloader objects.

		epochs : int, optional
			Number of (possibly additional) epochs to train the model for.
			Defaults to 100.

		test_freq : int, optional
			Testing is performed every <test_freq> epochs. Defaults to 2.

		save_freq : int, optional
			The model is saved every <save_freq> epochs. Defaults to 10.

		vis_freq : int, optional
			Syllable reconstructions are plotted every <vist_freq> epochs.
			Defaults to 1.
		"""
		print("="*40)
		print("Training: epochs", self.epoch, "to", self.epoch+epochs-1)
		print("Test set:", len(loaders['test'].dataset))
		print("Training set:", len(loaders['train'].dataset))
		print("="*40)
		# For some number of epochs...
		for epoch in range(self.epoch, self.epoch+epochs):
			# Run through the training data and record a loss.
			loss = self.train_epoch(loaders['train'])
			self.loss['train'][epoch] = loss
			# Run through the test data and record a loss.
			if epoch % test_freq == 0:
				loss = self.test_epoch(loaders['test'])
				self.loss['test'][epoch] = loss
			# Save the model.
			if epoch % save_freq == 0:
				filename = "checkpoint_"+str(epoch).zfill(3)+'.tar'
				self.save_state(filename)
			# Visualize reconstructions.
			if epoch % vis_freq == 0:
				self.visualize(loaders['test'])


	def save_state(self, filename):
		"""Save all the model parameters to the given file."""
		layers = self.get_layers()
		state = {}
		for layer_name in layers:
			state[layer_name] = layers[layer_name].state_dict()
		state['optimizer_state'] = self.optimizer.state_dict()
		state['loss'] = self.loss
		state['z_dim'] = self.z_dim
		state['epoch'] = self.epoch
		state['lr'] = self.lr
		state['save_dir'] = self.save_dir
		filename = os.path.join(self.save_dir, filename)
		torch.save(state, filename)


	def load_state(self, filename):
		"""
		Load all the model parameters from the given file.

		Note that self.lr, self.save_dir, and self.z_dim are not loaded.
		"""
		checkpoint = torch.load(filename)
		assert checkpoint['z_dim'] == self.z_dim
		layers = self.get_layers()
		for layer_name in layers:
			layer = layers[layer_name]
			layer.load_state_dict(checkpoint[layer_name])
		self.optimizer.load_state_dict(checkpoint['optimizer_state'])
		self.loss = checkpoint['loss']
		self.epoch = checkpoint['epoch']


	def visualize(self, loader, num_specs=5, gap=4, save_filename='temp.pdf'):
		"""
		Plot spectrograms and their reconstructions.

		Spectrograms are chosen at random from the Dataloader Dataset.

		Parameters
		----------
		loader : torch.utils.data.Dataloader
			Spectrogram Dataloader

		num_specs : int, optional
			Number of spectrogram pairs to plot. Defaults to 5.

		gap : int, optional
			Number of empty pixels between images. Defaults to 4.

		save_filename : str, optional
			Where to save the plot, relative to self.save_dir. Defaults to
			'temp.pdf'.
		"""
		# Collect random indices.
		assert num_specs <= len(loader.dataset) and num_specs >= 1
		indices = np.random.choice(np.arange(len(loader.dataset)),
			size=num_specs,replace=False)
		# Retrieve spectrograms from the loader.
		specs = torch.stack(loader.dataset[indices]).to(self.device)
		# Get resonstructions.
		with torch.no_grad():
			_, _, rec_specs = self.forward(specs, return_latent_rec=True)
		specs = specs.detach().cpu().numpy()
		# Plot.
		height = 2*X_SHAPE[0]+gap
		width = num_specs*X_SHAPE[1] + (num_specs-1)*gap
		img = np.zeros((height, width))
		for i in range(num_specs):
			x = i * (X_SHAPE[1] + gap)
			img[:X_SHAPE[0],x:x+X_SHAPE[1]] = rec_specs[i]
			img[-X_SHAPE[0]:,x:x+X_SHAPE[1]] = specs[i]
		plt.imshow(img, aspect='equal', origin='lower', interpolation='none',
			vmin=0, vmax=1)
		plt.axis('off')
		plt.tight_layout()
		plt.savefig(os.path.join(self.save_dir, save_filename))
		plt.close('all')


	def get_latent(self, loader):
		"""
		Get latent means for all syllable in the given loader.

		Parameters
		----------
		loader : torch.utils.data.Dataloader
			SyllableDataset Dataloader.

		Returns
		-------
		latent : numpy.ndarray
			Latent means. Shape: [len(loader.dataset), self.z_dim]

		Note
		----
		- Make sure your loader is not set to shuffle if you're going to match
		  these with labels or other fields later.
		"""
		latent = np.zeros((len(loader.dataset), self.z_dim))
		i = 0
		for data in loader:
			data = data.to(self.device)
			with torch.no_grad():
				mu, _, _ = self.encode(data)
			mu = mu.detach().cpu().numpy()
			latent[i:i+len(mu)] = mu
			i += len(mu)
		return latent



if __name__ == '__main__':
	pass


###
