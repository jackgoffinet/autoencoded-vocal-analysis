"""normal distribution with a rank-1 covariance structure."""
import numpy as np
import torch
from torch.autograd import Variable
import math
from functools import reduce


from pyro.distributions.distribution import Distribution



class RankOneNormal(Distribution):
	"""
	Sparsely parameterized multivariate normal distribution
	"""
	reparameterized = True

	def __init__(self, mu, log_d, u, batch_size=None, log_pdf_mask=None, *args, **kwargs):
		assert len(mu.shape) == 2
		assert mu.shape == log_d.shape and log_d.shape == u.shape
		self.mu = mu
		self.construct_matrix(log_d, u)
		self.r_inv = None
		self.log_pdf_mask = log_pdf_mask
		super(RankOneNormal, self).__init__(*args, **kwargs)


	def construct_matrix(self, log_d, u):
		# define R, caching stuff in case R^-1 needs to be calculated.
		d = torch.exp(log_d)
		d_inv = torch.reciprocal(d)
		d_rsqrt = torch.rsqrt(d)

		d_rsqrt_diag = []
		for vec in d_rsqrt:
			d_rsqrt_diag.append(torch.diag(vec))
		self.d_rsqrt_diag = torch.stack(d_rsqrt_diag)

		d_sqrt_diag = []
		for vec in torch.reciprocal(d_rsqrt):
			d_sqrt_diag.append(torch.diag(vec))
		self.d_sqrt_diag = torch.stack(d_sqrt_diag)

		d_inv_diag = []
		for vec in d_inv:
			d_inv_diag.append(torch.diag(vec))
		d_inv_diag = torch.stack(d_inv_diag)

		ut_dinv_u = torch.sum(torch.mul(torch.pow(u,2), d_inv), dim=1)
		eta = torch.reciprocal(ut_dinv_u + ut_dinv_u.new_ones(ut_dinv_u.shape))
		self.sqrt_eta = torch.sqrt(eta)
		coeff = eta.new_ones(eta.shape) - self.sqrt_eta
		coeff = torch.div(coeff, ut_dinv_u).view(-1,1,1)
		self.u = torch.unsqueeze(u, -1) # add a dimension before transposing
		self.u_t = torch.transpose(self.u,1,2)
		matrices = [d_inv_diag, self.u, coeff, self.u_t, self.d_rsqrt_diag]
		self.r = self.d_rsqrt_diag - reduce(torch.bmm, matrices) # eq. 21


	def construct_matrix_inverse(self):
		"""Construct <self.r_inv>, the inverse of <self.r>"""
		coeff = torch.reciprocal(self.sqrt_eta)
		coeff = coeff + self.sqrt_eta.new_ones(self.sqrt_eta.shape)
		coeff = torch.reciprocal(coeff).view(-1,1,1)
		matrices = [self.d_rsqrt_diag, self.u, coeff, self.u_t]
		self.r_inv = self.d_sqrt_diag + reduce(torch.matmul, matrices)


	def batch_shape(self, x=None):
		return (self.mu.shape[0],)


	def event_shape(self):
		return (self.mu.shape[1],)


	def sample(self):
		eps = Variable(torch.randn(self.mu.size()).type_as(self.mu.data))
		z = self.mu + torch.bmm(self.r, eps.unsqueeze(-1)).squeeze(-1)
		return z if self.reparameterized else z.detach()


	def log_prob(self, x):
		return self.batch_log_pdf(x)


	def batch_log_pdf(self, x):
		assert x.shape == self.mu.shape
		assert self.log_pdf_mask is None
		if self.r_inv is None:
			self.construct_matrix_inverse()
		spherical = torch.bmm(self.r_inv, (x - self.mu).unsqueeze(-1)).squeeze(-1)
		if torch.sum(torch.isnan(spherical)) > 0:
			print("caught in batch log pdf spherical")
			print(torch.max(x), torch.min(x))
			print(self.mu[0,:10])
			quit()
		log_pxs = -0.5 * (math.log(2.0 * np.pi) + torch.pow(spherical , 2))
		if torch.sum(torch.isnan(log_pxs)) > 0:
			print("caught in batch log pdf log pxs")
			print(torch.max(x), torch.min(x))
			print(self.mu[0,:10])
			quit()
		return torch.sum(log_pxs, dim=1)


	def analytic_mean(self):
		return self.mu


	def analytic_var(self):
		raise NotImplementedError



if __name__ == '__main__':
	shape = (5,4)
	sphere_dist = RankOneNormal(torch.zeros(shape), torch.zeros(shape), torch.zeros(shape))
	rand_dist = RankOneNormal(torch.rand(shape), torch.rand(shape), torch.rand(shape))
	s = rand_dist.sample()
	print("random sample")
	print(s)
	print("log p:")
	print(rand_dist.batch_log_pdf(s))
	print("min log p")
	print(sphere_dist.batch_log_pdf(torch.zeros(shape)))
	print("other log p")
	print(sphere_dist.batch_log_pdf(s))
