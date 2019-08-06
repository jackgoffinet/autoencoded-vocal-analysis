"""
Plot generated spectrograms along the latent axes.

"""
__author__ = "Jack Goffinet"
__date__ = "August 2019"

from models.vae import VAE

MAX_VAL = 2.5


def latent_axes_plot_DC(dc, height=4, width=5, filename='latent_axes.png'):
	# Request things.
	latent = dc.request('latent_means')
	model_fn = dc.model_filename
	assert model_fn is not None
	model = VAE()
	model.load_state(model_fn)
	# Collect generated spectrograms.
	specs = []
	for i in range(height):
		spec_row = []
		for j in range(width):
			latent = np.zeros((1,model.z_dim))
			# NOTE: FINISH







if __name__ == '__main__':
	quit()


###
