"""
Measure correlations between latent features and traditional features.

NOTE: FINISH THIS!
"""
__author__ = "Jack Goffinet"
__date__ = "July 2019"


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import numpy as np
from sklearn.linear_model import LinearRegression

from .data_container import PRETTY_NAMES



def correlation_plot_DC(dc, fields, filename='feature_correlation.pdf'):
	"""


	"""
	latent = dc.request('latent_means')
	field_data = {}
	field_corrs = {}
	for field in fields:
		field_data[field] = dc.request(field)
		field_corrs[field] = get_correlation(latent, field_data[field])
	filename = os.path.join(dc.plots_dir, filename)
	# Sort.
	corrs = np.array([field_corrs[field] for field in fields])
	perm = np.argsort(corrs)
	corrs = corrs[perm]
	fields = np.array(fields)[perm]
	# Plot.
	X = np.arange(len(field_data))
	tick_labels = [PRETTY_NAMES[field] for field in fields]
	plt.bar(X, corrs)
	plt.xticks(X, tick_labels, rotation=45)
	plt.tight_layout()
	plt.title("Latent/Traditional Feature $R^2$ Values")
	plt.savefig(filename)
	plt.close('all')



def get_correlation(latent, feature_vals):
	""" """
	reg = LinearRegression().fit(latent, feature_vals.reshape(-1,1))
	return reg.score(latent, feature_vals.reshape(-1,1))
