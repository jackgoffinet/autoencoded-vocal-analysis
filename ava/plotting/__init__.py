"""
ava/plotting

"""
from .cluster_pca_plot import cluster_pca_plot_DC, cluster_pca_feature_plot_DC
from .feature_correlation_plots import correlation_plot_DC, \
		knn_variance_explained_plot_DC, pairwise_correlation_plot_DC, \
		feature_pca_plot_DC, two_subplot_correlation_plot_DC, \
		triptych_correlation_plot_DC
from .grid_plot import indexed_grid_plot, grid_plot
# from .html_plots import save_image, make_html_plot # DELETE?
# from .instantaneous_plots import ...
# from .latent_axes_plot import ...
from .latent_projection import latent_projection_plot_DC, projection_plot
# from .longitudinal_gif import ...
from .mmd_plots import mmd_matrix_DC
from .pairwise_distance_plots import pairwise_distance_scatter_DC, \
		pairwise_distance_scatter, knn_display_DC, bridge_plot_DC, \
		random_walk_plot
from .rolloff_plot import clustering_performance_plot
from .tooltip_plot import tooltip_plot_DC, tooltip_plot
from .trace_plot import trace_plot, spectrogram_plot, variability_plot, \
		warped_trace_plot_DC, warped_variability_plot_DC
# from .vocal_motor_gif import ...
