"""
Semisupervised noise detection.


TO DO:
	- parallelize audio segmenting
"""
__author__ = "Jack Goffinet"
__date__ = "February 2019"


import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from tqdm import tqdm
import umap
import hdbscan

from multiprocessing import Pool
from itertools import repeat



class NoiseDetector():
	"""Noise detector base class."""

	def __init__(self, load_dirs, load_filename, save_filename, params, funcs, prefit=False, batch_queries=False, max_num_files=100):
		# Collect labeled data.
		self.save_filename = save_filename
		self.funcs = funcs
		self.prefit = prefit
		self.batch_queries = batch_queries
		self.p = {**funcs['default_params'], **params}
		if load_filename is not None:
			labeled_data = np.load(load_filename).item()
		else:
			labeled_data = {
					'specs':np.array([], dtype='object'),
					'filenames':np.array([], dtype=np.str_),
					'onsets':np.array([], dtype='int'),
					'offsets':np.array([], dtype='int'),
					'labels':np.array([], dtype='int')
			}
		temp = zip(labeled_data['filenames'], labeled_data['onsets'])
		labeled_set = {(i,j):0 for i,j in temp}
		# Segment audio and collect unlabeled data.
		print("Segmenting audio...")
		unlabeled_data = {
				'specs':[],
				'filenames':[],
				'onsets':[],
				'offsets':[],
				'labels':[]
		}
		filenames = []
		for load_dir in load_dirs:
			filenames += [load_dir + i for i in os.listdir(load_dir) if i[-4:] in ['.wav', '.mat']]
		filenames = np.array(filenames)
		assert len(filenames) > 0
		np.random.shuffle(filenames)
		filenames = filenames[:max_num_files]
		with Pool(os.cpu_count()-1) as pool:
			results = pool.starmap(segment_audio_in_file, zip(repeat(self), filenames, repeat(labeled_set)))
		for result in results:
			for key in unlabeled_data:
				unlabeled_data[key] += result[key]
		for field in unlabeled_data:
			if field == 'filenames':
				unlabeled_data[field] = np.array(unlabeled_data[field])
			else:
				temp = np.empty(len(unlabeled_data[field]), dtype=labeled_data[field].dtype)
				temp[:] = unlabeled_data[field]
				unlabeled_data[field] = temp
		# np.save('unlabeled_data.npy', unlabeled_data)
		# print("dt", dt)
		# unlabeled_data = np.load('unlabeled_data.npy').item()
		# dt = 0.004096
		self.specs = np.concatenate((labeled_data['specs'], unlabeled_data['specs']))
		self.filenames = np.concatenate((labeled_data['filenames'], unlabeled_data['filenames']))
		self.onsets = np.concatenate((labeled_data['onsets'], unlabeled_data['onsets']))
		self.offsets = np.concatenate((labeled_data['offsets'], unlabeled_data['offsets']))
		self.labels = np.concatenate((labeled_data['labels'], unlabeled_data['labels']))
		self.N_l = len(labeled_data['specs'])
		self.N_ul = len(unlabeled_data['specs'])
		self.N = self.N_l + self.N_ul


	def ask_for_label(self, index):
		"""Get a true label from the user."""
		dt, fs = self.p['dt'], self.p['fs']
		dur_seconds = 1.0
		dur_samples = int(round(dur_seconds * fs))
		filename = self.filenames[self.N_l + index]
		onset = self.onsets[self.N_l + index]
		offset = self.offsets[self.N_l + index]
		start = int(round(0.5 * dt * fs * (onset+offset))) - dur_samples // 2
		end = start + dur_samples
		start = max(0, start)
		end = min(self.funcs['get_wav_len'](filename), end)
		spec, _, _, _, _ = self.funcs['get_spec'](filename, self.p, start_index=start, end_index=end)
		plt.imshow(spec, origin='lower', aspect='auto', \
				extent=[start/fs,end/fs,0,self.p['num_freq_bins']])
		plt.axvline(x=onset*dt, c='r', lw=0.5)
		plt.axvline(x=offset*dt, c='r', lw=0.5)
		plt.title("Set label:")
		plt.savefig('temp.pdf')
		plt.close('all')
		response = 'invalid'
		while response not in [0,1]:
			response = input("Set label: ")
			try:
				response = int(response)
				assert response in [0,1]
			except (ValueError, AssertionError):
				print("Invalid response!")
				pass
		return response


	def train(self):
		"""Repeatedly cluster and ask for user input."""
		i = 0
		while True:
			print("iteration", str(i)+":", self.N_l, self.N_ul, self.N)
			self.run_training_iteration(i)
			response = 'not valid'
			while response not in ['', 'y', 'n']:
				response = input("Continue? [y]/n ")
			if response == 'n':
				return
			i += 1


	def run_training_iteration(self, iter_num):
		"""Run a single training iteration."""
		# Pre-computation.
		self.pre_iteration_computation()
		# Plot.
		if iter_num > 0 or not self.prefit or self.N_l > 0:
			self.plot()
		# Get user input.
		avail_indices, taken_indices = list(range(self.N_ul)), []
		if iter_num == 0 and self.prefit and self.N_l == 0:
			print("Sort unlabeled until both labels are found:")
			unique_labels = []
			n = 0
			while len(unique_labels) < 2:
				index = np.random.choice(avail_indices)
				avail_indices.remove(index)
				taken_indices.append(index)
				# Ask for a label.
				label = self.ask_for_label(index)
				if label not in unique_labels:
					unique_labels.append(label)
				self.labels[self.N_l + index] = label
				# Update classifier.
				self.new_label_computation(self.N_l+index, label)
				n += 1
		else:
			n = min(5, len(avail_indices))
			temp = input("Sort unlabeled data ["+str(n)+"]: ")
			try:
				n = min(int(temp), len(avail_indices))
			except ValueError:
				pass
			if self.batch_queries:
				indices = self.get_queries(n)
			for i in range(n):
				if self.batch_queries:
					index = indices[i]
				else:
					spec_scores = self.get_spec_scores()
					big_number = np.max(spec_scores) - np.min(spec_scores) + 1.0
					for j in taken_indices:
						spec_scores[j] -= big_number
					# Select the best.
					index = np.argmax(spec_scores)
				avail_indices.remove(index)
				taken_indices.append(index)
				# Ask for a label.
				label = self.ask_for_label(index)
				self.labels[self.N_l + index] = label
				# Update classifier.
				self.new_label_computation(self.N_l+index, label)
		# Move answers to the labeled portion of the dataset.
		avail_indices = np.array(avail_indices, dtype='int') + self.N_l
		taken_indices = np.array(taken_indices, dtype='int') + self.N_l
		perm = np.concatenate((np.arange(self.N_l), taken_indices, avail_indices))
		self.specs = self.specs[perm]
		self.filenames = self.filenames[perm]
		self.onsets = self.onsets[perm]
		self.offsets = self.offsets[perm]
		self.labels = self.labels[perm]
		self.N_l += n
		self.N_ul -= n
		# Then save things.
		self.save_state()


	def classify(self, spec):
		"""Classify the given spectrogram."""
		return self.batch_classify(np.array([spec]))


	def batch_classify(self, specs):
		"""Classify the given an array of spectrograms."""
		raise NotImplementedError


	def get_spec_scores(self):
		"""Score the unlabeled spectrograms."""
		raise NotImplementedError

	def get_queries(self, n):
		"""Return the indices of spectrograms to label."""
		raise NotImplementedError


	def pre_iteration_computation(self):
		"""Do computation before a training iteration."""
		raise NotImplementedError


	def new_label_computation(self, index, label):
		"""Adjust beliefs given a new label."""
		raise NotImplementedError


	def save_state(self):
		"""Save all the labeled data."""
		d = {
			'specs':self.specs[:self.N_l],
			'filenames':self.filenames[:self.N_l],
			'onsets':self.onsets[:self.N_l],
			'offsets':self.offsets[:self.N_l],
			'labels':self.labels[:self.N_l]
		}
		np.save(self.save_filename, d)



def segment_audio_in_file(self, filename, labeled_set):
	unlabeled_data = {
			'specs':[],
			'filenames':[],
			'onsets':[],
			'offsets':[],
			'labels':[]
	}
	spec, f, dt, i1, i2 = self.funcs['get_spec'](filename, self.p)
	# audio = self.funcs['get_audio'](filename, self.p)
	if 'f' not in self.p['seg_params']:
		self.p['seg_params']['f'] = f
	if 'dt' not in self.p:
		self.p['dt'] = dt
	# Add file-level segmentation parameters if they exist.
	try:
		temp = '/'.join(filename.split('/')[:-1])+'/seg_params.npy'
		temp = np.load(temp).item()
		self.p['seg_params'] = {**self.p['seg_params'], **temp}
	except FileNotFoundError:
		pass
	onsets, offsets = self.funcs['get_onsets_offsets'](spec, dt, self.p['seg_params'])
	# Remove the data that's already been labeled.
	mask = np.array([i for i in range(len(onsets)) if (filename, onsets[i]) not in labeled_set], dtype='int')
	onsets, offsets = np.array(onsets)[mask].tolist(), np.array(offsets)[mask].tolist()
	# Get spectrograms.
	syll_specs, _ = self.funcs['get_syll_specs'](onsets, offsets, spec, 0.0, dt, self.p)
	# Append stuff to <unlabeled_data>.
	unlabeled_data['specs'] += syll_specs
	unlabeled_data['filenames'] += [filename] * len(onsets)
	unlabeled_data['onsets'] += onsets
	unlabeled_data['offsets'] += offsets
	unlabeled_data['labels'] += [-1] * len(onsets)
	return unlabeled_data



class GaussianProcessDetector(NoiseDetector):
	"""
	Classify spectrograms using dimensionality reduction and a GP classifier.
	"""


	def __init__(self, load_dirs, load_filename, save_filename, params, funcs, ndims=2, max_num_files=100):
		super(GaussianProcessDetector, self).__init__(load_dirs, load_filename, save_filename, params, funcs, prefit=True, batch_queries=True, max_num_files=max_num_files)
		self.ndims = ndims
		self.clf = None
		self.reducer = None


	def pre_iteration_computation(self):
		"""Do computation before a training iteration."""
		# Fit the classifier.
		self.dim_reduce()
		embedding = self.reducer.embedding_
		kernel = 1.0 * RBF(1.0)
		self.clf = GaussianProcessClassifier(kernel=kernel)
		if self.N_l > 0:
			self.clf.fit(embedding[:self.N_l], self.labels[:self.N_l])


	def new_label_computation(self, index, label):
		"""Adjust beliefs given a new label."""
		pass


	def get_spec_scores(self):
		"""Score the unlabeled spectrograms."""
		temp = self.reducer.embedding_[self.N_l:].reshape(self.N_ul,-1)
		spec_probs = self.clf.predict_proba(temp)[:,1]
		return -1.0 * np.abs(spec_probs - 0.5)

	def get_queries(self, n):
		"""Return the indices of spectrograms to label."""
		temp = self.reducer.embedding_[self.N_l:].reshape(self.N_ul,-1)
		spec_probs = self.clf.predict_proba(temp)[:,1]
		spec_scores = -1.0 * np.abs(spec_probs - 0.5)
		thresh = np.percentile(spec_scores, 85.0)
		return np.random.choice(np.arange(self.N_ul)[spec_scores > thresh], n, replace=False)

	def dim_reduce(self):
		"""Run spectrograms through UMAP."""
		all_specs = specs_to_block_array(self.specs, self.p['num_freq_bins'], self.p['num_time_bins'])
		all_specs = all_specs.reshape(self.N, -1)
		self.reducer = umap.UMAP(n_components=self.ndims).fit(all_specs, y=self.labels)


	def batch_classify(self, specs, threshold=0.5):
		"""Classify the given an array of spectrograms."""
		if len(specs) == 0:
			return []
		if self.reducer is None or self.clf is None:
			self.pre_iteration_computation()
		all_specs = specs_to_block_array(specs, self.p['num_freq_bins'], self.p['num_time_bins'])
		all_specs = all_specs.reshape(len(specs), -1)
		temp = self.reducer.transform(all_specs).reshape(len(specs),-1)
		spec_probs = self.clf.predict_proba(temp)[:,1]
		return [spec_prob > threshold for spec_prob in spec_probs]


	def plot(self):
		if self.reducer is None:
			self.dim_reduce()
		cmap = plt.cm.RdBu
		all_specs = specs_to_block_array(self.specs, self.p['num_freq_bins'], self.p['num_time_bins'])
		all_specs = all_specs.reshape(self.N, -1)
		vis_umap = umap.UMAP().fit(all_specs,y=self.labels)
		embedding = vis_umap.embedding_
		temp = self.reducer.embedding_[self.N_l:].reshape(self.N_ul,-1)
		spec_probs = self.clf.predict_proba(temp)[:,1]
		plt.scatter(embedding[self.N_l:,0], embedding[self.N_l:,1], c=spec_probs, alpha=0.3, s=0.5)
		indices = np.where(self.labels[:self.N_l] == 0)
		plt.scatter(embedding[indices,0], embedding[indices,1], c='red', alpha=0.9, s=1.2)
		indices = np.where(self.labels[:self.N_l] == 1)
		plt.scatter(embedding[indices,0], embedding[indices,1], c='goldenrod', alpha=0.9, s=1.2)
		plt.savefig('temp.pdf')
		plt.close('all')



# class ClusteringDetector(NoiseDetector):
# 	"""
# 	Classify spectrograms using dimensionality reduction and clustering.
# 	"""
#
# 	def __init__(self, load_dirs, load_filename, save_filename, params, funcs):
# 		super(ClusteringDetector, self).__init__(load_dirs, load_filename, save_filename, params, funcs, prefit=False)
# 		self.reducer, self.clusterer = None, None
#
#
# 	def batch_classify(self, specs, threshold=0.5):
# 		"""Classify the spectrograms."""
# 		if len(specs) == 0:
# 			return []
# 		if self.reducer is None or self.clusterer is None:
# 			self.dim_reduce_and_cluster()
# 		self.cluster_prob = (self.cluster_positive_counts + 0.5) / (self.cluster_counts + 1.0)
# 		specs = specs_to_block_array(specs, self.p['num_freq_bins'], self.p['num_time_bins'])
# 		specs = specs.reshape(len(specs), -1)
# 		embedding = self.reducer.transform(specs)
# 		temp = hdbscan.prediction.membership_vector(self.clusterer, embedding.astype(np.float64))
# 		soft_labels = np.zeros((temp.shape[0], temp.shape[1]+1))
# 		soft_labels[:,:-1] = temp
# 		soft_labels[:,-1] = 1.0 - np.sum(soft_labels, axis=1)
# 		spec_probs = np.einsum('ic,c->i', soft_labels, self.cluster_prob)
# 		return [prob > threshold for prob in spec_probs]
#
#
# 	def pre_iteration_computation(self):
# 		"""Do computation before a training iteration."""
# 		self.dim_reduce_and_cluster()
# 		self.compute_cluster_scores()
#
#
# 	def get_spec_scores(self):
# 		"""Score the unlabeled spectrograms."""
# 		return np.einsum('ij,j->i', self.soft_labels[self.N_l:], self.cluster_scores)
#
#
# 	def new_label_computation(self, index, label):
# 		"""Adjust beliefs given a new label."""
# 		self.cluster_counts += self.soft_labels[index]
# 		if label == 1:
# 			self.cluster_positive_counts += self.soft_labels[index]
# 		self.compute_cluster_scores()
#
#
# 	def compute_cluster_scores(self):
# 		"""Weighted expected change in accuracy, w/ Jeffreys prior"""
# 		self.cluster_prob = (self.cluster_positive_counts + 0.5) / (self.cluster_counts + 1.0)
# 		cluster_prob_pos = (self.cluster_positive_counts + 1.5) / (self.cluster_counts + 2.0)
# 		cluster_prob_neg = (self.cluster_positive_counts + 0.5) / (self.cluster_counts + 2.0)
# 		cluster_g = 2.0*self.cluster_prob**2 - 2.0*self.cluster_prob + 1.0
# 		cluster_g_pos = 2.0*cluster_prob_pos**2 - 2.0*cluster_prob_pos + 1.0
# 		cluster_g_neg = 2.0*cluster_prob_neg**2 - 2.0*cluster_prob_neg + 1.0
# 		cluster_delta_g = self.cluster_prob * (cluster_g_pos - cluster_g) + \
# 						(1.0 - self.cluster_prob) * (cluster_g_neg - cluster_g)
# 		self.cluster_scores = self.cluster_weights * cluster_delta_g
#
#
# 	def dim_reduce_and_cluster(self):
# 		# Perform semisupervised dimensionality reduction.
# 		all_specs = specs_to_block_array(self.specs, self.p['num_freq_bins'], self.p['num_time_bins'])
# 		all_specs = all_specs.reshape(self.N, -1)
# 		self.reducer = umap.UMAP().fit(all_specs, y=self.labels)
# 		embedding = self.reducer.embedding_
# 		# Then cluster.
# 		self.clusterer = hdbscan.HDBSCAN(prediction_data=True)
# 		self.clusterer.fit(embedding.astype(np.float64))
# 		# Precompute some things.
# 		temp = hdbscan.all_points_membership_vectors(self.clusterer)
# 		self.soft_labels = np.zeros((temp.shape[0], temp.shape[1]+1))
# 		print("soft_labels", self.soft_labels.shape)
# 		self.soft_labels[:,:-1] = temp
# 		self.soft_labels[:,-1] = 1.0 - np.sum(self.soft_labels, axis=1)
# 		n_clusters = self.soft_labels.shape[1]
# 		print("n clusters", n_clusters)
# 		self.cluster_weights = np.sum(self.soft_labels, axis=0)
# 		self.cluster_counts = np.zeros(n_clusters)
# 		self.cluster_positive_counts = np.zeros(n_clusters)
# 		for i, label in enumerate(self.labels[:self.N_l]):
# 			if label == 1:
# 				self.cluster_positive_counts[:] += self.soft_labels[i]
# 			self.cluster_counts[:] += self.soft_labels[i]
#
#
# 	def plot(self, filename='temp.pdf'):
# 		"""Plot the embedding and assigned probabilities."""
# 		if self.reducer is None or self.clusterer is None:
# 			self.dim_reduce_and_cluster()
# 		_, axarr = plt.subplots(3,1, sharex=True)
# 		axarr[0].set_title("HDBSCAN Clustering")
# 		embedding = self.reducer.embedding_
# 		axarr[0].scatter(embedding[:,0], embedding[:,1], c=self.clusterer.labels_, cmap='Spectral', alpha=0.3, s=0.5)
# 		axarr[2].set_title("Inferred Labels")
# 		predictions = np.einsum('ij,j->i', self.soft_labels, self.cluster_prob)
# 		plot1 = axarr[2].scatter(embedding[:,0], embedding[:,1], c=predictions, cmap='viridis', vmin=0, vmax=1, alpha=0.3, s=0.5)
# 		cbar = plt.colorbar(plot1,ax=axarr[2])
# 		cbar.solids.set_rasterized(True)
# 		axarr[1].set_title("True Labels")
# 		axarr[1].scatter(embedding[self.N_l:,0], embedding[self.N_l:,1], c='k', alpha=0.04, s=0.5)
# 		axarr[1].scatter(embedding[:self.N_l,0], embedding[:self.N_l,1], c=self.labels[:self.N_l], cmap='viridis', vmin=0, vmax=1, alpha=0.8, s=0.8)
# 		plt.savefig(filename)
# 		plt.close('all')



# class NaiveBayesDetector(NoiseDetector):
# 	"""
# 	Classify spectrograms using dimensionality reduction and Naive Bayes.
# 	"""
#
#
# 	def __init__(self, load_dirs, load_filename, save_filename, params, funcs):
# 		super(NaiveBayesDetector, self).__init__(load_dirs, load_filename, save_filename, params, funcs, prefit=True)
# 		self.clf = None
# 		self.reducer = None
#
#
# 	def pre_iteration_computation(self):
# 		"""Do computation before a training iteration."""
# 		# Fit the classifier.
# 		self.dim_reduce()
# 		embedding = self.reducer.embedding_
# 		self.clf = GaussianNB()
# 		if self.N_l > 0:
# 			self.clf.partial_fit(embedding[:self.N_l], self.labels[:self.N_l], classes=[0,1])
# 		else:
# 			self.clf.partial_fit([[0,0],[1,1]], [0,1], classes=[0,1])
#
#
# 	def new_label_computation(self, index, label):
# 		"""Adjust beliefs given a new label."""
# 		# Update classifier.
# 		self.clf.partial_fit([self.reducer.embedding_[index]], [label])
#
#
# 	def get_spec_scores(self):
# 		"""Score the unlabeled spectrograms."""
# 		temp = self.reducer.embedding_[self.N_l:].reshape(self.N_ul,-1)
# 		spec_probs = self.clf.predict_proba(temp)[:,1]
# 		return -1.0 * np.abs(spec_probs - 0.5)
#
#
# 	def dim_reduce(self):
# 		"""Run spectrograms through UMAP."""
# 		all_specs = specs_to_block_array(self.specs, self.p['num_freq_bins'], self.p['num_time_bins'])
# 		all_specs = all_specs.reshape(self.N, -1)
# 		self.reducer = umap.UMAP().fit(all_specs, y=self.labels)
#
#
# 	def batch_classify(self, specs):
# 		"""Classify the given an array of spectrograms."""
# 		raise NotImplementedError
#
#
# 	def plot(self):
# 		h = 0.25 # mesh step size
# 		gap = 1.0
# 		if self.reducer is None:
# 			self.dim_reduce()
# 		cmap = plt.cm.RdBu
# 		embedding = self.reducer.embedding_
# 		x_min, x_max = np.min(embedding[:,0])-gap, np.max(embedding[:,0])+gap
# 		y_min, y_max = np.min(embedding[:,1])-gap, np.max(embedding[:,1])+gap
# 		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# 		Z = self.clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
# 		Z = Z.reshape(xx.shape)
# 		plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
# 		plt.scatter(embedding[self.N_l:,0], embedding[self.N_l:,1], c='k', alpha=0.3, s=0.5)
# 		plt.scatter(embedding[:self.N_l,0], embedding[:self.N_l,1], c=self.labels[:self.N_l], alpha=0.9, s=0.8)
# 		plt.savefig('temp.pdf')
# 		plt.close('all')


def specs_to_block_array(specs, height, width):
	"""Turn an object array into a proper zero-padded array."""
	result = np.zeros((len(specs), height, width))
	for i, spec in enumerate(specs):
		result[i,:spec.shape[0],:spec.shape[1]] = spec[:height,:width]
	return result



###
