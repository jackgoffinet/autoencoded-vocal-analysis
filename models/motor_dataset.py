from __future__ import print_function, division
"""Dataset for animal vocalizations"""

__author__ = "Jack Goffinet"
__date__ = "December 2018"

import numpy as np
import h5py
from os import listdir, sep
from os.path import join
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_partition(dirs, split):
	assert(split > 0.0 and split <= 1.0)
	filenames = []
	for dir in dirs:
		filenames += [join(dir, i) for i in listdir(dir) if i[-5:] == '.hdf5']
	np.random.seed(42)
	np.random.shuffle(filenames)
	index = int(round(split * len(filenames)))
	return {'train': filenames[:index], 'test': filenames[index:]}


def get_data_loaders(partition, batch_size=64, num_time_bins=128, \
			shuffle=(True, False), sylls_per_file=1000):
	train_dataset = SyllableDataset(filenames=partition['train'], \
			transform=ToTensor(), sylls_per_file=sylls_per_file)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, \
			shuffle=shuffle[0], num_workers=4)
	if not partition['test']:
		return {'train':train_loader, 'test':None}
	test_dataset = SyllableDataset(filenames=partition['test'], \
			transform=ToTensor(), sylls_per_file=sylls_per_file)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, \
			shuffle=shuffle[1], num_workers=4)
	return {'train':train_loader, 'test':test_loader}


class SyllableDataset(Dataset):
	"""Dataset for animal vocalization syllables"""

	def __init__(self, filenames, sylls_per_file=1000, transform=None):
		self.filenames = filenames
		self.sylls_per_file = sylls_per_file
		self.transform = transform

	def __len__(self):
		return len(self.filenames) * self.sylls_per_file

	def __getitem__(self, index):
		# First find the file.
		filename = self.filenames[index // self.sylls_per_file]
		file_index = index % self.sylls_per_file
		with h5py.File(filename, 'r') as f:
			image = f['syll_specs'][file_index]
			duration = f['syll_lens'][file_index]
			time = f['syll_times'][file_index]
		individual = int(filename.split('/')[-2][1])
		sample = {
				'image': image,
				'duration': duration,
				'time': time,
				'individual': individual,
		}
		if self.transform:
			sample = self.transform(sample)
		return sample



class ToTensor(object):
	"""Convert numpy arrays to pytorch tensors."""

	def __call__(self, sample):
		image = sample['image']
		image -= np.min(image)
		image *= 0.9 / np.max(image)
		image += 0.05
		img = torch.from_numpy(image).type(torch.FloatTensor)
		sample['image'] = img.permute(1,0)
		return sample





if __name__ == '__main__':
	pass


###
