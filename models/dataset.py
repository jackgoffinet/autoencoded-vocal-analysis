"""Dataset for animal vocalization syllables"""

__author__ = "Jack Goffinet"
__date__ = "November 2018"

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
			transform=ToTensor(),
			sylls_per_file=sylls_per_file)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, \
			shuffle=shuffle[0], num_workers=3)
	if not partition['test']:
		return train_dataloader, None
	test_dataset = SyllableDataset(filenames=partition['test'], \
			transform=ToTensor(),
			sylls_per_file=sylls_per_file)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, \
			shuffle=shuffle[1], num_workers=3)
	return train_dataloader, test_dataloader



class SyllableDataset(Dataset):
	"""Dataset for animal vocalization syllables"""

	def __init__(self, filenames, sylls_per_file=1000, transform=None):
		self.filenames = filenames
		self.sylls_per_file = sylls_per_file
		self.transform = transform

	def __len__(self):
		return len(self.filenames) * self.sylls_per_file

	def __getitem__(self, index):
		result = []
		single_index = False
		try:
			iterator = iter(index)
		except TypeError:
			index = [index]
			single_index = True
		for i in index:
			# First find the file.
			load_filename = self.filenames[i // self.sylls_per_file]
			file_index = i % self.sylls_per_file
			# Then collect fields from the file.
			with h5py.File(load_filename, 'r') as f:
				sample = {
					'image': f['specs'][file_index],
					'duration': f['durations'][file_index],
					'time': f['times'][file_index],
					'file_time': f['file_times'][file_index],
					'filename': str(f['filenames'][file_index]),
				}
			if self.transform:
				sample = self.transform(sample)
			result.append(sample)
		if single_index:
			return result[0]
		return result


class ToTensor(object):

	def __call__(self, sample):
		image = sample['image']
		image -= np.min(image)
		image *= 0.9 / np.max(image)
		image += 0.05
		img = torch.from_numpy(image).type(torch.FloatTensor)
		sample['image'] = img
		return sample



if __name__ == '__main__':
	pass


###
