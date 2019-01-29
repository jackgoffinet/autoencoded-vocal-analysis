from __future__ import print_function, division
"""Dataset for animal vocalization syllables

TO DO: time shift transform
"""
__author__ = "Jack Goffinet"
__date__ = "November 2018"

import numpy as np
import h5py
from os import listdir, sep
from os.path import join
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
plt.switch_backend('agg')



def get_partition(dirs, split):
	assert(split > 0.0 and split <= 1.0)
	filenames = []
	for dir in dirs:
		filenames += [join(dir, i) for i in listdir(dir) if i[-5:] == '.hdf5']
	np.random.seed(42)
	np.random.shuffle(filenames)
	# filenames = [i for i in filenames if 'dur' not in i]
	# NOTE: TEMP
	# filenames = [i for i in filenames if int(i.split(sep)[-1].split('.')[0].split('_')[1]) <= 20]
	index = int(round(split * len(filenames)))
	return {'train': filenames[:index], 'test': filenames[index:]}


def get_data_loaders(partition, batch_size=64, num_time_bins=128, \
			shuffle=(True, False), time_shift=(True, True), sylls_per_file=1000):
	transforms_list = [ToTensor(),
			transforms.Compose([TimeShift(num_time_bins), ToTensor()])]
	train_dataset = SyllableDataset(filenames=partition['train'], \
			transform=transforms_list[int(time_shift[0])],
			sylls_per_file=sylls_per_file)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, \
			shuffle=shuffle[0], num_workers=4)
	if not partition['test']:
		return train_dataloader, None
	test_dataset = SyllableDataset(filenames=partition['test'], \
			transform=transforms_list[int(time_shift[1])],
			sylls_per_file=sylls_per_file)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, \
			shuffle=shuffle[1], num_workers=4)
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
			filename = self.filenames[i // self.sylls_per_file]
			file_index = i % self.sylls_per_file
			# Then the image is a slice of the file's spectrogram.
			with h5py.File(filename, 'r') as f:
				image = f['syll_specs'][file_index]
				duration = f['syll_lens'][file_index]
				time = f['syll_times'][file_index]
			# individual = int(filename.split('/')[-2][1])
			# individual = int(filename.split('/')[-2].split('_')[1]) # for helium mice
			# session = int(filename.split('/')[-2].split('_')[2][1:])
			# condition = int(session != 1)
			sample = {
					'image': image,
					'duration': duration,
					'time': time,
					# 'individual': individual,
					# 'session': session,
					# 'conddition': condition,
			}
			if self.transform:
				sample = self.transform(sample)
			result.append(sample)
		if single_index:
			return result[0]
		return result


class ToTensor(object):
	"""Convert numpy arrays to pytorch tensors."""

	def __call__(self, sample):
		image = sample['image']
		image -= np.min(image)
		image *= 0.9 / np.max(image)
		image += 0.05
		img = torch.from_numpy(image).type(torch.FloatTensor)
		# # NOTE: TEMP
		# img = np.zeros((128,128))
		# img[:64,:64] = 1.0
		# img[64:,64:] = 1.0
		# img = np.roll(img, np.random.randint(64), axis=0)
		# img = torch.from_numpy(img).type(torch.FloatTensor)

		sample['image'] = img
		return sample


class TimeShift(object):
	"""Shift the spectrogram randomly in time."""

	def __init__(self, num_time_bins):
		self.num_time_bins = num_time_bins

	def __call__(self, sample):
		image = sample['image']
		dur = sample['duration']
		shift = np.random.randint(self.num_time_bins - dur + 1)
		img = np.roll(image, shift, axis=1)
		sample['image'] = img
		return sample



def save_image(image, filename='temp.pdf'):
	plt.imshow(image, aspect='auto', origin='lower')
	plt.savefig(filename)
	plt.close('all')




if __name__ == '__main__':
	pass
