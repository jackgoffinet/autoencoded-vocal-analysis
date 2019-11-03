Training
========

This section describes how to train the VAE.

Training on Syllables
#####################

By this point, we should have a list of directories containing preprocessed
syllable spectrograms. Our first step is to define PyTorch Dataloader objects
that are responsible for shipping data to and from the GPU. We want one for a
training set, and one for a test set.

.. code:: Python3

	spec_dirs = [...] # directories containing saved spectrograms (hdf5s)
	split = 0.8 # 80/20 train/test split

	# Construct a random train/test partition.
	from ava.models.vae_dataset import get_syllable_partition
	partition = get_syllable_partition(spec_dirs, split)

	# Make Dataloaders.
	from ava.model.vae_dataset import get_syllable_data_loaders
	loaders = get_syllable_data_loaders(partition)



Now we're ready to train the VAE.


.. code:: Python3

	# Construct network.
	from ava.models.vae import VAE
	save_dir = 'model/parameters/should/be/saved/here/'
	model = VAE(save_dir=save_dir)

	# Train.
	model.train_loop(loaders, epochs=101)



This should periodically save the model, print train and test loss, and write
a file in :code:`save_dir` called :code:`reconstruction.pdf` which displays
several spectrograms and their reconstructions.

You may also want to continue training a previously saved model:


.. code:: Python3

	# Make an untrained model.
	model = VAE(save_dir=save_dir)

	# Load saved state.
	model_checkpoint = 'path/to/checkpoint_100.tar'
	model.load_state(model_checkpoint)

	# Train another few epochs.
	model.train_loop(loaders, epochs=51)


Shotgun VAE Training
####################

TO DO

Warped Shotgun VAE Training
###########################

TO DO
