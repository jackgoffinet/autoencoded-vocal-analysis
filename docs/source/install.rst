Install
=======

AVA requires `Python3 <https://www.python.org/>`__ and standard packages available
in the `Anaconda distribution <https://www.anaconda.com/distribution/>`__. In
addition to these packages, AVA also requires `PyTorch <https://pytorch.org>`__
(>=v1.1), `UMAP <https://umap-learn.readthedocs.io/>`__, and
`affinewarp <https://github.com/ahwillia/affinewarp>`__.

Installation
############

AVA can be installed by opening a
`shell <https://en.wikipedia.org/wiki/Command-line_interface>`__, navigating to a
suitable directory, and entering the following commands:


.. code:: bash

	$ git clone https://github.com/jackgoffinet/autoencoded-vocal-analysis.git
	$ cd autoencoded-vocal-analysis
	$ pip install .

This will install AVA as a python package in your current environment.


Operating systems
#################

AVA is built and tested with Ubuntu and OSX operating systems. It has also run
on Windows, but open `issues <https://github.com/pytorch/pytorch/issues/12831>`__ in PyTorch make for slow training.

GPU Acceleration
################

Like most neural network models, the VAE trains much faster on GPU than on
CPU. Training times on GPU are about 30 minutes to a few hours, and roughly
10x slower on CPU. If you have a GPU, make sure to install PyTorch with CUDA.
