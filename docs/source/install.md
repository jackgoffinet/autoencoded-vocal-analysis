Install
=======

AVA requires [Python3](https://www.python.org/) and standard packages available
in the [Anaconda distribution](https://www.anaconda.com/distribution/). In
addition to these packages, AVA also requires [PyTorch](https://pytorch.org)
(>=v1.1), [UMAP](https://umap-learn.readthedocs.io/), and
[affinewarp](https://github.com/ahwillia/affinewarp).

AVA can be installed by opening a
[shell](https://en.wikipedia.org/wiki/Command-line_interface), navigating to a
suitable directory, and entering the following commands:

```
$ git clone https://github.com/jackgoffinet/autoencoded-vocal-analysis.git
$ cd autoencoded-vocal-analysis
$ pip install .
```

This will install AVA as a python package in your current environment.


#### Operating systems
> AVA is built and tested with a Linux machine running Ubuntu. It has also run
> on Mac and Windows, but open
> [issues](https://github.com/pytorch/pytorch/issues/12831) in PyTorch make for
> slow training on Windows machines.

#### GPU Acceleration
> Like most neural network models, the VAE trains much faster on GPU than on
> CPU. Training times on GPU are about 30 minutes to a couple hours, and roughly
> 10-20x slower on CPU.
