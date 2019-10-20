## Autoencoded Vocal Analysis
#### Generative modeling of animal vocalizations
November 2018 - October 2019

[zebra_finch](https://raw.githubusercontent.com/jackgoffinet/autoencoded-vocal-analysis/master/zebra_finch.png)

See our preprint on bioRxiv (coming soon) for details. See `mouse_sylls_mwe.py` for usage.

To build package:
```
$ cd path/to/ava
$ python setup.py sdist bdist_wheel
```

To build docs:
```
$ cd path/to/ava/docs
$ sphinx-apidoc -f -o docs/source ../ava
$ make html
$ open build/html/index.html
```

Dependencies:
* [PyTorch](https://pytorch.org)
* [Joblib](https://joblib.readthedocs.io/)
* [UMAP](https://umap-learn.readthedocs.io/)
* [affinewarp](https://github.com/ahwillia/affinewarp)

Issues and/or pull requests are appreciated!
