## Autoencoded Vocal Analysis
#### Generative modeling of animal vocalizations
Current version: 0.2.1

See our [preprint](https://doi.org/10.1101/811661) on bioRxiv for details.

See `mouse_sylls_mwe.py` and the other `*_mwe.py` files for usage.

To build package:
```
$ cd path/to/autoencoded-vocal-analysis
$ python setup.py sdist bdist_wheel
```

To build docs:
```
$ cd path/to/autoencoded-vocal-analysis/docs
$ sphinx-apidoc -f -o docs/source ../ava
$ make html
$ open build/html/index.html
```

Dependencies:
* Python 3
* [PyTorch](https://pytorch.org)
* [Joblib](https://joblib.readthedocs.io/)
* [UMAP](https://umap-learn.readthedocs.io/)
* [affinewarp](https://github.com/ahwillia/affinewarp)
* [Bokeh](https://docs.bokeh.org/en/latest/)

Issues and/or pull requests are appreciated!
