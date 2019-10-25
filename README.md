## Autoencoded Vocal Analysis
#### Generative modeling of animal vocalizations
Current version: 0.2.1

See our [preprint](https://doi.org/10.1101/811661) on bioRxiv for details.

See `examples/` for usage.

See [readthedocs](https://autoencoded-vocal-analysis.readthedocs.io/en/latest/ava.html)
for documentation.

To build:
```
$ git clone https://github.com/jackgoffinet/autoencoded-vocal-analysis.git
$ cd path/to/autoencoded-vocal-analysis
$ pip install .
```

Dependencies:
* Python 3
* [PyTorch](https://pytorch.org)
* [Joblib](https://joblib.readthedocs.io/)
* [UMAP](https://umap-learn.readthedocs.io/)
* [affinewarp](https://github.com/ahwillia/affinewarp)
* [Bokeh](https://docs.bokeh.org/en/latest/)
* [Sphinx read-the-docs theme](https://sphinx-rtd-theme.readthedocs.io/en/stable/)

Issues and/or pull requests are appreciated!

See also:
* [Animal Vocalization Generative Network](https://github.com/timsainb/AVGN), a
	nice repo by Tim Sainburg for clustering birdsong syllables and generating
	syllable interpolations.
* [DeepSqueak](https://github.com/DrCoffey/DeepSqueak) and
	[MUPET](https://github.com/mvansegbroeck/mupet), MATLAB packages for
	detecting and classifying rodent ultrasonic vocalizations.
