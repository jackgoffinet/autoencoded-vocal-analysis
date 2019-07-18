import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="syllable_modeling",
	version="0.0.1",
	author="Jack Goffinet",
	author_email="jg420@duke.edu",
	description="Generative modeling of animal vocalizations",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/jackgoffinet/syllable_modeling",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
)
