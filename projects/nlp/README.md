# Language with NuPIC

This directory contains scripts for running NLP-based applications using [NuPIC](https://github.com/numenta/nupic) and [Cortical.io's API](http://www.cortical.io/developers.html).

Additional requirements:

- [cortipy](https://github.com/numenta/cortipy)
- Python modules [enum](https://pypi.python.org/pypi/enum34), [pandas](http://pandas.pydata.org/), and [numpy](http://www.numpy.org/).
- You must have a valid REST API key from [Cortical.io](http://www.cortical.io/developers.html).

## Use cases
In this directory are scripts for several NLP applications. They each use one or more of the following models:

- Keywords model gives a random (seeded) encoding to each word.
- Cortical.io word fingerprints model encodes each word with their API, and then makes a union representation over the words in a block of text.
- Cortical.io document fingerprints model uses their API to encode a block of text.
- Cortical.io windows model encodes with word-level fingerprints and builds progressive union representations as a "sliding window" over a sequence of text.
- HTM Network models use the Cortical.io word-level encoder, and then run the patterns through the Network API.
- Cortical.io classify endpoint model is an end-to-end classification tool in their API.

#### Text querying

For a set of text samples, query the model with a phrase to get a sorted print out of the most similar text samples. To run this on a subset of the [IMDb movie reviews dataset](http://ai.stanford.edu/~amaas/data/sentiment/):

	python imbu_runner.py data/etc/imdb_subset.csv

This runs the Cortical.io windows model, but you can swap models easily in the script. It may take some time to encode the IMDb data, but the Cortical.io API calls cache so this is slow only the first run.

#### Classification experiments

The experiments test the models on classifying labeled samples of text.

- run_classification_experiment.py includes two experiments:
	- incrementally increases the training size each trial; we expect to see the classification accuracies increase.
	- k-folds cross validation.
- run_buckets_experiment.py: an experiment to quantify model performance for the text querying task.
	


#### Additional examples

Please see the integration tests for the [classification models](https://github.com/numenta/nupic.research/blob/master/tests/nlp/integration/classification_models_validation_test.py) and the [Cortical.io encoders](https://github.com/numenta/nupic.research/blob/master/tests/nlp/integration/cio_encodings_test.py) for example usage.

The ["Fox-eats"](https://www.youtube.com/watch?v=X4XjYXFRIAQ&start=7084) demo is an example from the NuPIC 2013 Hackathon, using a previous version Numenta's NLP code.
