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
- HTM Network models use any of the above encoders, and then run the text through the Network API.
- Cortical.io classify endpoint model is an end-to-end classification tool in their API.

#### Text querying

For a set of text samples, query the model with a phrase to get a sorted print out of the most similar text samples.

	python projects/nlp/imbu_runner.py projects/nlp/data/sample_reviews/sample_reviews.csv


#### Classification experiments

The experiments test the models on classifying labeled samples of text. Here are some results for the Cortical.io word fingerprints model:

(insert plot)

The data here is a small set of survey question responses in data/sample_reviews/sample_reviews.csv. The experiment incrementally increases the training size each trial; we expect to see the classification accuracies increase.


#### Additional examples

Please see the [classification models integration tests](https://github.com/numenta/nupic.research/blob/master/tests/nlp/integration/classification_models_validation_test.py) for example usage.

The ["Fox-eats"](https://www.youtube.com/watch?v=X4XjYXFRIAQ&start=7084) demo is an example from the NuPIC 2013 Hackathon, using a previous version of this repo.
