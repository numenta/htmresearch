# Language with NuPIC

This directory contains scripts for running NLP-based applications using [NuPIC](https://github.com/numenta/nupic) and [Cortical.io's API](http://www.cortical.io/developers.html).

Additional requirements:

- [cortipy](https://github.com/numenta/cortipy)
- Python modules [enum](https://pypi.python.org/pypi/enum34), [pandas](http://pandas.pydata.org/), and [numpy](http://www.numpy.org/).
- You must have a valid REST API key from [Cortical.io](http://www.cortical.io/developers.html).

## Use cases
In this directory are scripts for some NLP experiments. They each use one or more of the following models:

- Keywords model gives a random (seeded) encoding to each word.
- Cortical.io word fingerprints model encodes each word with their API, and then makes a union representation over the words in a block of text.
- Cortical.io document fingerprints model uses their API to encode a block of text.
- HTM Network models use the Cortical.io word-level encoder, and then run the patterns through the Network API.
- Cortical.io classify endpoint model is an end-to-end classification tool in their API.

#### Text querying

Please refer to the [Imbu project directory](https://github.com/numenta/nupic.research/tree/master/projects/imbu).

#### Classification examples

See the [hello_classification.py](https://github.com/numenta/nupic.research/blob/master/projects/nlp/hello_classification.py) and [simple_labels.py](https://github.com/numenta/nupic.research/blob/master/projects/nlp/simple_labels.py) scripts.
	


#### Additional examples

The ["Fox-eats"](https://www.youtube.com/watch?v=X4XjYXFRIAQ&start=7084) demo is an example from the NuPIC 2013 Hackathon, using a previous version Numenta's NLP code.
