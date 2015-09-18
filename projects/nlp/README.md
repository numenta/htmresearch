# Fluent - Language with NuPIC

[![Build Status](https://travis-ci.org/numenta/nupic.fluent.svg?branch=master)](https://travis-ci.org/numenta/nupic.fluent) [![Coverage Status](https://coveralls.io/repos/numenta/nupic.fluent/badge.png?branch=master)](https://coveralls.io/r/numenta/nupic.fluent?branch=master)

A platform for building language / NLP-based applications using [NuPIC](https://github.com/numenta/nupic) and [Cortical.io's API](http://www.cortical.io/developers.html). The current version is v0.2.

###NOTE this repo contains experimental code. A few disclaimers:

- the contents may
    - change without warning or explanation
    - change quickly, or not at all
    - not function properly
    - be buggy and sloppy; don't judge :)
- work with external partners will not be included here
- we might decide at some point to not do our NLP research in the open anymore and instead delete the whole repository

The motivation here is we would like to move quickly in research while maintaining transparency.

## Installation

Requirements:

- [NuPIC](https://github.com/numenta/nupic)
- [NuPIC.research/classification](https://github.com/numenta/nupic.research/tree/master/classification); this repo does not have a setup.py, so it should be included on your Python path (see command below).
- [cortipy](https://github.com/numenta/cortipy)
- Python modules [enum](https://pypi.python.org/pypi/enum34), [pandas](http://pandas.pydata.org/), and [numpy](http://www.numpy.org/); included in the setup script below.

You must have a valid REST API key from [Cortical.io](http://www.cortical.io/developers.html).

To install, run:

    python setup.py install

Then, set up the following environment variables with your Cortical.io REST API credentials:

    export CORTICAL_API_KEY=api_key

Add nupic.research/classification to your Python path:

	export PYTHONPATH=$PYTHONPATH:/<path_to_nupic.research/classification>

## Usage

### Examples (under development)

Please see the [classification models integration tests](https://github.com/numenta/nupic.fluent/blob/master/tests/integration/classification_models_validation_test.py) for example usage.

The ["Fox-eats"](https://www.youtube.com/watch?v=X4XjYXFRIAQ&start=7084) demo is an example from the NuPIC 2013 Hackathon, using a previous version of this repo.

## Advanced

### Using a different Cortical.io retina

If you want to use a different Cortical.io retina, you'll have to specify when instantiating the `CioEncoder`.
