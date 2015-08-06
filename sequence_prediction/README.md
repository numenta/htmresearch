# Description

Compare HTM with other algorithms on standard sequence prediction problems

# Dependency
    SciPy
    PyBrain (for LSTM algorithm): https://github.com/pybrain/pybrain
    NuPIC

# Installation

	cd path/to/nupic.research/sequence_prediction
	python setup.py develop

Or (doesn't require sudo but does require manual PYTHONPATH setup):

	python setup.py develop --prefix=/some/other/path/

# Example Usage

run swarm on continous time series prediction problems

	python run_swarm.py -d DATASET

run swarm using the delta encoder
	python run_swarm.py -d DATASET -f True

Existing DATASET includes sine, MackeyGlass, SantaFe_A

compare TM performance with a trivial shift predictor
	python comparePerformance.py -d DATASET
