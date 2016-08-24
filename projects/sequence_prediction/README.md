# Description

Compare HTM with other algorithms on standard sequence prediction problems

# Dependency
    SciPy
    PyBrain (for LSTM algorithm): https://github.com/pybrain/pybrain
    R (for ARIMA algorithm)
    NuPIC

# Installation

	cd path/to/nupic.research/
	python setup.py develop --user


## On EC2

To get `pybrain` working on EC2, use the following commands to install dependencies:

    sudo yum install atlas-devel blas blas-devel
    ATLAS=/usr/lib64/atlas sudo pip install scipy
