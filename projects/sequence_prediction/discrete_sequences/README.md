# Description

Run HTM and other competing algorithms on discrete time series prediction problem

This experiment is used in 
Cui Y, Ahmad S, Hawkins J Continuous online sequence learning with an 
unsupervised neural network model. Neural Computation (in press)


# run HTM model
in ./tm/ 
python suite.py -e EXPERIMENTNAME
EXPERIMENTNAME is defined in experiments.cfg (e.g., reber, high-order-distributed-random-perturbed)

# run LSTM model
in ./lstm/ python suite.py -e EXPERIMENTNAME

# run TDNN model
in ./tdnn/ python suite.py -e EXPERIMENTNAME

Experiment results can be visualized with the set of "plot" scripts

For example, plotRepeatedPerturbExperiment.py reproduces the online learning example
in the paper (Fig. 4, Fig. 5)