# Poirazi & Mel Neuron Model

This project contains an implementation of the neuron model from Poirazi & Mel's 2001 paper *Impact of Active Dendrites and Structural Plasticity on the Memory Capacity
of Neural Tissue*, and a number of experiments which apply that neuron model to the in-progress SDR paper.

## Installation

Several external libraries are required, including Numpy and Scikit-learn.
An up-to-date version of nupic is also required, for the sparse matrix in nupic.bindings.math.

## Usage

Scripts for individual experiments are contained in files named run_experimentname.py.  Parameters for experiments should be directly modified within the script.
By default, all experiments write their results to a .txt file.
