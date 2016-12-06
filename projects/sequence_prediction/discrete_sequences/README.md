# Description

Run HTM and other competing algorithms on discrete sequence prediction problem

This experiment is used in:
 
Cui Y, Ahmad S, Hawkins J. Continuous online sequence learning with an unsupervised neural network model. (2016) *Neural Computation,* 28(11) 2474-2504.  DOI: 10.1162/NECO_a_00893


# run HTM model
	cd ./tm/ 
	python suite.py -e EXPERIMENT_NAME
	
EXPERIMENT_NAME is defined in experiments.cfg (e.g., reber, )
Below is a list of experiment.

* **high-order-distributed-random-perturbed**. High-order sequence prediction task. We switch to a new dataset after 10000 to evalulate ho fast the algorithms adapt to changes. (Figure 4, 5)
* **high-order-distributed-random-multiple-predictions**. High-order sequence prediction with multiple possible endings (Figure 6)
* **high-order-variable-length**. High order sequence prediction with variable lengths. (Figure 7). This task evalulates how fast does HTM learn very long sequences.
* **high-order-noise**. This task evaluates sensitivity to temporal noises (Figure 8)
* **high-order-distributed-random-kill-cell**. This task evaluates fault tolerance (Figure 9). A fraction of the cells are removed from the network after it learns the sequence. 
* **reber** The Reber Grammar dataset (not shown in the paper). 

The experiment names are the same for LSTM, ELM and TDNN models.

# run LSTM model
	cd ./lstm/ 
	python suite.py -e EXPERIMENT_NAME

# run TDNN model
	cd ./tdnn/ 
	python suite.py -e EXPERIMENT_NAME

Experiment results can be visualized with the set of "plot" scripts

For example, plotRepeatedPerturbExperiment.py reproduces the online learning example
in the paper (Fig. 4, Fig. 5)