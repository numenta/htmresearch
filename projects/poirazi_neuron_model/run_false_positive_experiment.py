from neuron_model import HTM_Style_Neuron as Neuron
from generate_data import generate_evenly_distributed_data

import numpy
import random


def run_false_positive_experiment(num_neurons = 1,
						          num_neg_neurons = 1,
						          a = 4000,
						          dim = 16000,
						          num_samples = 1000,
						          num_dendrites = 500,
						          test_dendrite_lengths = range(2, 22, 2),
						          num_replacement = 25,
						          training_iterations = 10000,
						          new_style_training = True,
						          num_trials = 100):
	"""
	General idea:
	Train a new-style neuron on some number of inputs, perhaps 1000.  Accuracy should ideally be 1.0.
	Test this neuron with varying noise levels to see the false negative rate.

	Train new-style neurons on sets of 1000 inputs of varying sparsity, until they predict them perfectly.
	Then throw random inputs at these neurons, and graph the frequency of false positives.
	Expected error should be roughly 1000 times higher, as it could match any of these patterns.

	Issue: we're subsampling heavily.  Patterns have a bits, usually with a = 40, but dendrites
	connect to a maximum of k of them.  This should be taken into account in our equations for expected errors.

	Resulting idea: Attempt to find a new way of demonstrating effect of doing this subsampling.
	Estimate capacity of a neuron, based on the SDR properties and ability to distinguish, and then show that
	actual capacity is comparable to this capacity. 

	dimension ranges:



	"""

	nonlinearity = lambda x: x**2
	for dendrite_length in test_dendrite_lengths:

		fps = []
		fns = []

		for trial in range(num_trials):

			neuron = Neuron(size = dendrite_length*num_dendrites, num_dendrites = num_dendrites, dendrite_length = dendrite_length, dim = dim, nonlinearity = nonlinearity)
			neg_neuron = Neuron(size = dendrite_length*num_dendrites, num_dendrites = num_dendrites, dendrite_length = dendrite_length, dim = dim, nonlinearity = nonlinearity)
			pos, neg = generate_evenly_distributed_data(dim = dim, num_active = a, num_samples = num_samples, num_negatives = num_samples/2)
			labels = [1 for i in range(len(pos))] + [-1 for i in range(len(neg))]
			flipped_labels = [-1 for i in range(len(pos))] + [1 for i in range(len(neg))]
			data = pos + neg

			neuron.HTM_style_initialize_on_data(data, labels)
			neg_neuron.HTM_style_initialize_on_data(data, flipped_labels)

			error, fp, fn = get_error(data, labels, [neuron], [neg_neuron])

			fps.append(fp)
			fns.append(fn)
			print "Error at {} synapses per dendrite is {}, with {} false positives and {} false negatives".format(dendrite_length, error, fp, fn)

		with open("pm_num_dentrites_FP_{}_{}.txt".format(a, dim), "a") as f:
			f.write(str(dendrite_length) + ", " + str(sum(fns + fps)) + ", " + str(num_trials*num_samples) + "\n")


def get_error(data, labels, pos_neurons, neg_neurons = []):
	num_correct = 0
	num_false_positives = 0
	num_false_negatives = 0
	for datapoint, label in zip(data, labels):
		classification = 0
		for neuron in pos_neurons:
			classification += neuron.calculate_activation(datapoint)
		for neuron in neg_neurons:
			classification -= neuron.calculate_activation(datapoint)
		classification = numpy.sign(classification)
		if classification == label:
			num_correct += 1.0
		elif classification == 1:
			num_false_positives += 1
		else:
			num_false_negatives += 1

	return (1.*num_false_positives + num_false_negatives)/len(data), num_false_positives, num_false_negatives
	

if __name__ == "__main__":
	run_false_positive_experiment()
