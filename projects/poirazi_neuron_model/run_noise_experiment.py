from neuron_model import HTM_Style_Neuron as Neuron
from generate_data import generate_evenly_distributed_data

import numpy
import random
import copy



def apply_noise(data, noise):
	for x, datapoint in enumerate(data):
		indices = [i for i, d in enumerate(datapoint) if d == 1]
		replace_indices = numpy.random.choice(indices, size=int(1.0*len(indices)*noise/100), replace = False)

		#print len(replace_indices)

		for index in replace_indices:
			datapoint[index] = 0

		new_indices = numpy.random.choice(len(datapoint), size =int(1.0*len(indices)*noise/100), replace = False)

		for i in new_indices:
			while datapoint[i] == 1:
				i = numpy.random.randint(0, len(datapoint))
			datapoint[i] = 1

		data[x] = datapoint

	return data



def run_noise_experiment(num_neurons = 1,
						 nun_neg_neurons = 1,
						 a = 128,
						 dim = 6000,
						 #test_dims = [500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000],
						 test_noise_levels = range(15, 100, 5),
						 num_samples = 1000,
						 num_dendrites = 500,
						 dendrite_length = 30,
						 num_replacement = 25,
						 training_iterations = 10000,
						 new_style_training = True,
						 theta = 8,
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

	nonlinearity = lambda x: 1 if x > theta else 0
	for noise in test_noise_levels:

		fps = []
		fns = []

		for trial in range(num_trials):

			successful_initialization = False
			while not successful_initialization:
				neuron = Neuron(size = dendrite_length*num_dendrites, num_dendrites = num_dendrites, dendrite_length = dendrite_length, dim = dim, nonlinearity = nonlinearity)
				pos, neg = generate_evenly_distributed_data(dim = dim, num_active = a, num_samples = num_samples, num_negatives = 0)
				labels = [1 for i in range(len(pos))] + [0 for i in range(len(neg))]
				data = pos + neg

				neuron.HTM_style_initialize_on_data(data, labels)

				error, fp, fn = get_error(data, labels, neuron)
				print "Initialization error is {}, with {} false positives and {} false negatives".format(error, fp, fn)
				if error == 0:
					successful_initialization = True
				else:
					print "Repeating to get a successful initialization"

			data = apply_noise(data, noise)
			error, fp, fn = get_error(data, labels, neuron)
			fps.append(fp)
			fns.append(fn)
			print "Error at noise {} is {}, with {} false positives and {} false negatives".format(noise, error, fp, fn)

		with open("noise_FN_{}.txt".format(theta), "a") as f:
			f.write(str(noise) + ", " + str(sum(fns + fps)) + ", " + str(num_trials*num_samples) + "\n")



def run_pm_style_noise_experiment(num_neurons = 1,
						 num_neg_neurons = 1,
						 a = 128,
						 dim = 6000,
						 #test_dims = [500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000],
						 test_noise_levels = range(15, 100, 5),
						 num_samples = 1000,
						 num_dendrites = 500,
						 dendrite_length = 30,
						 num_replacement = 25,
						 training_iterations = 10000,
						 new_style_training = True,
						 theta = 16,
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
	for noise in test_noise_levels:

		fps = []
		fns = []

		for trial in range(num_trials):

			successful_initialization = False
			while not successful_initialization:
				neuron = Neuron(size = dendrite_length*num_dendrites, num_dendrites = num_dendrites, dendrite_length = dendrite_length, dim = dim, nonlinearity = nonlinearity)
				neg_neuron = Neuron(size = dendrite_length*num_dendrites, num_dendrites = num_dendrites, dendrite_length = dendrite_length, dim = dim, nonlinearity = nonlinearity)
				pos, neg = generate_evenly_distributed_data(dim = dim, num_active = a, num_samples = num_samples, num_negatives = num_samples/2)
				labels = [1 for i in range(len(pos))] + [-1 for i in range(len(neg))]
				flipped_labels = [-1 for i in range(len(pos))] + [1 for i in range(len(neg))]
				data = pos + neg

				neuron.HTM_style_initialize_on_data(data, labels)
				neg_neuron.HTM_style_initialize_on_data(data, flipped_labels)

				error, fp, fn = get_error(data, labels, [neuron], [neg_neuron])
				print "Initialization error is {}, with {} false positives and {} false negatives".format(error, fp, fn)
				if error == 0:
					successful_initialization = True
				else:
					print "Repeating to get a successful initialization"

			data = apply_noise(data, noise)
			error, fp, fn = get_error(data, labels, [neuron], [neg_neuron])
			fps.append(fp)
			fns.append(fn)
			print "Error at noise {} is {}, with {} false positives and {} false negatives".format(noise, error, fp, fn)

		with open("pm_noise_FN_{}.txt".format(theta), "a") as f:
			f.write(str(noise) + ", " + str(sum(fns)) + ", " + str(num_trials*num_samples) + "\n")


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
	run_pm_style_noise_experiment()
