import random
import numpy
from neuron_model import Neuron
from generate_data import generate_data

def run_classification_experiment(num_neurons = 50,
								  dim = 40,
								  num_bins = 10,
								  num_samples = 1000,
								  neuron_size = 10000,
								  num_dendrites = 400,
								  dendrite_length = 25,
								  nonlinearity = lambda x: x**2,
								  nonlinearity_derivative = lambda x: 2*x,
								  num_replacement = 25,
								  training_iterations = 10000,
								  use_temperature = True,
								  batch_train = False,
								  batch_size = 10,
								  batch_size_increase = True):
	pos_neurons = [Neuron(size = neuron_size, num_dendrites = num_dendrites, dendrite_length = dendrite_length, nonlinearity = nonlinearity, nonlinearity_derivative = nonlinearity_derivative, num_replacement = num_replacement, dim = dim*num_bins) for i in range(num_neurons/2)]
	neg_neurons = [Neuron(size = neuron_size, num_dendrites = num_dendrites, dendrite_length = dendrite_length, nonlinearity = nonlinearity, nonlinearity_derivative = nonlinearity_derivative, num_replacement = num_replacement, dim = dim*num_bins) for i in range(num_neurons/2)]
	pos, neg = generate_data(dim = dim, num_bins = num_bins, num_samples = num_samples)

	data = [(datapoint, 1) for datapoint in pos] + [(datapoint, -1) for datapoint in neg]
	random.shuffle(data)

	shuffled_data, labels = zip(*data)
	inverse_labels = [1 if l == -1 else 1 for l in labels]
	data = shuffled_data

	error_history = [[]]
	if use_temperature:
		temperature = 0.9
	else:
		temperature = 0

	if batch_train:
		for i in range(training_iterations):
			for b in range(num_samples/batch_size):
				for neuron in pos_neurons:
					neuron.train_on_entire_dataset(data[b*batch_size:], labels[b*batch_size:], temperature)
				for neuron in neg_neurons:
					neuron.train_on_entire_dataset(data[b*batch_size:], inverse_labels[b*batch_size:], temperature)
			error = get_error(data, labels, pos_neurons, neg_neurons)
			(error_history[-1]).append(error)
			print "Error at iteration {} is {}".format(i, error)
			if batch_size_increase & batch_size < num_samples:
				batch_size*= 1.2
			if use_temperature:
				error_history, temperature = update_temperature(temperature, error_history)

	else:
		for i in range(training_iterations):
			for neuron in pos_neurons:
				neuron.train_on_entire_dataset(data, labels, temperature)
			for neuron in neg_neurons:
				neuron.train_on_entire_dataset(data, inverse_labels, temperature)
			error = get_error(data, labels, pos_neurons, neg_neurons)
			(error_history[-1]).append(error)
			print "Error at iteration {} is {}".format(i, error)
			if use_temperature:
				error_history, temperature = update_temperature(temperature, error_history)

def update_temperature(temperature, error_history):
	if len(error_history[-1]) >= 800:
		error_history.append([])
		return error_history, temperature*0.9

	min_error = 1
	new_min_counter = 0
	for error in error_history[-1]:
		if error < min_error:
			min_error = error
			new_min_counter += 1
	if new_min_counter > 180:
		error_history.append([])
		return error_history, temperature*0.9

	if error_history.count(min(error_history[-1])) >= 100:
		error_history.append([])
		return error_history, temperature/(0.9 ** 3)

	elif temperature < 0.1:
		error_history.append([])
		return error_history, temperature/(0.9 ** 3)

	else:
		return error_history, temperature



def get_error(data, labels, pos_neurons, neg_neurons):
	num_correct = 0
	num_incorrect = 0
	for datapoint, label in zip(data, labels):
		classification = numpy.sign(numpy.sum([n.calculate_activation(datapoint) for n in pos_neurons]) - numpy.sum([n.calculate_activation(datapoint) for n in neg_neurons]))
		if classification == label:
			num_correct += 1.0
		else:
			num_incorrect += 1.0

	return num_incorrect/len(data)

if __name__ == "__main__":
	run_classification_experiment()
