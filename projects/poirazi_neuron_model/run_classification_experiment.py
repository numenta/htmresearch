import random
import numpy
from neuron_model import Neuron, Matrix_Neuron, threshold_nonlinearity, power_nonlinearity
from generate_data import generate_data, generate_evenly_distributed_data_sparse
from nupic.bindings.math import SM32
from collections import Counter

def shuffle_sparse_matrix_and_labels(matrix, labels):
    print "Shuffling data"
    new_matrix = matrix.toDense()
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(new_matrix)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(labels)

    return SM32(new_matrix), numpy.asarray(labels)

def split_sparse_matrix(matrix, num_categories):
	if matrix.nRows() < num_categories:
		return [matrix.getSlice(i, i+1, 0, matrix.nCols()) for i in range(matrix.nRows())] + [SM32() for i in range(num_categories - matrix.nRows())]
	else:
		inc = matrix.nRows()/num_categories
		divisions = [matrix.getSlice(i*inc, (i+1)*inc, 0, matrix.nCols()) for i in range(num_categories - 1)]

		# Handle the last bin separately.  All overflow goes into it.
		divisions.append(matrix.getSlice((num_categories - 1)*inc, matrix.nRows(), 0, matrix.nCols()))

		return divisions	

def run_HTM_classification_experiment(num_neurons = 50,
								      dim = 40,
								      num_bins = 10,
								      num_samples = 50*600,
								      neuron_size = 9600,
								      num_dendrites = 600,
								      dendrite_length = 16,
								      num_iters = 1000,
								      nonlinearity = power_nonlinearity(3)
								      ):
	"""
	Runs an experiment testing classifying a binary dataset, based on Poirazi &
	Mel's original experiment.  Learning is using our modified variant of their
	rule, and positive and negative neurons compete to classify a datapoint.

	Performance has historically been poor, noticeably worse than what is
	achieved with only a single neuron using an HTM-style learning rule on
	datasets of similar size.  It is suspected that the simplifications made
	to the P&M learning rule are having a negative effect.

	Furthermore, P&M report that they are willing to train for an exceptional
	amount of time, up to 96,000 iterations per neuron.  We have never even
	begun to approach this long a training time, so it is possible that our
	performance would converge with theirs given more time.
	"""

	
	pos_neurons = [Matrix_Neuron(size = neuron_size, num_dendrites = num_dendrites, dendrite_length = dendrite_length, nonlinearity = nonlinearity, dim = dim*num_bins) for i in range(num_neurons/2)]
	neg_neurons = [Matrix_Neuron(size = neuron_size, num_dendrites = num_dendrites, dendrite_length = dendrite_length, nonlinearity = nonlinearity, dim = dim*num_bins) for i in range(num_neurons/2)]
	pos, neg = generate_evenly_distributed_data_sparse(dim = 400, num_active = 40, num_samples = num_samples/2), generate_evenly_distributed_data_sparse(dim = 400, num_active = 40, num_samples = num_samples/2)
	#pos, neg = generate_data(dim = dim, num_bins = num_bins, num_samples = num_samples, sparse = True)
	
	if (pos.nRows() > num_dendrites*len(pos_neurons)):
		print "Too much data to have unique dendrites for positive neurons, clustering"
		pos = pos.toDense()
		model = AgglomerativeClustering(n_clusters = len(pos_neurons), affinity = "manhattan", linkage = "average")
		clusters = model.fit_predict(pos)
		neuron_data = [SM32() for i in range(len(pos_neurons))]
		for datapoint, cluster in zip(pos, clusters):
			neuron_data[cluster].append(SM32([datapoint]))
		for i, neuron in enumerate(pos_neurons):
			neuron.HTM_style_initialize_on_data(neuron_data[i], [1 for i in range(neuron_data[i].nRows())])
		pos = SM32(pos)
	else:
		print "Directly initializing positive neurons with unique dendrites"
		neuron_data = split_sparse_matrix(pos, len(pos_neurons))
		for neuron, data in zip(pos_neurons, neuron_data):
			neuron.HTM_style_initialize_on_data(data, [1 for i in range(data.nRows())])


	if (neg.nRows() > num_dendrites*len(neg_neurons)):
		print "Too much data to have unique dendrites for negative neurons, clustering"
		neg = neg.toDense()
		model = AgglomerativeClustering(n_clusters = len(neg_neurons), affinity = "manhattan", linkage = "average")
		clusters = model.fit_predict(neg)
		neuron_data = [SM32() for i in range(len(neg_neurons))]
		for datapoint, cluster in zip(neg, clusters):
			neuron_data[cluster].append(SM32([datapoint]))
		for i, neuron in enumerate(neg_neurons):
			neuron.HTM_style_initialize_on_data(neuron_data[i], [1 for i in range(neuron_data[i].nRows())])
		neg = SM32(neg)
	else:
		print "Directly initializing negative neurons with unique dendrites"
		neuron_data = split_sparse_matrix(neg, len(neg_neurons))
		for neuron, data in zip(neg_neurons, neuron_data):
			neuron.HTM_style_initialize_on_data(data, [1 for i in range(data.nRows())])


	labels = [1 for i in range(pos.nRows())] + [-1 for i in range(neg.nRows())]
	data = pos
	data.append(neg)
	data, labels = shuffle_sparse_matrix_and_labels(data, labels)

	error, fp, fn = get_error(data, labels, pos_neurons, neg_neurons)
	print "Error at initialization is {}, with {} false positives and {} false negatives".format(error, fp, fn)
	for iter in range(num_iters):
		for neuron in pos_neurons:
			neuron.HTM_style_train_on_data(data, labels)
		for neuron in neg_neurons:
			neuron.HTM_style_train_on_data(data, (labels * -1.))

		error, fp, fn = get_error(data, labels, pos_neurons, neg_neurons)
		print "Error at iter {} is {}, with {} false positives and {} false negatives".format(iter, error, fp, fn)
		with open("classification_experiment.txt", "a") as f:
			f.write(str(iter) + ", " + str(error) + "\n")


def get_error(data, labels, pos_neurons, neg_neurons = []):
	"""
	Calculates error, including number of false positives and false negatives.

	Written to allow the use of multiple neurons, in case we attempt to use a
	population in the future.

	"""
	num_correct = 0
	num_false_positives = 0
	num_false_negatives = 0
	classifications = numpy.zeros(data.nRows())
	for neuron in pos_neurons:
		classifications += neuron.calculate_on_entire_dataset(data)
	for neuron in neg_neurons:
		classifications -= neuron.calculate_on_entire_dataset(data)
	classifications = numpy.sign(classifications)
	for classification, label in zip(classifications, labels):
		if classification > 0 and label > 0:
			num_correct += 1.0
		elif classification <= 0 and label <= 0:
			num_correct += 1.0
		elif classification == 1. and label < 0:
			num_false_positives += 1
		else:
			num_false_negatives += 1
	return (1.*num_false_positives + num_false_negatives)/data.nRows(), num_false_positives, num_false_negatives


def run_classification_experiment(num_neurons = 50,
								  dim = 40,
								  num_bins = 10,
								  num_samples = 30000,
								  neuron_size = 10000,
								  num_dendrites = 250,
								  dendrite_length = 40,
								  nonlinearity = lambda x: x**2,
								  nonlinearity_derivative = lambda x: 2*x,
								  num_replacement = 25,
								  training_iterations = 10000,
								  use_temperature = True,
								  batch_train = False,
								  batch_size = 10,
								  batch_size_increase = True):
	"""
	Runs an experiment testing classifying a binary dataset, based on Poirazi &
	Mel's original experiment.  Learning is using our modified variant of their
	rule, and positive and negative neurons compete to classify a datapoint.

	Performance has historically been poor, noticeably worse than what is
	achieved with only a single neuron using an HTM-style learning rule on
	datasets of similar size.  It is suspected that the simplifications made
	to the P&M learning rule are having a negative effect.

	Furthermore, P&M report that they are willing to train for an exceptional
	amount of time, up to 96,000 iterations per neuron.  We have never even
	begun to approach this long a training time, so it is possible that our
	performance would converge with theirs given more time.
	"""
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
			error = get_error_traditional(data, labels, pos_neurons, neg_neurons)
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
			error = get_error_traditional(data, labels, pos_neurons, neg_neurons)
			(error_history[-1]).append(error)
			print "Error at iteration {} is {}".format(i, error)
			if use_temperature:
				error_history, temperature = update_temperature(temperature, error_history)

def update_temperature(temperature, error_history):
	"""
	Updates the temperature for P&M's learning rule, based on appendix 4 of the
	paper. 
	"""
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



def get_error_traditional(data, labels, pos_neurons, neg_neurons):
	"""
	Calculates error, including number of false positives and false negatives.

	"""
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
	run_HTM_classification_experiment()
