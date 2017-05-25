import random
import numpy
from neuron_model import HTM_Style_Neuron as Neuron
from generate_data import generate_data, generate_evenly_distributed_data

def run_classification_experiment(num_neurons = 1,
								  dim = 40,
								  num_bins = 10,
								  num_samples = 2000,
								  neuron_size = 10000,
								  num_dendrites = 1000,
								  dendrite_length = 10,
								  num_replacement = 25,
								  training_iterations = 10000,
								  new_style_training = True,
								  batch_train = False,
								  batch_size = 10,
								  batch_size_increase = True):
	neuron = Neuron(size = neuron_size, num_dendrites = num_dendrites, dendrite_length = dendrite_length, num_replacement = num_replacement, dim = dim*num_bins)
	pos, neg = generate_data(dim = dim, num_bins = num_bins, num_samples = num_samples)


	data = [(datapoint, 1) for datapoint in pos] + [(datapoint, 0) for datapoint in neg]
	random.shuffle(data)

	shuffled_data, labels = zip(*data)
	inverse_labels = [1 if l == -1 else 1 for l in labels]
	data = shuffled_data

	error_history = []

	if new_style_training:
		neuron.HTM_style_initialize_on_data(data, labels)
		error, fp, fn = get_error(data, labels, neuron)
		print "Pretraining error is {}, with {} false positives and {} false negatives".format(error, fp, fn)
		for i in range(training_iterations):
			for datapoint, label in zip(data, labels):
				neuron.HTM_style_train_on_datapoint(datapoint, label)
			error, fp, fn = get_error(data, labels, neuron)
			error_history.append(error)
			print "Error at iteration {} is {}, with {} false positives and {} false negatives".format(i, error, fp, fn)


	elif batch_train:
		for i in range(training_iterations):
			for b in range(int(numpy.ceil(num_samples/batch_size))):
				neuron.train_on_entire_dataset(data[b*batch_size:], labels[b*batch_size:])
			error, fp, fn = get_error(data, labels, neuron)
			error_history.append(error)
			print "Error at iteration {} is {}, with {} false positives and {} false negatives".format(i, error, fp, fn)
			if batch_size_increase & batch_size < num_samples:
				batch_size = int(batch_size*1.2)

	else:
		for i in range(training_iterations):
			neuron.train_on_entire_dataset(data, labels)
			error, fp, fn = get_error(data, labels, neuron)
			error_history.append(error)
			print "Error at iteration {} is {}, with {} false positives and {} false negatives".format(i, error, fp, fn)

def get_error(data, labels, neuron):
	num_correct = 0
	num_false_positives = 0
	num_false_negatives = 0
	for datapoint, label in zip(data, labels):
		classification = numpy.sign(neuron.calculate_activation(datapoint))
		#print classification
		if classification == label:
			num_correct += 1.0
		elif classification == 1 and label == 0:
			num_false_positives += 1
		else:
			num_false_negatives += 1

	return (1.*num_false_positives + num_false_negatives)/len(data), num_false_positives, num_false_negatives
	

if __name__ == "__main__":
	run_classification_experiment()
