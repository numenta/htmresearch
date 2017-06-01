from neuron_model import power_nonlinearity, threshold_nonlinearity
from neuron_model import Matrix_Neuron as Neuron
from data_tools import generate_evenly_distributed_data_sparse

import numpy
import random

def run_false_positive_experiment_dim(num_neurons = 1,
                                      num_neg_neurons = 1,
                                      a = 64,
                                      test_dims = range(500, 700, 200),
                                      num_samples = 1000,
                                      num_dendrites = 500,
                                      dendrite_length = 24,
                                      num_trials = 10000,
                                      nonlinearity = threshold_nonlinearity(12)):
    """
    Run an experiment to test the false positive rate based on number of
    synapses per dendrite, dimension and sparsity.  Uses two competing neurons,
    along the P&M model, with nonlinearity l(x) = x^2.  

    Based on figure 5B in the original SDR paper.
    """
    for dim in test_dims:

        fps = []
        fns = []

        for trial in range(num_trials):

            neuron = Neuron(size = dendrite_length*num_dendrites, num_dendrites = num_dendrites, dendrite_length = dendrite_length, dim = dim, nonlinearity = nonlinearity)
            neg_neuron = Neuron(size = dendrite_length*num_dendrites, num_dendrites = num_dendrites, dendrite_length = dendrite_length, dim = dim, nonlinearity = nonlinearity)
            data = generate_evenly_distributed_data_sparse(dim = dim, num_active = a, num_samples = num_samples)
            labels = numpy.asarray([1 for i in range(num_samples/2)] + [-1 for i in range(num_samples/2)])
            flipped_labels = labels * -1

            neuron.HTM_style_initialize_on_data(data, labels)
            neg_neuron.HTM_style_initialize_on_data(data, flipped_labels)

            error, fp, fn = get_error_matrix(data, labels, [neuron], [neg_neuron])

            fps.append(fp)
            fns.append(fn)
            print "Error at n = {} is {}, with {} false positives and {} false negatives".format(dim, error, fp, fn)

        with open("pm_dim_FP_{}.txt".format(a), "a") as f:
            f.write(str(dendrite_length) + ", " + str(sum(fns + fps)) + ", " + str(num_trials*num_samples) + "\n")
    
def get_error_matrix(data, labels, pos_neurons, neg_neurons = []):
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
        if classification >= 1 and label >= 1:
            num_correct += 1.0
        elif classification <= 0 and label <= 0:
        	num_correct += 1.
        elif classification >= 1.:
            num_false_positives += 1
        else:
            num_false_negatives += 1
    return (1.*num_false_positives + num_false_negatives)/data.nRows(), num_false_positives, num_false_negatives


if __name__ == "__main__":
    run_false_positive_experiment_dim()
