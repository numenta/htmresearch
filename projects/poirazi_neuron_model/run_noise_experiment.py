from neuron_model import Matrix_Neuron as Neuron
from neuron_model import threshold_nonlinearity, power_nonlinearity
from generate_data import generate_evenly_distributed_data_sparse
from nupic.bindings.math import *

import numpy
import random
import copy

def apply_noise(data, noise):
    """
    Applies noise to a sparse matrix.  Noise should be an integer between 0 and
    100, indicating the percentage of ones in the original input to move.
    The input matrix is modified in-place, and nothing is returned.
    This operation does not affect the sparsity of the matrix, or of any
    individual datapoint.
    """

    for i in range(data.nRows()):
        ones = data.rowNonZeros(i)[0]
        replace_indices = numpy.random.choice(ones, size=int(1.0*len(ones)*noise/100), replace = False)
        for index in replace_indices:
            data[i, index] = 0

        new_indices = numpy.random.choice(data.nCols(), size = int(1.0*len(ones)*noise/100.), replace = False)

        for index in new_indices:
            while data[i, index] == 1:
                index = numpy.random.randint(0, data.nCols())
            data[i, index] = 1




def run_noise_experiment(num_neurons = 1,
                         a = 128,
                         dim = 6000,
                         test_noise_levels = range(15, 100, 5),
                         num_samples = 500,
                         num_dendrites = 500,
                         dendrite_length = 30,
                         theta = 16,
                         num_trials = 100):
    """
    Tests the impact of noise on a neuron, using an HTM approach to a P&M
    model of a neuron.  Nonlinearity is a simple threshold at theta, as in the
    original version of this experiment, and each dendrite is bound by the 
    initialization to a single pattern.  Only one neuron is used, unlike in the
    P&M classification experiment, and a successful identification is simply
    defined as at least one dendrite having theta active synapses.

    Training is done via HTM-style initialization.  In the event that the init
    fails to produce an error rate of 0 without noise (which anecdotally never
    occurs), we simple reinitialize.

    Results are saved to the file noise_FN_{theta}.txt.
    """

    nonlinearity = threshold_nonlinearity(theta)
    for noise in test_noise_levels:

        fps = []
        fns = []

        for trial in range(num_trials):

            successful_initialization = False
            while not successful_initialization:
                neuron = Neuron(size = dendrite_length*num_dendrites, num_dendrites = num_dendrites, dendrite_length = dendrite_length, dim = dim, nonlinearity = nonlinearity)
                data = generate_evenly_distributed_data_sparse(dim = dim, num_active = a, num_samples = num_samples)
                labels = [1 for i in range(num_samples)]

                neuron.HTM_style_initialize_on_data(data, labels)

                error, fp, fn = get_error(data, labels, [neuron])
                print "Initialization error is {}, with {} false positives and {} false negatives".format(error, fp, fn)
                if error == 0:
                    successful_initialization = True
                else:
                    print "Repeating to get a successful initialization"

            apply_noise(data, noise)
            error, fp, fn = get_error(data, labels, [neuron])
            fps.append(fp)
            fns.append(fn)
            print "Error at noise {} is {}, with {} false positives and {} false negatives".format(noise, error, fp, fn)

        with open("noise_FN_{}.txt".format(theta), "a") as f:
            f.write(str(noise) + ", " + str(numpy.sum(fns)) + ", " + str(num_trials*num_samples) + "\n")



def run_pm_style_noise_experiment(num_neurons = 1,
                         num_neg_neurons = 1,
                         a = 128,
                         dim = 6000,
                         test_noise_levels = range(30, 100, 5),
                         num_samples = 1000,
                         num_dendrites = 500,
                         dendrite_length = 30,
                         num_replacement = 25,
                         training_iterations = 10000,
                         new_style_training = True,
                         num_trials = 200,
                         nonlinearity = power_nonlinearity(2)):
    """
    A version of the above test, but using two P&M style competing neurons with
    nonlinearity l(x) = x^2.  As the test is now symmetric, with a false
    positive for one neuron being a false negative for the other, we only track
    overall error.  There is no "theta", as identification is merely dependent
    on which neuron in the pair responds more strongly.  This is most likely
    not biologically plausible, especially at high noise values.

    Training is done via HTM-style initialization.  In the event that the init
    fails to produce an error rate of 0 without noise (which anecdotally never
    occurs), we simple reinitialize.

    Results are saved to the file pm_noise_FN.txt.
    """
    for noise in test_noise_levels:

        fps = []
        fns = []

        for trial in range(num_trials):

            successful_initialization = False
            while not successful_initialization:
                neuron = Neuron(size = dendrite_length*num_dendrites, num_dendrites = num_dendrites, dendrite_length = dendrite_length, dim = dim, nonlinearity = nonlinearity)
                neg_neuron = Neuron(size = dendrite_length*num_dendrites, num_dendrites = num_dendrites, dendrite_length = dendrite_length, dim = dim, nonlinearity = nonlinearity)
                data = generate_evenly_distributed_data_sparse(dim = dim, num_active = a, num_samples = num_samples)
                labels = numpy.asarray([1 for i in range(num_samples/2)] + [-1 for i in range(num_samples/2)])
                flipped_labels = labels * -1

                neuron.HTM_style_initialize_on_data(data, labels)
                neg_neuron.HTM_style_initialize_on_data(data, flipped_labels)

                error, fp, fn = get_error(data, labels, [neuron], [neg_neuron])
                print "Initialization error is {}, with {} false positives and {} false negatives".format(error, fp, fn)
                if error == 0:
                    successful_initialization = True
                else:
                    print "Repeating to get a successful initialization"

            apply_noise(data, noise)
            error, fp, fn = get_error(data, labels, [neuron], [neg_neuron])
            fps.append(fp)
            fns.append(fn)
            print "Error at noise {} is {}, with {} false positives and {} false negatives".format(noise, error, fp, fn)

        with open("pm_noise_FN.txt", "a") as f:
            f.write(str(noise) + ", " + str(sum(fns)) + ", " + str(num_trials*num_samples) + "\n")


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
    

if __name__ == "__main__":
    run_pm_style_noise_experiment()
