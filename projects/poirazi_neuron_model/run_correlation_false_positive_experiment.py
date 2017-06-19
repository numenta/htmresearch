# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy
import random
from htmresearch.frameworks.poirazi_neuron_model.neuron_model import (
  power_nonlinearity, threshold_nonlinearity, sigmoid_nonlinearity)
from htmresearch.frameworks.poirazi_neuron_model.neuron_model import Matrix_Neuron as Neuron
from htmresearch.frameworks.poirazi_neuron_model.data_tools import generate_correlated_data_sparse, generate_evenly_distributed_data_sparse, covariance_generate_correlated_data
from nupic.bindings.math import *
numpy.random.seed(19)

def run_false_positive_experiment_correlation(num_neurons = 1,
                    a = 32,
                    possible_cluster_sizes = [1,2,4,8,16],
                    dim = 2000,
                    num_samples = 1000,
                    num_dendrites = 500,
                    dendrite_length = 10,
                    num_trials = 1000,
                    nonlinearity = threshold_nonlinearity(5)):
  """
  Run an experiment to test the false positive rate based on number of
  synapses per dendrite, dimension and sparsity.  Uses two competing neurons,
  along the P&M model.

  Based on figure 5B in the original SDR paper.
  """
  for trial in range(num_trials):

    fps = []
    fns = []

    neuron = Neuron(size = dendrite_length*num_dendrites, num_dendrites = num_dendrites, dendrite_length = dendrite_length, dim = dim, nonlinearity = nonlinearity)

    cluster_size = numpy.random.choice(possible_cluster_sizes)
    num_clusters = numpy.random.randint(0, int(dim/cluster_size))
    data = generate_correlated_data_sparse(dim = dim, num_active = a, num_samples = num_samples, num_clusters = num_clusters, cluster_size = cluster_size)
    labels = numpy.asarray([1 for i in range(num_samples/2)] + [-1 for i in range(num_samples/2)])

    patterns = [data.rowNonZeros(i)[0] for i in range(data.nRows())]
    pattern_correlations = []
    correlations = numpy.corrcoef(data.toDense(), rowvar = False)
    for i, pattern in enumerate(patterns):
      pattern_correlation = [correlations[i, j] for i in pattern for j in pattern if i != j]
      pattern_correlations.append(pattern_correlation)
    correlation = numpy.mean(pattern_correlations)

    neuron.HTM_style_initialize_on_data(data, labels)
    error, fp, fn = get_error(data, labels, [neuron])
    fps.append(fp)
    fns.append(fn)
    print "Error at r = {}, with {} clusters of size {} is {}, with {} false positives and {} false negatives".format(correlation, num_clusters, cluster_size, error, fp, fn)

    with open("correlation_results_{}.txt".format(a), "a") as f:
      f.write(str(correlation) + ", " + str(sum(fps)) + ", " + str(num_trials*num_samples/2) + "\n")

def get_error(data, labels, pos_neurons, neg_neurons = [], add_noise = False):
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
  for classification, label in zip(classifications, labels):
    classification = numpy.sign(classification)
    if classification >= 1 and label >= 1:
      num_correct += 1.0
    elif classification <= 0 and label <= 0:
      num_correct += 1.
    elif classification >= 1. and label <= 0:
      num_false_positives += 1
    elif classification < 0 and label >= 1:
      num_false_negatives += 1
  return (1.*num_false_positives + num_false_negatives)/data.nRows(), num_false_positives, num_false_negatives


if __name__ == "__main__":
  run_false_positive_experiment_correlation()
