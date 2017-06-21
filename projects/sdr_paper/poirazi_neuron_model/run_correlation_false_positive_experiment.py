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
from htmresearch.frameworks.poirazi_neuron_model.data_tools import (
  generate_correlated_data_sparse, generate_evenly_distributed_data_sparse,
  covariance_generate_correlated_data, generate_correlated_data_clusters)
from nupic.bindings.math import *
from multiprocessing import Pool, cpu_count


def run_false_positive_experiment_correlation(seed,
                                              num_neurons = 1,
                                              a = 32,
                                              dim = 2000,
                                              num_samples = 20000,
                                              num_dendrites = 500,
                                              dendrite_length = 20,
                                              num_trials = 1000,
                                              nonlinearity = threshold_nonlinearity(10)):
  """
  Run an experiment to test the false positive rate based on number of
  synapses per dendrite, dimension and sparsity.  Uses two competing neurons,
  along the P&M model.

  Based on figure 5B in the original SDR paper.
  """
  numpy.random.seed(seed)
  possible_cluster_sizes = range(2, 10)


  for trial in range(num_trials):
    num_cluster_sizes = numpy.random.choice([1, 1, 2] + range(1, 8), 1)
    cluster_sizes = numpy.random.choice(possible_cluster_sizes, num_cluster_sizes, replace = False)
    num_cells_per_cluster_size = [numpy.random.randint(dim, 3*dim) for i in range(num_cluster_sizes)]
    data = generate_correlated_data_clusters(dim = dim,
                                             num_active = a,
                                             num_samples = num_samples,
                                             num_cells_per_cluster_size =
                                                 num_cells_per_cluster_size,
                                             cluster_sizes = cluster_sizes)
    patterns = [data.rowNonZeros(i)[0] for i in range(data.nRows())]
    pattern_correlations = []
    correlations = numpy.corrcoef(data.toDense(), rowvar = False)
    for i, pattern in enumerate(patterns):
      pattern_correlation = [correlations[i, j] for i in pattern for j in pattern if i != j]
      pattern_correlations.append(pattern_correlation)
    correlation = numpy.mean(pattern_correlations)
    print "Generated {} samples with total average pattern correlation {}, using cluster sizes {} with cells per cluster size of {}".format(num_samples, correlation, cluster_sizes, num_cells_per_cluster_size)


    fps = []
    fns = []
    errors = []
    for i in range((num_samples/2)/num_dendrites):
      current_data = data.getSlice(i*(num_dendrites*2), (i+1)*(num_dendrites*2), 0, dim)
      neuron = Neuron(size = dendrite_length*num_dendrites, num_dendrites = num_dendrites, dendrite_length = dendrite_length, dim = dim, nonlinearity = nonlinearity)
      labels = numpy.asarray([1 for i in range(num_dendrites)] + [-1 for i in range(num_dendrites)])

      neuron.HTM_style_initialize_on_data(current_data, labels)
      error, fp, fn = get_error(current_data, labels, [neuron])
      fps.append(fp)
      fns.append(fn)
      errors.append(error)

    print "Error at r = {} is {}, with {} false positives out of {} samples".format(correlation, numpy.mean(errors), sum(fps), num_samples/2)
    with open("correlation_results_a{}_n{}_s{}.txt".format(a, dim, num_dendrites), "a") as f:
      f.write(str(correlation) + ", " + str(sum(fps)) + ", " + str(num_samples/2) + "\n")

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
  p = Pool(cpu_count())
  p.map(run_false_positive_experiment_correlation, [19+i for i in range(cpu_count())])
