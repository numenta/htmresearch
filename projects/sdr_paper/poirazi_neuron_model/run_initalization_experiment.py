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

import random
import numpy
from htmresearch.frameworks.poirazi_neuron_model.neuron_model import (
  power_nonlinearity, threshold_nonlinearity)
from htmresearch.frameworks.poirazi_neuron_model.neuron_model import Matrix_Neuron as Neuron
from htmresearch.frameworks.poirazi_neuron_model.data_tools import (
  generate_data, generate_evenly_distributed_data_sparse, split_sparse_matrix)
from sklearn.cluster import KMeans
from multiprocessing import Pool, cpu_count
from nupic.bindings.math import *
from collections import Counter

def run_initialization_experiment(seed,
                                  num_neurons = 50,
                                  dim = 40,
                                  num_bins = 10,
                                  num_samples = 50*600,
                                  neuron_size = 10000,
                                  num_dendrites = 400,
                                  dendrite_length = 25,
                                  test_powers = range(10, 11)
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

  numpy.random.seed(seed)
  for power in test_powers:
    print "Testing power:", power
    nonlinearity = power_nonlinearity(power)
    pos_neurons = [Neuron(size = neuron_size, num_dendrites = num_dendrites, dendrite_length = dendrite_length, nonlinearity = nonlinearity, dim = dim*num_bins) for i in range(num_neurons/2)]
    neg_neurons = [Neuron(size = neuron_size, num_dendrites = num_dendrites, dendrite_length = dendrite_length, nonlinearity = nonlinearity, dim = dim*num_bins) for i in range(num_neurons/2)]
    #pos, neg = generate_evenly_distributed_data_sparse(dim = 400, num_active = 40, num_samples = num_samples/2), generate_evenly_distributed_data_sparse(dim = 400, num_active = 40, num_samples = num_samples/2)
    pos, neg = generate_data(dim = dim, num_bins = num_bins, num_samples = num_samples, sparse = True)

    if (pos.nRows() > num_dendrites*len(pos_neurons)):
      print "Too much data to have unique dendrites for positive neurons, clustering"
      pos = pos.toDense()
      model = KMeans(n_clusters = len(pos_neurons), n_jobs=1)
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
      model = KMeans(n_clusters = len(neg_neurons), n_jobs=1)
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


    print "Calculating error"
    labels = [1 for i in range(pos.nRows())] + [-1 for i in range(neg.nRows())]
    data = pos
    data.append(neg)

    error, fp, fn = get_error(data, labels, pos_neurons, neg_neurons)
    print "Error at initialization is {}, with {} false positives and {} false negatives".format(error, fp, fn)
    #with open("initialization_experiment.txt", "a") as f:
    #  f.write(str(power) + ", " + str(error) + "\n")
    return error


def get_error(data, labels, pos_neurons, neg_neurons = [], add_noise = True):
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
  if add_noise:
    classifications += (numpy.random.rand() - 0.5)/1000
  classifications = numpy.sign(classifications)
  for classification, label in zip(classifications, labels):
    if classification > 0 and label > 0:
      num_correct += 1.0
    elif classification <= 0 and label <= 0:
      num_correct += 1.0
    elif classification > 0 and label <= 0:
      num_false_positives += 1
    else:
      num_false_negatives += 1
  return (1.*num_false_positives + num_false_negatives)/data.nRows(), num_false_positives, num_false_negatives


if __name__ == "__main__":
  p = Pool(cpu_count())
  errors = p.map(run_initialization_experiment, [100+i for i in range(50)])
  print numpy.mean(errors)
