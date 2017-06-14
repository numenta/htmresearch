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
from htmresearch.frameworks.poirazi_neuron_model.neuron_model import Matrix_Neuron as Neuron
from htmresearch.frameworks.poirazi_neuron_model.neuron_model import threshold_nonlinearity
from htmresearch.frameworks.poirazi_neuron_model.data_tools import (
  generate_data, generate_evenly_distributed_data)
from nupic.bindings.math import SM32
from collections import Counter

def run_classification_experiment(num_neurons = 1,
                  dim = 40,
                  num_bins = 10,
                  num_samples = 4000,
                  neuron_size = 10000,
                  num_dendrites = 1000,
                  dendrite_length = 10,
                  num_replacement = 25,
                  training_iterations = 10000,
                  nonlinearity = threshold_nonlinearity(5)):

  """
  Run an experiment testing the ability of a single HTM-style neuron to learn
  a classification task, with an HTM learning rule and threshold nonlinearity

  The number of samples can also be dramatically increased, to test the
  ability of the neuron to initalize itself using clustering, although
  this will result in inferior performance.  In this case, it is
  recommended to use a relatively low threshold, such as dendrite_length/2.
  It may also help to initialize the neuron with a perm_dec to perm_inc ratio
  the default, as this causes it to properly prioritize unlearning poor
  connections (which, in clustering, are numerous).
  """
  neuron = Neuron(size = neuron_size, num_dendrites = num_dendrites, dendrite_length = dendrite_length, nonlinearity = nonlinearity, dim = dim*num_bins)
  pos, neg = generate_data(dim = dim, num_bins = num_bins, num_samples = num_samples)
  data = [(datapoint, 1) for datapoint in pos] + [(datapoint, -1) for datapoint in neg]
  random.shuffle(data)
  shuffled_data, labels = zip(*data)
  data = SM32(shuffled_data)

  error_history = []

  neuron.HTM_style_initialize_on_data(data, labels)
  error, fp, fn = get_error(data, labels, [neuron])
  print "Pretraining error is {}, with {} false positives and {} false negatives".format(error, fp, fn)
  for i in range(training_iterations):
    neuron.HTM_style_train_on_data(data, labels)
    error, fp, fn = get_error(data, labels, [neuron])
    error_history.append(error)
    print "Error at iteration {} is {}, with {} false positives and {} false negatives".format(i, error, fp, fn)


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
  run_classification_experiment()
