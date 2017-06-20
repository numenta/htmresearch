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
from htmresearch.frameworks.poirazi_neuron_model.data_tools import generate_evenly_distributed_data_sparse
numpy.random.seed(19)

def run_false_positive_experiment_dim(num_neurons = 1,
                    num_neg_neurons = 1,
                    a = 128,
                    test_dims = range(1100, 2100, 200),
                    num_samples = 1000,
                    num_dendrites = 500,
                    dendrite_length = 24,
                    num_trials = 10000,
                    nonlinearity = sigmoid_nonlinearity(11.5, 5)):
  """
  Run an experiment to test the false positive rate based on number of
  synapses per dendrite, dimension and sparsity.  Uses two competing neurons,
  along the P&M model.

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

      error, fp, fn, uc = get_error(data, labels, [neuron], [neg_neuron], add_noise = True)

      fps.append(fp)
      fns.append(fn)
      print "Error at n = {} is {}, with {} false positives and {} false negatives, with {} unclassified".format(dim, error, fp, fn, uc)

    with open("pm_dim_FP_{}.txt".format(a), "a") as f:
      f.write(str(dim) + ", " + str(sum(fns + fps)) + ", " + str(num_trials*num_samples) + "\n")

def get_error(data, labels, pos_neurons, neg_neurons = [], add_noise = False):
  """
  Calculates error, including number of false positives and false negatives.

  Written to allow the use of multiple neurons, in case we attempt to use a
  population in the future.

  """
  num_correct = 0
  num_false_positives = 0
  num_false_negatives = 0
  num_unclassified = 0
  classifications = numpy.zeros(data.nRows())
  for neuron in pos_neurons:
    classifications += neuron.calculate_on_entire_dataset(data)
  for neuron in neg_neurons:
    classifications -= neuron.calculate_on_entire_dataset(data)
  for classification, label in zip(classifications, labels):
    if add_noise:
      # Add a tiny bit of noise, to stop perfect ties.
      classification += ((numpy.random.rand() - 0.5)/1000.)
    classification = numpy.sign(classification)
    if classification == 0.:
      num_unclassified += 1
    elif classification >= 1 and label >= 1:
      num_correct += 1.0
    elif classification < 0 and label <= 0:
      num_correct += 1.
    elif classification >= 1. and label <= 0:
      num_false_positives += 1
    elif classification < 0 and label >= 1:
      num_false_negatives += 1
  return (1.*num_false_positives + num_false_negatives + num_unclassified)/data.nRows(), num_false_positives, num_false_negatives, num_unclassified


if __name__ == "__main__":
  run_false_positive_experiment_dim()
