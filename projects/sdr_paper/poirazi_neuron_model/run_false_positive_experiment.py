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
from htmresearch.frameworks.poirazi_neuron_model.neuron_model import power_nonlinearity, threshold_nonlinearity
from htmresearch.frameworks.poirazi_neuron_model.neuron_model import Matrix_Neuron as Neuron
from htmresearch.frameworks.poirazi_neuron_model.data_tools import generate_evenly_distributed_data_sparse
numpy.random.seed(19)

def run_HTM_false_positive_experiment_synapses(num_neurons = 1,
                         a = 512,
                         dim = 16000,
                         num_samples = 1000,
                         num_dendrites = 500,
                         test_dendrite_lengths = range(2, 32, 2),
                         num_trials = 1000):
  """
  Run an experiment to test the false positive rate based on number of
  synapses per dendrite, dimension and sparsity.  Uses a single neuron,
  with a threshold nonlinearity of theta = s/2.

  Based on figure 5B in the original SDR paper.

  The parameters used in generating the figure for this experiment are:
  1.  a = 512, dim = 16000
  2.  a = 4000, dim = 16000
  3.  a = 32, dim = 2000
  4.  a = 32, dim = 4000
  In each case, we used 1000 samples per trial, 1000 trials, 500 dendrite
  segments, and tested dendrite lengths in (2, 4, ..., 30), with the threshold
  for each segment being half its length.
  """
  for dendrite_length in test_dendrite_lengths:
    nonlinearity = threshold_nonlinearity(dendrite_length / 2)

    fps = []
    fns = []

    for trial in range(num_trials):

      neuron = Neuron(size = dendrite_length*num_dendrites, num_dendrites = num_dendrites, dendrite_length = dendrite_length, dim = dim, nonlinearity = nonlinearity)
      pos, neg = generate_evenly_distributed_data_sparse(dim = dim, num_active = a, num_samples = num_samples/2), generate_evenly_distributed_data_sparse(dim = dim, num_active = a, num_samples = num_samples/2)
      #labels = numpy.asarray([1 for i in range(num_samples/2)] + [-1 for i in range(num_samples/2)])

      neuron.HTM_style_initialize_on_data(pos, numpy.asarray([1 for i in range(num_samples/2)]))

      error, fp, fn = get_error(neg, [-1 for i in range(num_samples/2)], [neuron])

      fps.append(fp)
      fns.append(fn)
      print "Error at {} synapses per dendrite is {}, with {} false positives and {} false negatives".format(dendrite_length, fp/(num_samples/2.), fp, fn)

    with open("num_dendrites_FP_{}_{}.txt".format(a, dim), "a") as f:
      f.write(str(dendrite_length) + ", " + str(sum(fps)) + ", " + str(num_trials*num_samples/2.) + "\n")


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
  run_HTM_false_positive_experiment_synapses()
