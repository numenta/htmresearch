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
import time
import os
from multiprocessing import Pool, cpu_count
import cPickle

from htmresearch.frameworks.poirazi_neuron_model.neuron_model import sigmoid_nonlinearity
from htmresearch.frameworks.poirazi_neuron_model.neuron_model import Matrix_Neuron as Neuron
from htmresearch.frameworks.poirazi_neuron_model.data_tools import generate_evenly_distributed_data_sparse

def run_false_positive_experiment_dim(
                              numActive = 128,
                              dim = 500,
                              numSamples = 1000,
                              numDendrites = 500,
                              synapses = 24,
                              numTrials = 10000,
                              seed = 42,
                              nonlinearity = sigmoid_nonlinearity(11.5, 5)):
  """
  Run an experiment to test the false positive rate based on number of synapses
  per dendrite, dimension and sparsity.  Uses two competing neurons, along the
  P&M model.

  Based on figure 5B in the original SDR paper.
  """
  numpy.random.seed(seed)
  fps = []
  fns = []
  totalUnclassified = 0

  for trial in range(numTrials):

    # data = generate_evenly_distributed_data_sparse(dim = dim,
    #                                                num_active = numActive,
    #                                                num_samples = numSamples)
    # labels = numpy.asarray([1 for i in range(numSamples / 2)] +
    #                        [-1 for i in range(numSamples / 2)])
    # flipped_labels = labels * -1

    negData = generate_evenly_distributed_data_sparse(dim = dim,
                                                   num_active = numActive,
                                                   num_samples = numSamples/2)
    posData = generate_evenly_distributed_data_sparse(dim = dim,
                                                   num_active = numActive,
                                                   num_samples = numSamples/2)
    halfLabels = numpy.asarray([1 for _ in range(numSamples / 2)])
    flippedHalfLabels = halfLabels * -1

    neuron = Neuron(size =synapses * numDendrites,
                    num_dendrites = numDendrites,
                    dendrite_length = synapses,
                    dim = dim, nonlinearity = nonlinearity)
    neg_neuron = Neuron(size =synapses * numDendrites,
                        num_dendrites = numDendrites,
                        dendrite_length = synapses,
                        dim = dim, nonlinearity = nonlinearity)

    neuron.HTM_style_initialize_on_positive_data(posData)
    neg_neuron.HTM_style_initialize_on_positive_data(negData)

    # Get error for positively labeled data
    fp, fn, uc = get_error(posData, halfLabels, [neuron], [neg_neuron])
    totalUnclassified += uc
    fps.append(fp)
    fns.append(fn)

    # Get error for negatively labeled data
    fp, fn, uc = get_error(negData, flippedHalfLabels, [neuron], [neg_neuron])
    totalUnclassified += uc
    fps.append(fp)
    fns.append(fn)


  print "Error with n = {} : {} FP, {} FN, {} unclassified".format(
    dim, sum(fps), sum(fns), totalUnclassified)

  result = {
    "dim": dim,
    "totalFP": sum(fps),
    "totalFN": sum(fns),
    "total mistakes": sum(fns + fps) + totalUnclassified,
    "error": float(sum(fns + fps) + totalUnclassified) / (numTrials * numSamples),
    "totalSamples": numTrials * numSamples,
    "a": numActive,
    "num_dendrites": numDendrites,
    "totalUnclassified": totalUnclassified,
    "synapses": 24,
    "seed": seed,
  }

  return result


def get_error(data, labels, pos_neurons, neg_neurons = [], add_noise = True):
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
  return num_false_positives, num_false_negatives, num_unclassified


def exp_wrapper(params):
  return run_false_positive_experiment_dim(**params)


def runExperiments(resultsFilename):
  exp_params = []
  for dim in reversed(range(400, 1600, 200)):
    exp_params.append({
      "dim": dim,
      "numTrials" : 1000,
      "seed" : dim,
      "numActive": 128,
    })
  for dim in reversed(range(400, 3000, 200)):
    exp_params.append({
      "dim": dim,
      "numTrials" : 1000,
      "seed" : dim,
      "numActive": 256,
    })
  for dim in reversed(range(400, 1000, 200)):
    exp_params.append({
      "dim": dim,
      "numTrials" : 1000,
      "seed" : dim,
      "numActive": 64,
    })
  numExperiments = len(exp_params)

  pool = Pool(processes=cpu_count()-1)
  rs = pool.map_async(exp_wrapper, exp_params, chunksize=1)
  while not rs.ready():
    remaining = rs._number_left
    pctDone = 100.0 - (100.0*remaining) / numExperiments
    print "    =>", remaining, "experiments remaining, percent complete=",pctDone
    time.sleep(5)
  pool.close()  # No more work
  pool.join()
  result = rs.get()

  # Pickle results for later use
  with open(resultsFilename,"wb") as f:
    cPickle.dump(result,f)


def printResults(results):
  headers = ["a", "dim", "error", "dendrite_length"]
  for h in headers: print h,
  print

  for i, r in enumerate(results):
    for h in headers: print r[h],
    print


if __name__ == "__main__":

  # print run_false_positive_experiment_dim(dim=800, numTrials=10)
  dirName = os.path.dirname(os.path.realpath(__file__))
  resultsFilename = os.path.join(dirName, "classificationResults.pkl")

  # runExperiments(resultsFilename)

  # Debugging
  with open(resultsFilename, "rb") as f:
    results = cPickle.load(f)
    printResults(results)
