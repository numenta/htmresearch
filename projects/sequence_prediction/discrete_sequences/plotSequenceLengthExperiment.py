#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

"""
Plot temporal noise experiment result
"""
import os
import pickle

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

from plot import plotAccuracy
from plot import movingAverage
from plot import computeAccuracy
from plot import readExperiment
mpl.rcParams['pdf.fonttype'] = 42
plt.ion()
plt.close('all')


def computeAccuracy(predictions, truths, iterations,
                    resets=None, randoms=None, num=None,
                    sequenceCounter=None):
  accuracy = []
  numIteration = []
  numSequences = []

  for i in xrange(len(predictions) - 1):
    if num is not None and i > num:
      continue

    if truths[i] is None:
      continue

    # identify the end of sequence
    if resets is not None or randoms is not None:
      if not (resets[i+1] or randoms[i+1]):
        continue

    correct = truths[i] is None or truths[i] in predictions[i]
    accuracy.append(correct)
    numSequences.append(sequenceCounter[i])
    numIteration.append(iterations[i])

  return (accuracy, numIteration, numSequences)



if __name__ == '__main__':

  lengths = [10, 20, 40, 60, 80, 100]

  tmResults = os.path.join("tm/results",
                           "high-order-variable-length")

  try:
    # Load raw experiment results
    # You have to run the experiments in ./tm
    # python tm_suite.py --experiment="high-order-variable-length" -d
    expResults = {}
    for length in lengths:
      experiment = os.path.join(tmResults,
                                "sequence_length"+"{:.1f}".format(length),
                                "0.log")
      data = readExperiment(experiment)

      (accuracy, numIteration, numSequences) = computeAccuracy(
        data['predictions'],
        data['truths'],
        data['iterations'],
        resets=data['resets'],
        randoms=data['randoms'],
        sequenceCounter=data['sequenceCounter'])

      expResult = {"length": length,
                   "accuracy": accuracy,
                   "numIteration": numIteration,
                   "numSequences": numSequences}
      expResults[length] = expResult
    output = open(os.path.join(tmResults,
                               'SequenceLengthExperiment.pkl'), 'wb')
    pickle.dump(expResults, output, -1)
    output.close()
  except:
    print "Cannot find raw experiment results"
    print "Plot using saved processed experiment results"

  input = open(os.path.join(tmResults,
                            'SequenceLengthExperiment.pkl'), 'rb')
  expResults = pickle.load(input)

  # load processed experiment results and plot them
  numSequenceRequired = []
  numIterationRequired = []
  lengths = np.sort(expResults.keys())
  for length in lengths:
    expResult = expResults[length]
    accuracy = expResult["accuracy"]
    numIteration = expResult["numIteration"]
    numSequences = expResult["numSequences"]

    movingData = movingAverage(accuracy, min(len(accuracy), 100))
    numSequenceRequired.append(
      numSequences[np.where(np.array(movingData) >= 0.999)[0][1]])
    numIterationRequired.append(
      numIteration[np.where(np.array(movingData) >= 0.999)[0][1]])
    # injectNoiseAt = data['sequenceCounter'][12000]
    # x = numpy.array(x) - injectNoiseAt + 1400
    plotAccuracy((accuracy, numIteration),
                 data['trains'],
                 window=50,
                 type=type,
                 label='NoiseExperiment',
                 hideTraining=True,
                 lineSize=1.0)
    plt.xlabel('# of sequences seen')

  plt.ylabel('Prediction accuracy')
  # plt.xlim([0, 200])
  plt.ylim([0, 1.05])
  plt.legend(lengths, loc=4)
  plt.savefig('./result/sequence_length_experiment_performance.pdf')

  plt.figure()
  plt.plot(lengths, numSequenceRequired, '-*')
  plt.xlabel(' Sequence Order ')
  plt.ylabel(' # Sequences seen to achieve perfect prediction')
  plt.savefig('./result/requred_sequence_number_vs_sequence_order.pdf')

  plt.figure()
  plt.plot(lengths, numIterationRequired, '-*')
  plt.xlabel(' Sequence Order ')
  plt.ylabel(' # Elements seen to achieve perfect prediction')
  plt.savefig('./result/requred_elements_number_vs_sequence_order.pdf')
