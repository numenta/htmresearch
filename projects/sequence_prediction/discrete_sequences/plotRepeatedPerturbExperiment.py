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
Plot sequence prediction & perturbation experiment result
"""
import os
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

from plot import movingAverage
from plot import computeAccuracy
from plot import readExperiment
mpl.rcParams['pdf.fonttype'] = 42
plt.ion()
plt.close('all')


def loadExperiment(experiment):
  print "Loading experiment ", experiment
  data = readExperiment(experiment)
  (accuracy, x) = computeAccuracy(data['predictions'],
                                  data['truths'],
                                  data['iterations'],
                                  resets=data['resets'],
                                  randoms=data['randoms'])
  accuracy = movingAverage(accuracy, min(len(accuracy), 100))
  return (accuracy, x)



def calculateMeanStd(accuracyAll):
  numRepeats = len(accuracyAll)
  numLength = min([len(a) for a in accuracyAll])
  accuracyMat = np.zeros(shape=(numRepeats, numLength))
  for i in range(numRepeats):
    accuracyMat[i, :] = accuracyAll[i][:numLength]
  meanAccuracy = np.mean(accuracyMat, axis=0)
  stdAccuracy = np.std(accuracyMat, axis=0)
  return (meanAccuracy, stdAccuracy)



def plotWithErrBar(x, y, error, color):
  plt.fill_between(x, y-error, y+error,
    alpha=0.3, edgecolor=color, facecolor=color)
  plt.plot(x, y, color, color=color, linewidth=4)
  plt.ylabel('Prediction Accuracy')
  plt.xlabel(' Number of elements seen')

if __name__ == '__main__':

  try:
    # Load raw experiment results
    # You have to run the experiments
    # In ./tm/
    # python tm_suite.py --experiment="high-order-distributed-random-perturbed" -d
    # In ./lstm/
    # python suite.py --experiment="high-order-distributed-random-perturbed" -d
    expResults = {}
    tmResults = os.path.join("tm/results",
                             "high-order-distributed-random-perturbed")
    accuracyAll = []
    for seed in range(20):
      experiment = os.path.join(tmResults,
                                "seed" + "{:.1f}".format(seed), "0.log")
      (accuracy, x) = loadExperiment(experiment)
      accuracyAll.append(np.array(accuracy))
    (meanAccuracy, stdAccuracy) = calculateMeanStd(accuracyAll)
    expResult = {'x': x, 'meanAccuracy': meanAccuracy, 'stdAccuracy': stdAccuracy}
    expResults['HTM'] = expResult

    lstmResults = os.path.join("lstm/results",
                                 "high-order-distributed-random-perturbed")


    for learningWindow in [1000.0, 3000.0, 9000.0]:
      accuracyAll = []
      for seed in range(20):
        experiment = os.path.join(
          lstmResults, "seed{:.1f}learning_window{:.1f}".format(seed, learningWindow),
          "0.log")
        (accuracy, x) = loadExperiment(experiment)
        accuracyAll.append(np.array(accuracy))

      (meanAccuracy, stdAccuracy) = calculateMeanStd(accuracyAll)

      expResults['LSTM-'+"{:.0f}".format(learningWindow)] = {
        'x': x, 'meanAccuracy': meanAccuracy, 'stdAccuracy': stdAccuracy}

    output = open('./result/ContinuousLearnExperiment.pkl', 'wb')
    pickle.dump(expResults, output, -1)
    output.close()
  except:
    print "Cannot find raw experiment results"
    print "Plot using saved processed experiment results"

  input = open('./result/ContinuousLearnExperiment.pkl', 'rb')
  expResults = pickle.load(input)

  plt.figure()
  colorList = {"HTM": "r", "LSTM-1000": "b", "LSTM-3000": "y", "LSTM-9000": "g"}
  for model in ['HTM', 'LSTM-1000', 'LSTM-3000', 'LSTM-9000']:
    expResult = expResults[model]
    plotWithErrBar(expResult['x'],
                   expResult['meanAccuracy'], expResult['stdAccuracy'],
                   colorList[model])

  plt.legend(['HTM', 'LSTM-1000', 'LSTM-3000', 'LSTM-9000'], loc=4)

  retrainLSTMAt = np.arange(start=1000, stop=20000, step=1000)
  for line in retrainLSTMAt:
    plt.axvline(line, color='orange')

  plt.axvline(10000, color='black')
  plt.ylim([-0.05, 1.05])
  plt.savefig('./result/model_performance_high_order_prediction.pdf')