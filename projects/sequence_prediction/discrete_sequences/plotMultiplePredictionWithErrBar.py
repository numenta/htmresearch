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
Plot multiple prediction experiment result with error bars
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
                             "high-order-distributed-random-multiple-predictions")
    lstmResults = os.path.join("lstm/results",
                               "high-order-distributed-random-multiple-predictions")
    elmResults = os.path.join("elm/results",
                              "high-order-distributed-random-multiple-predictions")

    for numPrediction in [2, 4]:
      accuracyTM = []
      accuracyLSTM = []
      accuracyELM = []
      for seed in range(10):
        experiment = os.path.join(tmResults,
                                  "num_predictions{:.1f}seed{:.1f}".format(numPrediction, seed),
                                  "0.log")
        (accuracy, x) = loadExperiment(experiment)
        accuracyTM.append(np.array(accuracy))

        experiment = os.path.join(lstmResults,
                                  "seed{:.1f}num_predictions{:.1f}".format(seed, numPrediction),
                                  "0.log")
        (accuracy, x) = loadExperiment(experiment)
        accuracyLSTM.append(np.array(accuracy))

        experiment = os.path.join(elmResults,
                                  "seed{:.1f}num_predictions{:.1f}".format(seed, numPrediction),
                                  "0.log")
        (accuracy, x) = loadExperiment(experiment)
        accuracyELM.append(np.array(accuracy))

      (meanAccuracy, stdAccuracy) = calculateMeanStd(accuracyTM)
      expResult = {'x': x[:len(meanAccuracy)], 'meanAccuracy': meanAccuracy, 'stdAccuracy': stdAccuracy}
      expResults['HTMNumPrediction{:.0f}'.format(numPrediction)] = expResult


      (meanAccuracy, stdAccuracy) = calculateMeanStd(accuracyLSTM)
      expResult = {'x': x[:len(meanAccuracy)], 'meanAccuracy': meanAccuracy, 'stdAccuracy': stdAccuracy}
      expResults['LSTMNumPrediction{:.0f}'.format(numPrediction)] = expResult

      (meanAccuracy, stdAccuracy) = calculateMeanStd(accuracyELM)
      expResult = {'x': x[:len(meanAccuracy)], 'meanAccuracy': meanAccuracy, 'stdAccuracy': stdAccuracy}
      expResults['ELMNumPrediction{:.0f}'.format(numPrediction)] = expResult

    output = open('./result/MultiPredictionExperiment.pkl', 'wb')
    pickle.dump(expResults, output, -1)
    output.close()
  except:
    print "Cannot find raw experiment results"
    print "Plot using saved processed experiment results"

  input = open('./result/MultiPredictionExperiment.pkl', 'rb')
  expResults = pickle.load(input)


  colorList = {"HTMNumPrediction2": "r",
               "LSTMNumPrediction2": "g",
               "ELMNumPrediction2": "b",
               "HTMNumPrediction4": "r",
               "LSTMNumPrediction4": "g",
               "ELMNumPrediction4": "b"}

  modelList = ['HTMNumPrediction2',
               'LSTMNumPrediction2',
               'ELMNumPrediction2',
               'HTMNumPrediction4',
               'LSTMNumPrediction4',
               'ELMNumPrediction4']
  plt.figure(1)
  for model in ['HTMNumPrediction2',
               'LSTMNumPrediction2',
               'ELMNumPrediction2']:
    expResult = expResults[model]
    plotWithErrBar(expResult['x'],
                   expResult['meanAccuracy'], expResult['stdAccuracy'],
                   colorList[model])
  plt.legend(['HTM', 'LSTM', 'ELM'], loc=4)

  plt.figure(2)
  for model in ['HTMNumPrediction4',
               'LSTMNumPrediction4',
               'ELMNumPrediction4']:
    expResult = expResults[model]
    plotWithErrBar(expResult['x'],
                   expResult['meanAccuracy'], expResult['stdAccuracy'],
                   colorList[model])
  plt.legend(['HTM', 'LSTM', 'ELM'], loc=4)
  for fig in [1, 2]:
    plt.figure(fig)
    retrainLSTMAt = np.arange(start=1000, stop=12000, step=1000)
    for line in retrainLSTMAt:
      plt.axvline(line, color='orange')
    plt.ylim([-0.05, 1.05])
    # plt.xlim([0, 11000])

  plt.figure(1)
  plt.savefig('./result/model_performance_2_prediction_errbar.pdf')

  plt.figure(2)
  plt.savefig('./result/model_performance_4_prediction_errbar.pdf')