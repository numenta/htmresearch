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



def analyzeResult(x, accuracy, perturbAt=10000, movingAvg=True, smooth=True):
  if movingAvg:
    accuracy = movingAverage(accuracy, min(len(accuracy), 100))

  x = np.array(x)
  accuracy = np.array(accuracy)
  if smooth:
    # perform smoothing convolution
    mask = np.ones(shape=(100,))
    mask = mask/np.sum(mask)
    # extend accuracy vector to eliminate boundary effect of convolution
    accuracy = np.concatenate((accuracy, np.ones((200, ))*accuracy[-1]))
    accuracy = np.convolve(accuracy, mask, 'same')
    accuracy = accuracy[:len(x)]


  perturbAtX = np.where(x > perturbAt)[0][0]

  finalAccuracy = accuracy[perturbAtX-len(mask)/2]
  learnTime = min(np.where(np.logical_and(accuracy > finalAccuracy * 0.95,
                                          x < x[perturbAtX - len(mask)/2-1]))[0])
  learnTime = x[learnTime]

  finalAccuracyAfterPerturbation = accuracy[-1]
  learnTimeAfterPerturbation = min(np.where(
    np.logical_and(accuracy > finalAccuracyAfterPerturbation * 0.95,
                   x > x[perturbAtX + len(mask)]))[0])

  learnTimeAfterPerturbation = x[learnTimeAfterPerturbation] - perturbAt

  result = {"finalAccuracy": finalAccuracy,
            "learnTime": learnTime,
            "finalAccuracyAfterPerturbation": finalAccuracyAfterPerturbation,
            "learnTimeAfterPerturbation": learnTimeAfterPerturbation}
  return result



if __name__ == '__main__':

  try:
    # Load raw experiment results
    # You have to run the experiments
    # In ./tm/
    # python tm_suite.py --experiment="high-order-distributed-random-perturbed" -d
    # In ./lstm/
    # python suite.py --experiment="high-order-distributed-random-perturbed" -d
    expResults = {}
    expResultsAnaly = {}

    # TDNN
    tdnnResults = os.path.join("tdnn/results",
                               "high-order-distributed-random-perturbed")
    accuracyAll = []
    exptLabel = 'TDNN'
    expResultsAnaly[exptLabel] = []
    for seed in range(20):
      experiment = os.path.join(tdnnResults,
                                "seed" + "{:.1f}".format(
                                  seed) + "learning_window3000.0", "0.log")
      (accuracy, x) = loadExperiment(experiment)
      expResultsAnaly[exptLabel].append(analyzeResult(x, accuracy))
      accuracy = movingAverage(accuracy, min(len(accuracy), 100))
      accuracyAll.append(np.array(accuracy))

    (meanAccuracy, stdAccuracy) = calculateMeanStd(accuracyAll)
    x = x[:len(meanAccuracy)]
    expResults[exptLabel] = {
      'x': x, 'meanAccuracy': meanAccuracy, 'stdAccuracy': stdAccuracy}

    tdnnResults = os.path.join("tdnn/results",
                               "high-order-distributed-random-perturbed-long-window")
    accuracyAll = []
    exptLabel = 'TDNN-long'
    expResultsAnaly[exptLabel] = []
    for seed in range(8):
      experiment = os.path.join(tdnnResults,
                                "seed" + "{:.1f}".format(
                                  seed) + "learning_window3000.0", "0.log")
      (accuracy, x) = loadExperiment(experiment)
      expResultsAnaly[exptLabel].append(analyzeResult(x, accuracy))
      accuracy = movingAverage(accuracy, min(len(accuracy), 100))
      accuracyAll.append(np.array(accuracy))

    (meanAccuracy, stdAccuracy) = calculateMeanStd(accuracyAll)
    x = x[:len(meanAccuracy)]
    expResults[exptLabel] = {
      'x': x, 'meanAccuracy': meanAccuracy, 'stdAccuracy': stdAccuracy}

    tdnnResults = os.path.join("tdnn/results",
                               "high-order-distributed-random-perturbed-short-window")
    accuracyAll = []
    exptLabel = 'TDNN-short'
    expResultsAnaly[exptLabel] = []
    for seed in range(8):
      experiment = os.path.join(tdnnResults,
                                "seed" + "{:.1f}".format(
                                  seed) + "learning_window3000.0", "0.log")
      (accuracy, x) = loadExperiment(experiment)
      expResultsAnaly[exptLabel].append(analyzeResult(x, accuracy))
      accuracy = movingAverage(accuracy, min(len(accuracy), 100))
      accuracyAll.append(np.array(accuracy))

    (meanAccuracy, stdAccuracy) = calculateMeanStd(accuracyAll)
    x = x[:len(meanAccuracy)]
    expResults[exptLabel] = {
      'x': x, 'meanAccuracy': meanAccuracy, 'stdAccuracy': stdAccuracy}

    # HTM
    tmResults = os.path.join("tm/results",
                             "high-order-distributed-random-perturbed-small-alphabet")
    accuracyAll = []
    exptLabel = 'HTM'
    expResultsAnaly[exptLabel] = []
    for seed in range(10):
      experiment = os.path.join(tmResults,
                                "seed" + "{:.1f}".format(seed), "0.log")
      (accuracy, x) = loadExperiment(experiment)
      expResultsAnaly[exptLabel].append(analyzeResult(x, accuracy))
      accuracy = movingAverage(accuracy, min(len(accuracy), 100))
      accuracyAll.append(np.array(accuracy))

    (meanAccuracy, stdAccuracy) = calculateMeanStd(accuracyAll)
    x = x[:len(meanAccuracy)]
    expResults[exptLabel] = {
      'x': x, 'meanAccuracy': meanAccuracy, 'stdAccuracy': stdAccuracy}

    # ELM
    elmResults = os.path.join("elm/results",
                             "high-order-distributed-random-perturbed")
    accuracyAll = []
    exptLabel = 'ELM'
    expResultsAnaly[exptLabel] = []
    for seed in range(10, 20):
      experiment = os.path.join(elmResults,
                                "seed" + "{:.1f}".format(seed), "0.log")
      (accuracy, x) = loadExperiment(experiment)
      expResultsAnaly[exptLabel].append(analyzeResult(x, accuracy))
      accuracy = movingAverage(accuracy, min(len(accuracy), 100))
      accuracyAll.append(np.array(accuracy))

    (meanAccuracy, stdAccuracy) = calculateMeanStd(accuracyAll)
    x = x[:len(meanAccuracy)]
    expResults[exptLabel] = {
      'x': x, 'meanAccuracy': meanAccuracy, 'stdAccuracy': stdAccuracy}

    # LSTM
    lstmResults = os.path.join("lstm/results",
                                 "high-order-distributed-random-perturbed")


    for learningWindow in [1000.0, 3000.0, 9000.0]:
      accuracyAll = []
      exptLabel = 'LSTM-'+"{:.0f}".format(learningWindow)
      expResultsAnaly[exptLabel] = []
      for seed in range(20):
        experiment = os.path.join(
          lstmResults, "seed{:.1f}learning_window{:.1f}".format(seed, learningWindow),
          "0.log")
        (accuracy, x) = loadExperiment(experiment)
        expResultsAnaly[exptLabel].append(analyzeResult(x, accuracy))
        accuracy = movingAverage(accuracy, min(len(accuracy), 100))
        accuracyAll.append(np.array(accuracy))

      (meanAccuracy, stdAccuracy) = calculateMeanStd(accuracyAll)

      expResults[exptLabel] = {
        'x': x, 'meanAccuracy': meanAccuracy, 'stdAccuracy': stdAccuracy}

    # online- LSTM
    lstmResults = os.path.join("lstm/results",
                                 "high-order-distributed-random-perturbed-online")

    for learningWindow in [100.0]:
      accuracyAll = []
      exptLabel = 'LSTM-online'+"{:.0f}".format(learningWindow)
      expResultsAnaly[exptLabel] = []
      for seed in range(10):
        experiment = os.path.join(
          lstmResults, "seed{:.1f}learning_window{:.1f}".format(seed, learningWindow),
          "0.log")
        (accuracy, x) = loadExperiment(experiment)
        expResultsAnaly[exptLabel].append(analyzeResult(x, accuracy))
        accuracy = movingAverage(accuracy, min(len(accuracy), 100))
        accuracyAll.append(np.array(accuracy))

      (meanAccuracy, stdAccuracy) = calculateMeanStd(accuracyAll)
      x = x[:len(meanAccuracy)]
      expResults[exptLabel] = {
        'x': x, 'meanAccuracy': meanAccuracy, 'stdAccuracy': stdAccuracy}

    output = open('./result/ContinuousLearnExperiment.pkl', 'wb')
    pickle.dump(expResults, output, -1)
    output.close()

    output = open('./result/ContinuousLearnExperimentAnaly.pkl', 'wb')
    pickle.dump(expResultsAnaly, output, -1)
    output.close()

  except:
    print "Cannot find raw experiment results"
    print "Plot using saved processed experiment results"

  expResults = pickle.load(open('./result/ContinuousLearnExperiment.pkl', 'rb'))
  expResultsAnaly = pickle.load(open('./result/ContinuousLearnExperimentAnaly.pkl', 'rb'))

  plt.figure(1)
  fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
  colorList = {"HTM": "r", "ELM": "b", "LSTM-1000": "y", "LSTM-9000": "g",
               "TDNN": "c", "LSTM-online100": "m"}
  modelList = ['HTM', 'ELM', 'TDNN', 'LSTM-1000',  'LSTM-9000', 'LSTM-online100']

  for model in modelList:
    expResult = expResults[model]

    plt.figure(1)
    plotWithErrBar(expResult['x'],
                   expResult['meanAccuracy'], expResult['stdAccuracy'],
                   colorList[model])

    perturbAtX = np.where(np.array(expResult['x']) > 10000)[0][0]

    result = analyzeResult(expResult['x'], expResult['meanAccuracy'], movingAvg=False)
    resultub = analyzeResult(expResult['x'],
                             expResult['meanAccuracy']-expResult['stdAccuracy'],
                             movingAvg=False)
    resultlb = analyzeResult(expResult['x'],
                             expResult['meanAccuracy']+expResult['stdAccuracy'],
                             movingAvg=False)

    learnTimeErr = [result['learnTime']-resultlb['learnTime'],
                    resultub['learnTime']-result['learnTime']]
    learnTimeErrAfterPerturb = [
      result['learnTimeAfterPerturbation']-resultlb['learnTimeAfterPerturbation'],
      resultub['learnTimeAfterPerturbation']-result['learnTimeAfterPerturbation']]

    axs[0].errorbar(x=result['learnTime'], y=result['finalAccuracy'],
                 yerr=expResult['stdAccuracy'][perturbAtX],
                 xerr=np.mean(learnTimeErr), ecolor=colorList[model])


    axs[1].errorbar(x=result['learnTimeAfterPerturbation'],
                 y=result['finalAccuracyAfterPerturbation'],
                 yerr=expResult['stdAccuracy'][-1],
                 xerr=np.mean(learnTimeErrAfterPerturb),
                  ecolor=colorList[model])

    axs[0].set_title("Before modification")
    axs[1].set_title("After modification")

  plt.figure(1)
  plt.legend(modelList, loc=4)

  retrainLSTMAt = np.arange(start=1000, stop=20000, step=1000)
  for line in retrainLSTMAt:
    plt.axvline(line, color='orange')

  plt.axvline(10000, color='black')
  plt.ylim([-0.05, 1.05])
  plt.xlim([0, 20000])

  for ax in axs:
    ax.legend(modelList, loc=4)
    ax.set_xlabel(' Number of samples required to achieve final accuracy')
    ax.set_ylabel(' Final accuracy ')
    ax.set_ylim([0.5, 1.05])
    # axs[1].set_xlim([0, 30000])

  axs[0].set_xlim([0, 10000])
  axs[1].set_xlim([0, 10000])

  plt.figure(1)
  plt.savefig('./result/model_performance_high_order_prediction.pdf')
  plt.figure(2)
  plt.savefig('./result/model_performance_summary_high_order_prediction.pdf')
  #
  # # plot accuracy vs
  # plt.figure(2)
  # plt.figure(3)
  # for model in ['HTM', 'LSTM-1000', 'LSTM-3000', 'LSTM-9000']:
  #   finalAccuracy = []
  #   finalAccuracyAfterPerturbation = []
  #   learnTime = []
  #   learnTimeAfterPerturbation = []
  #
  #   for result in expResultsAnaly[model]:
  #     finalAccuracy.append(result['finalAccuracy'])
  #     finalAccuracyAfterPerturbation.append(result['finalAccuracyAfterPerturbation'])
  #     learnTime.append(result['learnTime'])
  #     learnTimeAfterPerturbation.append(result['learnTimeAfterPerturbation'])
  #
  #   plt.figure(2)
  #   plt.errorbar(x=np.mean(learnTime), y=np.mean(finalAccuracy),
  #                yerr=np.std(finalAccuracy), xerr=np.std(learnTime))
  #
  #   plt.figure(3)
  #   plt.errorbar(x=np.mean(learnTimeAfterPerturbation),
  #                y=np.mean(finalAccuracyAfterPerturbation),
  #                yerr=np.std(finalAccuracyAfterPerturbation),
  #                xerr=np.std(learnTimeAfterPerturbation))
  #
  # for fig in [2, 3]:
  #   plt.figure(fig)
  #   plt.legend(['HTM', 'LSTM-1000', 'LSTM-3000', 'LSTM-9000'], loc=3)
  #   plt.xlabel(' Number of sequences required to achieve final accuracy')
  #   plt.ylabel(' Final accuracy ')