#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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


import os
import pickle

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy

from plot import computeAccuracy
from plot import readExperiment

mpl.rcParams['pdf.fonttype'] = 42
plt.ion()
plt.close('all')


def loadRepeatedExperiments(expDir, killCellPercentRange, seedRange, killCellAt):
  meanAccuracy = []
  stdAccuracy = []
  for killCellPercent in killCellPercentRange:
    accuracyList = []
    for seed in range(10):
      if killCellPercent == 0:
        experiment = os.path.join(
          expDir,"kill_cell_percent{:1.1f}seed{:1.1f}/0.log".format(
            killCellPercent, seed))
      else:
        experiment = os.path.join(
          expDir,"kill_cell_percent{:1.2f}seed{:1.1f}/0.log".format(
            killCellPercent, seed))
      expResults = readExperiment(experiment)
      print "Load Experiment: ", experiment

      (accuracy, x) = computeAccuracy(expResults['predictions'],
                                      expResults['truths'],
                                      expResults['iterations'],
                                      resets=expResults['resets'],
                                      randoms=expResults['randoms'])
      idx = numpy.array([i for i in range(len(x)) if x[i] > killCellAt])
      accuracy = numpy.array(accuracy)
      accuracyList.append(float(numpy.sum(accuracy[idx])) / len(accuracy[idx]))
    meanAccuracy.append(numpy.mean(accuracyList))
    stdAccuracy.append(numpy.std(accuracyList))
  return meanAccuracy, stdAccuracy



if __name__ == '__main__':

  try:
    # Load raw experiment results
    # You have to run the experiments

    killCellPercentRange = list(numpy.arange(6) / 10.0)
    seedRange = range(10)

    (meanAccuracyLSTM, stdAccuracyLSTM) = loadRepeatedExperiments(
      "lstm/results/high-order-distributed-random-kill-cell/",
      killCellPercentRange, seedRange, killCellAt=10000)

    (meanAccuracyELM, stdAccuracyELM) = loadRepeatedExperiments(
      "elm/results/high-order-distributed-random-kill-cell/",
      killCellPercentRange, seedRange, killCellAt=20000)

    (meanAccuracyHTM, stdAccuracyHTM) = loadRepeatedExperiments(
      "tm/results/high-order-distributed-random-kill-cell/",
      killCellPercentRange, seedRange, killCellAt=10000)

    expResults = {}
    expResults['HTM'] = {'x': killCellPercentRange,
                         'meanAccuracy': meanAccuracyHTM,
                         'stdAccuracy': stdAccuracyHTM}
    expResults['ELM'] = {'x': killCellPercentRange,
                         'meanAccuracy': meanAccuracyELM,
                         'stdAccuracy': stdAccuracyELM}
    expResults['LSTM'] = {'x': killCellPercentRange,
                         'meanAccuracy': meanAccuracyLSTM,
                         'stdAccuracy': stdAccuracyLSTM}
    output = open('./result/FaultTolerantExpt.pkl', 'wb')
    pickle.dump(expResults, output, -1)
    output.close()
  except:
    print "Cannot find raw experiment results"
    print "Plot using saved processed experiment results"
    expResults = pickle.load(open('./result/FaultTolerantExpt.pkl', 'rb'))

  plt.figure(1)
  colorList = {'HTM': 'r', 'ELM': 'b', 'LSTM': 'g'}
  for model in ['HTM', 'ELM', 'LSTM']:
    plt.errorbar(expResults[model]['x'],
                 expResults[model]['meanAccuracy'],
                 expResults[model]['stdAccuracy'],
                 color=colorList[model],
                 marker='o')
  plt.legend(['HTM', 'ELM', 'LSTM'], loc=3)
  plt.xlabel('Fraction of cell death ')
  plt.ylabel('Accuracy after cell death')
  plt.ylim([0.1, 1.05])
  plt.xlim([-0.02, .52])
  plt.savefig('./result/model_performance_after_cell_death.pdf')
  #
  # for killCellPercent in KILLCELL_PERCENT:
  #   # HTM experiments
  #   tmResultDir = 'tm/result/'
  #   experiment = os.path.join(tmResultDir, "kill_cell_percent{:1.1f}".format(
  #     killCellPercent)) + '/0.log'
  #
  #   expResults = readExperiment(experiment)
  #
  #   killCellAt = 10000
  #   (accuracy, x) = computeAccuracy(expResults['predictions'][killCellAt:],
  #                                   expResults['truths'][killCellAt:],
  #                                   expResults['iterations'][killCellAt:],
  #                                   resets=expResults['resets'][killCellAt:],
  #                                   randoms=expResults['randoms'][killCellAt:])
  #   accuracyListTM.append(float(numpy.sum(accuracy)) / len(accuracy))
  #
  #   # LSTM experiments
  #   lstmResultDir = "lstm/results/high-order-distributed-random-kill-cell/"
  #   experiment = lstmResultDir + \
  #                "kill_cell_percent{:1.2f}/0.log".format(killCellPercent)
  #
  #   expResults = readExperiment(experiment)
  #
  #   killCellAt = 10000
  #   (accuracy, x) = computeAccuracy(expResults['predictions'][killCellAt:],
  #                                   expResults['truths'][killCellAt:],
  #                                   expResults['iterations'][killCellAt:],
  #                                   resets=expResults['resets'][killCellAt:],
  #                                   randoms=expResults['randoms'][killCellAt:])
  #   accuracyListLSTM.append(float(numpy.sum(accuracy)) / len(accuracy))
  #
  #   # ELM
  #   experiment = 'elm/results/high-order-distributed-random-kill-cell/' \
  #                'kill_cell_percent' + "{:1.2f}".format(killCellPercent) + '/0.log'
  #
  #   expResults = readExperiment(experiment)
  #
  #   killCellAt = 20000
  #   (accuracy, x) = computeAccuracy(expResults['predictions'][killCellAt:],
  #                                   expResults['truths'][killCellAt:],
  #                                   expResults['iterations'][killCellAt:],
  #                                   resets=expResults['resets'][killCellAt:],
  #                                   randoms=expResults['randoms'][killCellAt:])
  #   accuracyListELM.append(float(numpy.sum(accuracy)) / len(accuracy))
  #
  # plt.figure(2)
  # plt.plot(KILLCELL_PERCENT, accuracyListTM, 'r-^', label="HTM")
  # plt.plot(KILLCELL_PERCENT, accuracyListLSTM, 'b-s', label="LSTM")
  # plt.plot(KILLCELL_PERCENT, accuracyListELM, 'g-s', label="ELM")
  # plt.xlabel('Fraction of cell death ')
  # plt.ylabel('Accuracy after cell death')
  # plt.ylim([0.1, 1.05])
  # plt.legend()
  # plt.savefig('./result/model_performance_after_cell_death.pdf')

