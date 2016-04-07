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
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy
from plot import movingAverage
from plot import computeAccuracy
from plot import readExperiment

mpl.rcParams['pdf.fonttype'] = 42
plt.ion()

if __name__ == '__main__':

  outdir = 'tm/result/'
  KILLCELL_PERCENT = list(numpy.arange(7) / 10.0)
  accuracyListTM = []
  accuracyListLSTM = []

  for killCellPercent in KILLCELL_PERCENT:
    experiment = os.path.join(outdir, "kill_cell_percent{:1.1f}".format(
      killCellPercent)) + '/0.log'

    expResults = readExperiment(experiment)

    killCellAt = 10000
    (accuracy, x) = computeAccuracy(expResults['predictions'][killCellAt:],
                                    expResults['truths'][killCellAt:],
                                    expResults['iterations'][killCellAt:],
                                    resets=expResults['resets'][killCellAt:],
                                    randoms=expResults['randoms'][killCellAt:])
    accuracyListTM.append(float(numpy.sum(accuracy)) / len(accuracy))

    experiment = 'lstm/results/high-order-distributed-random-kill-cell/' \
                 'kill_cell_percent' + "{:1.2f}".format(killCellPercent) + '/0.log'

    expResults = readExperiment(experiment)

    killCellAt = 10000
    (accuracy, x) = computeAccuracy(expResults['predictions'][killCellAt:],
                                    expResults['truths'][killCellAt:],
                                    expResults['iterations'][killCellAt:],
                                    resets=expResults['resets'][killCellAt:],
                                    randoms=expResults['randoms'][killCellAt:])
    accuracyListLSTM.append(float(numpy.sum(accuracy)) / len(accuracy))

  plt.figure(2)
  plt.plot(KILLCELL_PERCENT, accuracyListTM, 'r-^', label="TM")
  plt.plot(KILLCELL_PERCENT, accuracyListLSTM, 'b-s', label="LSTM")
  plt.xlabel('Fraction of cell death ')
  plt.ylabel('Accuracy after cell death')
  plt.ylim([0.1, 1.05])
  plt.legend()
  plt.savefig('./result/model_performance_after_cell_death.pdf')

