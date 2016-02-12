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


import json
import os
from matplotlib import pyplot
import matplotlib as mpl
import numpy
from plot import computeAccuracy
from plot import readExperiment

mpl.rcParams['pdf.fonttype'] = 42
pyplot.ion()

if __name__ == '__main__':

  outdir = 'tm/result/'
  KILLCELL_PERCENT = list(numpy.arange(7) / 10.0)
  accuracyListTM = []
  accuracyListLSTM = []
  for killCellPercent in KILLCELL_PERCENT:
    experiment = os.path.join(outdir, "kill_cell_percent{:1.1f}".format(
      killCellPercent)) + '/0.log'

    (predictions, truths, iterations,
     resets, randoms, trains, killCell) = readExperiment(experiment)

    killCellAt = 10000
    (accuracy, x) = computeAccuracy(predictions[killCellAt:],
                                    truths[killCellAt:],
                                    iterations[killCellAt:],
                                    resets=resets[killCellAt:],
                                    randoms=randoms[killCellAt:])

    accuracyListTM.append(float(numpy.sum(accuracy)) / len(accuracy))

    experiment = 'lstm/results/high-order-distributed-random-kill-cell/' \
                 'kill_cell_percent' + "{:1.2f}".format(killCellPercent) + '/0.log'

    (predictions, truths, iterations,
     resets, randoms, killCell) = readExperiment(experiment)

    killCellAt = 10000
    (accuracy, x) = computeAccuracy(predictions[killCellAt:],
                                    truths[killCellAt:],
                                    iterations[killCellAt:],
                                    resets=resets[killCellAt:],
                                    randoms=randoms[killCellAt:])
    accuracyListLSTM.append(float(numpy.sum(accuracy)) / len(accuracy))

  pyplot.figure()
  pyplot.plot(KILLCELL_PERCENT, accuracyListTM, 'r-^', label="TM")
  pyplot.plot(KILLCELL_PERCENT, accuracyListLSTM, 'b-s', label="LSTM")
  pyplot.xlabel('Fraction of cell death ')
  pyplot.ylabel('Accuracy after cell death')
  pyplot.ylim([0.1, 1.05])
  pyplot.legend()
  pyplot.savefig('./model_performance_after_cell_death.pdf')

