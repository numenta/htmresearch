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


import argparse
import json
import os
from matplotlib import pyplot
import numpy
from plot import computeAccuracy, plotAccuracy, movingAverage

pyplot.ion()



def readExperiment(experiments):
  with open(experiments, "r") as file:
    predictions = []
    truths = []
    iterations = []
    resets = []
    randoms = []
    trains = []
    killCell = []
    for line in file.readlines():
      dataRec = json.loads(line)
      iterations.append(dataRec['iteration'])
      predictions.append(dataRec['predictions'])
      truths.append(dataRec['truth'])
      resets.append(dataRec['reset'])
      randoms.append(dataRec['random'])
      trains.append(dataRec['train'])
      killCell.append(dataRec['killCell'])

  return predictions, truths, iterations, resets, randoms, killCell



if __name__ == '__main__':

  outdir = 'tm/result/'
  KILLCELL_PERCENT = list(numpy.arange(10) / 10.0)
  accuracyList = []
  for killCellPercent in KILLCELL_PERCENT:
    experiment = os.path.join(outdir, "kill_cell_percent{:1.1f}".format(
      killCellPercent)) + '/0.log'

    (predictions, truths, iterations,
     resets, randoms, killCell) = readExperiment(experiment)

    killCellAt = 10000
    (accuracy, x) = computeAccuracy(predictions[killCellAt:],
                                    truths[killCellAt:],
                                    iterations[killCellAt:],
                                    resets=resets[killCellAt:],
                                    randoms=randoms[killCellAt:])

    accuracyList.append(float(numpy.sum(accuracy)) / len(accuracy))

  pyplot.figure()
  pyplot.plot(KILLCELL_PERCENT, accuracyList, 'k-o')
  pyplot.xlabel('Fraction of cell death ')
  pyplot.ylabel('Accuracy after cell death')
  pyplot.ylim([-0.05, 1.05])
  pyplot.savefig('./tm/result/model_performance_after_cell_death.pdf')
 