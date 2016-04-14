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
from matplotlib import pyplot
import matplotlib as mpl
import numpy

from plot import plotAccuracy
from plot import readExperiment
mpl.rcParams['pdf.fonttype'] = 42
pyplot.ion()
pyplot.close('all')

def computeAccuracy(predictions, truths, sequenceCounter):
  accuracy = []
  x = []

  for i in xrange(len(predictions) - 1):
    if truths[i] is None:
      continue

    correct = predictions[i][0] in truths[i]

    accuracy.append(correct)
    x.append(sequenceCounter[i])

  return (accuracy, x)

if __name__ == '__main__':

  experiments = [os.path.join("tm/results", "reber", "0.log"),
                 os.path.join("lstm/results", "reber-distributed", "0.log"),
                 os.path.join("elm/results", "reber-basic", "0.log")]

  for experiment in experiments:
    data = readExperiment(experiment)
    (accuracy, x) = computeAccuracy(data['predictions'],
                                    data['truths'],
                                    data['sequenceCounter'])

    plotAccuracy((accuracy, x),
                 data['trains'],
                 window=100,
                 type=type,
                 label='NoiseExperiment',
                 hideTraining=True,
                 lineSize=1.0)
    pyplot.xlabel('# of sequences seen')

  pyplot.ylabel('Prediction accuracy')
  pyplot.xlim([0, 250])
  pyplot.ylim([0, 1.05])
  pyplot.legend(['HTM', 'LSTM', 'ELM'], loc=4)
  pyplot.savefig('./result/reber_grammar_performance.pdf')