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
Plot sequence prediction experiment with multiple possible outcomes
"""
import os
from matplotlib import pyplot
import matplotlib as mpl
import numpy

from plot import computeAccuracy
from plot import plotAccuracy
from plot import readExperiment
mpl.rcParams['pdf.fonttype'] = 42
pyplot.ion()
pyplot.close('all')



if __name__ == '__main__':

  experiments = []
  for num_prediction in [2, 4]:
    experiments.append(os.path.join("tm/results",
                                "high-order-distributed-random-multiple-predictions",
                                "num_predictions{:2.1f}".format(num_prediction),
                                "0.log"))

  for num_prediction in [2, 4]:
    experiments.append(os.path.join("lstm/results",
                                "high-order-distributed-random-multiple-predictions",
                                "learning_window9000.0num_predictions{:2.1f}".format(num_prediction),
                                "0.log"))

  for experiment in experiments:
    data = readExperiment(experiment)
    (accuracy, x) = computeAccuracy(data['predictions'],
                                    data['truths'],
                                    data['iterations'],
                                    resets=data['resets'],
                                    randoms=data['randoms'])


    # perturbAt = data['sequenceCounter'][10000]

    plotAccuracy((accuracy, x),
                 data['trains'],
                 window=100,
                 type=type,
                 label='NoiseExperiment',
                 hideTraining=True,
                 lineSize=1.0)
    # pyplot.xlim([1200, 1750])
    pyplot.xlabel('# of elements seen')

  pyplot.legend(['HTM: 2 predictions',
                 'HTM: 4 predictions',
                 'LSTM: 2 predictions',
                 'LSTM: 4 predictions'], loc=4)

  # pyplot.legend(['LSTM', 'HTM'])
  pyplot.savefig('./result/model_performance_multiple_prediction.pdf')