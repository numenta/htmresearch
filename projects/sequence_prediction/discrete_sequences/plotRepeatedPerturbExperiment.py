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
from matplotlib import pyplot
import matplotlib as mpl

from plot import plotAccuracy
from plot import computeAccuracy
from plot import readExperiment
mpl.rcParams['pdf.fonttype'] = 42
pyplot.ion()
pyplot.close('all')


if __name__ == '__main__':

  experiments = []
  tmResults = os.path.join("tm/results",
                           "high-order-distributed-random-perturbed")

  for seed in range(4):
    experiments.append(os.path.join(tmResults, "seed" + "{:.1f}".format(seed),
                                    "0.log"))

  for experiment in experiments:
    print experiment
    data = readExperiment(experiment)
    (accuracy, x) = computeAccuracy(data['predictions'],
                                    data['truths'],
                                    data['iterations'],
                                    resets=data['resets'],
                                    randoms=data['randoms'])
    print data['truths'][10012:10020]
    print data['predictions'][10012:10020]
    # perturbAt = data['sequenceCounter'][10000]

    plotAccuracy((accuracy, x),
                 data['trains'],
                 window=200,
                 type=type,
                 label='NoiseExperiment',
                 hideTraining=True,
                 lineSize=1.0)
    # pyplot.xlim([1200, 1750])
    pyplot.xlabel('# of sequences seen')

  pyplot.axvline(x=10000, color='k')
  # pyplot.legend(['HTM', 'LSTM-1000', 'LSTM-3000', 'LSTM-9000'], loc=4)
  pyplot.savefig('./result/model_performance_high_order_prediction.pdf')