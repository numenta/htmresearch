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
from matplotlib import pyplot as plt
import matplotlib as mpl

from plot import plotAccuracy
from plot import computeAccuracy
from plot import readExperiment
mpl.rcParams['pdf.fonttype'] = 42
plt.ion()
plt.close('all')


if __name__ == '__main__':

  experiments = []
  experiments.append(os.path.join("tm/results",
                                  "high-order-distributed-random-perturbed",
                                  "0.log"))

  for window in [1000.0, 9000.0]:
    experiments.append(os.path.join("lstm/results",
                                "high-order-distributed-random-perturbed",
                                "seed0.0learning_window{:6.1f}".format(window),
                                "0.log"))


  experiments.append(os.path.join("elm/results",
                                  "high-order-distributed-random-perturbed",
                                  "seed0.0",
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
                 window=200,
                 type=type,
                 label='NoiseExperiment',
                 hideTraining=True,
                 lineSize=1.0)
    # plt.xlim([1200, 1750])
    plt.xlabel('# of sequences seen')

  plt.axvline(x=10000, color='k')
  plt.legend(['HTM', 'LSTM-1000', 'LSTM-9000', 'Online-ELM'], loc=4)
  plt.savefig('./result/model_performance_high_order_prediction.pdf')


