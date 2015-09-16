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

from matplotlib import pyplot
from matplotlib import pyplot as plt
plt.ion()

from suite import Suite
from errorMetrics import *


def movingAverage(a, n):
  movingAverage = []

  for i in xrange(len(a)):
    start = max(0, i - n)
    values = a[start:i+1]
    movingAverage.append(sum(values) / float(len(values)))

  return movingAverage



def plotMovingAverage(data, window, label=None):
  movingData = movingAverage(data, min(len(data), window))
  style = 'ro' if len(data) < window else ''
  pyplot.plot(range(len(movingData)), movingData, style, label=label)



def plotAccuracy(results, train, window=100, type="sequences", label=None, params=None):
  pyplot.title(label)
  pyplot.xlabel("# of elements seen")
  pyplot.ylabel("NRMSE over last {0} tested {1}".format(window, type))

  accuracy = np.array(results[0])
  x = np.array(results[1])

  accuracy[np.where(x < params['compute_after'])[0]] = np.nan
  movingData = movingAverage(accuracy, min(len(accuracy), window))
  nrmse = np.sqrt(np.array(movingData))

  pyplot.plot(x, nrmse, label=label,
              marker='o', markersize=3, markeredgewidth=0)

  for i in xrange(len(train)):
    if train[i]:
      pyplot.axvline(i, color='orange')

  pyplot.xlim(0, x[-1])
  # pyplot.ylim(0, 1.001)



def computeAccuracy(predictions, truth, iteration, resets=None, randoms=None):

  square_deviation = np.square(predictions-truth)/np.nanvar(truth)
  x = range(0, len(square_deviation))
  # (window_center, nrmse_slide) = NRMSE_sliding(truth, predictions, 480)

  return (square_deviation, x)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('experiments', metavar='/path/to/experiment /path/...', nargs='+', type=str)
  parser.add_argument('-w', '--window', type=int, default=480)
  parser.add_argument('-f', '--full', action='store_true')

  suite = Suite()
  suite.parse_opt()
  suite.parse_cfg()

  args = parser.parse_args()

  from pylab import rcParams

  rcParams.update({'figure.autolayout': True})
  rcParams.update({'figure.facecolor': 'white'})
  rcParams.update({'ytick.labelsize': 8})
  rcParams.update({'figure.figsize': (12, 6)})

  experiments = args.experiments

  for experiment in experiments:

    experiment_name = experiment.split('/')[1]
    params = suite.items_to_params(suite.cfgparser.items(experiment_name))

    iteration = suite.get_history(experiment, 0, 'iteration')
    predictions = suite.get_history(experiment, 0, 'predictions')
    truth = suite.get_history(experiment, 0, 'truth')
    train = suite.get_history(experiment, 0, 'train')

    truth = np.array(truth, dtype=np.float)
    predictions = np.array(predictions, dtype=np.float)

    resets = None if args.full else suite.get_history(experiment, 0, 'reset')
    randoms = None if args.full else suite.get_history(experiment, 0, 'random')
    type = "elements" if args.full else "sequences"

    plotAccuracy(computeAccuracy(predictions, truth, iteration, resets=resets, randoms=randoms),
                 train,
                 window=args.window,
                 type=type,
                 label=experiment_name,
                 params=params)

  if len(experiments) > 1:
    pyplot.legend()

  pyplot.show()
