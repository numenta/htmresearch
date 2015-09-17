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

from matplotlib import pyplot as plt
plt.ion()

from suite import Suite
from errorMetrics import *
import pandas as pd

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
  plt.plot(range(len(movingData)), movingData, style, label=label)



def plotAccuracy(results, truth, train=None, window=100, label=None, params=None):
  plt.title('Prediction Error Over Time')
  plt.xlabel("# of elements seen")
  plt.ylabel("NRMSE over last {0} tested {1}".format(window, 'elements'))

  square_deviation = results[0]
  x = results[1]
  x = x[:len(square_deviation)]

  if params is not None:
    square_deviation[np.where(x < params['compute_after'])[0]] = np.nan
  movingData = movingAverage(square_deviation, min(len(square_deviation), window))
  nrmse = np.sqrt(np.array(movingData))/np.nanstd(truth)
  print " Avg NRMSE:", np.sqrt(np.nanmean(square_deviation))/np.nanstd(truth)

  plt.plot(x, nrmse, label=label,
              marker='o', markersize=3, markeredgewidth=0)

  if train is not None:
    for i in xrange(len(train)):
      if train[i]:
        plt.axvline(x[i], color='orange')

  if params is not None:
    if params['perturb_after'] < len(x):
      plt.axvline(x[params['perturb_after']], color='black', linestyle='--')

  plt.xlim(x[0], x[len(x)-1])
  
  # plt.ylim(0, 1.001)



def computeAccuracy(predictions, truth, iteration, resets=None, randoms=None):

  square_deviation = np.square(predictions-truth)
  x = range(0, len(square_deviation))
  # (window_center, nrmse_slide) = NRMSE_sliding(truth, predictions, 480)

  return (square_deviation, x)

def loadExperiment(experiment):
  suite = Suite()
  suite.parse_opt()
  suite.parse_cfg()

  experiment_dir = experiment.split('/')[1]
  params = suite.items_to_params(suite.cfgparser.items(experiment_dir))

  iteration = suite.get_history(experiment, 0, 'iteration')
  predictions = suite.get_history(experiment, 0, 'predictions')
  truth = suite.get_history(experiment, 0, 'truth')
  train = suite.get_history(experiment, 0, 'train')

  truth = np.array(truth, dtype=np.float)
  predictions = np.array(predictions, dtype=np.float)

  return (iteration, truth, predictions, train, params)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('experiments', metavar='/path/to/experiment /path/...', nargs='+', type=str)
  parser.add_argument('-w', '--window', type=int, default=480)
  parser.add_argument('-f', '--full', action='store_true')


  args = parser.parse_args()

  from pylab import rcParams

  rcParams.update({'figure.autolayout': True})
  rcParams.update({'figure.facecolor': 'white'})
  rcParams.update({'ytick.labelsize': 8})
  rcParams.update({'figure.figsize': (12, 6)})

  experiments = args.experiments

  for experiment in experiments:
    experiment_name = experiment.split('/')[-2]
    (iteration, truth, predictions, train, params) = loadExperiment(experiment)

    (square_deviation, x) = computeAccuracy(predictions, truth, iteration)

    # use datetime as x-axis
    filePath = './data/' + params['dataset'] + '.csv'
    data = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['datetime', 'value', 'timeofday', 'dayofweek'])

    x = pd.to_datetime(data['datetime'])
    plotAccuracy((square_deviation, x),
                 truth,
                 window=args.window,
                 label=experiment_name,
                 params=params)


  if len(experiments) > 1:
    plt.legend()

  plt.show()
