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
import numpy as np

def movingAverage(a, n):
  movingAverage = []

  for i in xrange(len(a)):
    start = max(0, i - n)
    end = i+1
    #
    # start = i
    # end = min(len(a), i + n)

    # start = max(0, i-n/2)
    # end = min(len(a), i+n/2)
    #
    values = a[start:end]
    movingAverage.append(sum(values) / float(len(values)))

  return movingAverage



def plotMovingAverage(data, window, label=None):
  movingData = movingAverage(data, min(len(data), window))
  style = 'ro' if len(data) < window else ''
  plt.plot(range(len(movingData)), movingData, style, label=label)



def plotAccuracy(results, truth, train=None, window=100, label=None, params=None, errorType=None):
  plt.title('Prediction Error Over Time')

  error = results[0]

  x = results[1]
  x = x[:len(error)]

  # print results
  # print params['compute_after']
  # if params is not None:
  #   error[np.where(x < params['compute_after'])[0]] = np.nan

  error[:5904] = np.nan

  movingData = movingAverage(error, min(len(error), window))

  if errorType == 'square_deviation':
    print label, " Avg NRMSE:", np.sqrt(np.nanmean(error))/np.nanstd(truth)
    meanError = np.sqrt(np.nanmean(error))/np.nanstd(truth)
    avgError = np.sqrt(np.array(movingData))/np.nanstd(truth)
  elif errorType == 'negLL':
    print label, " Avg negLL:", np.nanmean(error)
    meanError = np.nanmean(error)
    avgError = movingData
  elif errorType == 'mape':

    normFactor = np.nanstd(truth)
    print label, " MAPE:", np.nanmean(error)  / normFactor
    meanError = np.nanmean(error) / normFactor
    avgError = movingData / normFactor
  else:
    raise NotImplementedError

  plt.plot(x, avgError, label=label)
  plt.xlabel("# of elements seen")
  plt.ylabel("{0} over last {1} record".format(errorType, window))
  if train is not None:
    for i in xrange(len(train)):
      if train[i]:
        plt.axvline(x[i], color='orange')

  if params is not None:
    if params['perturb_after'] < len(x):
      plt.axvline(x[params['perturb_after']], color='black', linestyle='--')

  plt.xlim(x[0], x[len(x)-1])
  return error

  # plt.ylim(0, 1.001)

def computeAltMAPE(truth, prediction, startFrom=0):
  return np.nanmean(np.abs(truth[startFrom:] - prediction[startFrom:]))/np.nanmean(np.abs(truth[startFrom:]))


def computeNRMSE(truth, prediction, startFrom=0):
  squareDeviation = computeSquareDeviation(prediction, truth)
  squareDeviation[:startFrom] = None
  return np.sqrt(np.nanmean(squareDeviation))/np.nanstd(truth)


def computeSquareDeviation(predictions, truth):

  squareDeviation = np.square(predictions-truth)
  return squareDeviation


def computeLikelihood(predictions, truth, encoder):
  targetDistribution = np.zeros(predictions.shape)
  for i in xrange(len(truth)):
    if not np.isnan(truth[i]) and truth is not None:
      targetDistribution[i, :] = encoder.encode(truth[i])

  # calculate negative log-likelihood
  Likelihood = np.multiply(predictions, targetDistribution)
  Likelihood = np.sum(Likelihood, axis=1)

  minProb = 0.01
  Likelihood[np.where(Likelihood < minProb)[0]] = minProb
  negLL = -np.log(Likelihood)

  return negLL


def computeAbsouteError(predictions, truth):
  return np.abs( (predictions-truth))



class ExperimentResult(object):
  def __init__(self, experiment_name):
    self.name = experiment_name
    self.loadExperiment(experiment_name)
    self.computeError()

  def loadExperiment(self, experiment):
    suite = Suite()
    suite.parse_opt()
    suite.parse_cfg()

    experiment_dir = experiment.split('/')[1]
    params = suite.items_to_params(suite.cfgparser.items(experiment_dir))
    self.params = params

    predictions = suite.get_history(experiment, 0, 'predictions')
    truth = suite.get_history(experiment, 0, 'truth')

    self.iteration = suite.get_history(experiment, 0, 'iteration')
    self.train = suite.get_history(experiment, 0, 'train')

    self.truth = np.array(truth, dtype=np.float)

    if params['output_encoding'] == 'likelihood':
      from nupic.encoders.scalar import ScalarEncoder as NupicScalarEncoder
      self.outputEncoder = NupicScalarEncoder(w=1, minval=0, maxval=40000, n=22, forced=True)
      predictions_np = np.zeros((len(predictions), self.outputEncoder.n))
      for i in xrange(len(predictions)):
        if predictions[i] is not None:
          predictions_np[i, :] = np.array(predictions[i])
      self.predictions = predictions_np
    else:
      self.predictions = np.array(predictions, dtype=np.float)

  def computeError(self):
    if self.params['output_encoding'] == 'likelihood':
      self.errorType = 'negLL'
      self.error = computeLikelihood(self.predictions, self.truth, self.outputEncoder)
    elif self.params['output_encoding'] == None:
      self.errorType = 'square_deviation'
      self.error = computeSquareDeviation(self.predictions, self.truth)

    startAt = max(self.params['compute_after'], self.params['train_at_iteration'])
    self.error[:startAt] = np.nan


def plotLSTMresult(experiment, window, xaxis=None, label=None):
  expResult = ExperimentResult(experiment)

  if xaxis is not None:
    x = xaxis
  else:
    x = range(0, len(expResult.error))

  error = plotAccuracy((expResult.error, x),
                           expResult.truth,
                           train=expResult.train,
                           window=window,
                           label=label,
                           params=expResult.params,
                           errorType=expResult.errorType)
  return (error, expResult)


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
    expResult = ExperimentResult(experiment)

    # use datetime as x-axis
    filePath = './data/' + expResult.params['dataset'] + '.csv'
    data = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['datetime', 'value', 'timeofday', 'dayofweek'])

    x = pd.to_datetime(data['datetime'])
    plotAccuracy((expResult.error, x),
                 expResult.truth,
                 train=expResult.train,
                 window=args.window,
                 label=experiment_name,
                 params=expResult.params,
                 errorType=expResult.errorType)


  if len(experiments) > 1:
    plt.legend()

  plt.show()
