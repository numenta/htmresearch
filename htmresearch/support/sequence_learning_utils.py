# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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

import numpy as np
from matplotlib import pyplot as plt



def NRMSE(data, pred):
  return np.sqrt(np.nanmean(np.square(pred-data)))/\
         np.nanstd(data)



def NRMSE_sliding(data, pred, windowSize):
  """
  Computing NRMSE in a sliding window
  :param data:
  :param pred:
  :param windowSize:
  :return: (window_center, NRMSE)
  """

  halfWindowSize = int(round(float(windowSize)/2))
  window_center = range(halfWindowSize, len(data)-halfWindowSize, int(round(float(halfWindowSize)/5.0)))
  nrmse = []
  for wc in window_center:
    nrmse.append(NRMSE(data[wc-halfWindowSize:wc+halfWindowSize],
                       pred[wc-halfWindowSize:wc+halfWindowSize]))

  return (window_center, nrmse)



def altMAPE(groundTruth, prediction):
  error = abs(groundTruth - prediction)
  altMAPE = 100.0 * np.sum(error) / np.sum(abs(groundTruth))
  return altMAPE



def MAPE(groundTruth, prediction):
  MAPE = np.nanmean(
    np.abs(groundTruth - prediction)) / np.nanmean(np.abs(groundTruth))
  return MAPE



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



def plotAccuracy(results, truth, train=None, window=100, label=None, params=None, errorType=None,
                 skipRecordNum = 5904):
  plt.title('Prediction Error Over Time')

  error = results[0]

  x = results[1]
  x = x[:len(error)]

  # print results
  # print params['compute_after']
  # if params is not None:
  #   error[np.where(x < params['compute_after'])[0]] = np.nan

  error[:skipRecordNum] = np.nan

  movingData = movingAverage(error, min(len(error), window))

  if errorType == 'square_deviation':
    print label, " Avg NRMSE:", np.sqrt(np.nanmean(error))/np.nanstd(truth)
    avgError = np.sqrt(np.array(movingData))/np.nanstd(truth)
  elif errorType == 'negLL':
    print label, " Avg negLL:", np.nanmean(error)
    avgError = movingData
  elif errorType == 'mape':
    normFactor = np.nanmean(np.abs(truth))
    print label, " MAPE:", np.nanmean(error) / normFactor
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
  return movingData

  # plt.ylim(0, 1.001)

