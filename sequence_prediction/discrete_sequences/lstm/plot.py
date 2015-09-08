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
import numpy

from suite import Suite



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



def plotAccuracy(results, train, window=100, type="sequences", label=None):
  pyplot.title("High-order prediction")
  pyplot.xlabel("# of elements seen")
  pyplot.ylabel("High-order prediction accuracy over last {0} tested {1}".format(window, type))

  accuracy = results[0]
  x = results[1]
  movingData = movingAverage(accuracy, min(len(accuracy), window))

  pyplot.plot(x, movingData, label=label,
              marker='o', markersize=3, markeredgewidth=0)

  # dX = numpy.array([x[i+1] - x[i] for i in xrange(len(x) - 1)])
  # testEnd = numpy.array(x)[dX > dX.mean()].tolist()
  # testEnd = testEnd + [x[-1]]

  # dX = numpy.insert(dX, 0, 0)
  # testStart = numpy.array(x)[dX > dX.mean()].tolist()
  # testStart = [0] + testStart

  # for line in testStart:
  #   pyplot.axvline(line, color='orange')

  # for i in xrange(len(testStart)):
  #   pyplot.axvspan(testStart[i], testEnd[i], alpha=0.15, facecolor='black')

  for i in xrange(len(train)):
    if train[i]:
      pyplot.axvline(i, color='orange')

  pyplot.xlim(0, x[-1])
  pyplot.ylim(0, 1.001)



def computeAccuracy(predictions, truth, iteration, resets=None, randoms=None):
  accuracy = []
  x = []

  for i in xrange(len(predictions) - 1):
    if truth[i] is None:
      continue

    if resets is not None or randoms is not None:
      if not (resets[i+1] or randoms[i+1]):
        continue

    correct = truth[i] is None or truth[i] in predictions[i]
    accuracy.append(correct)
    x.append(iteration[i])

  return (accuracy, x)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('experiments', metavar='/path/to/experiment /path/...', nargs='+', type=str)
  parser.add_argument('-w', '--window', type=int, default=20)
  parser.add_argument('-f', '--full', action='store_true')

  suite = Suite()
  args = parser.parse_args()

  from pylab import rcParams

  rcParams.update({'figure.autolayout': True})
  rcParams.update({'figure.facecolor': 'white'})
  rcParams.update({'ytick.labelsize': 8})
  rcParams.update({'figure.figsize': (12, 6)})

  experiments = args.experiments

  for experiment in experiments:
    iteration = suite.get_history(experiment, 0, 'iteration')
    predictions = suite.get_history(experiment, 0, 'predictions')
    truth = suite.get_history(experiment, 0, 'truth')
    train = suite.get_history(experiment, 0, 'train')

    resets = None if args.full else suite.get_history(experiment, 0, 'reset')
    randoms = None if args.full else suite.get_history(experiment, 0, 'random')
    type = "elements" if args.full else "sequences"

    plotAccuracy(computeAccuracy(predictions, truth, iteration, resets=resets, randoms=randoms),
                 train,
                 window=args.window,
                 type=type,
                 label=experiment)

  if len(experiments) > 1:
    pyplot.legend()

  pyplot.show()
