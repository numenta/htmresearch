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
  weights = numpy.repeat(1.0, n)/n
  return numpy.convolve(a, weights, 'valid')



def plotMovingAverage(data, window, label=None):
  movingData = movingAverage(data, min(len(data), window))
  style = 'ro' if len(data) < window else ''
  pyplot.plot(range(len(movingData)), movingData, style, label=label)



def plotAccuracy(accuracy, iteration, window=100, label=None):
  pyplot.title("High-order prediction")
  pyplot.xlabel("# of elements seen")
  pyplot.ylabel("High-order prediction accuracy over last {0} elements".format(window))

  movingData = movingAverage(accuracy, min(len(accuracy), window))
  x = iteration[:len(movingData)]
  pyplot.plot(x, movingData, label=label,
              marker='o', markersize=3, markeredgewidth=0)



def computeAccuracy(predictions, truth, resets=None, randoms=None):
  accuracy = []

  for i in xrange(len(predictions) - 1):
    if resets is not None or randoms is not None:
      if not (resets[i+1] or randoms[i+1]):
        continue

    correct = truth[i] is None or truth[i] in predictions[i]
    accuracy.append(correct)

  return accuracy



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('experiment', metavar='/path/to/experiment', type=str)
  parser.add_argument('-w', '--window', type=int, default=100)
  parser.add_argument('-e', '--end-of-sequences-only', action='store_true')

  suite = Suite()
  args = parser.parse_args()

  experiment = args.experiment

  iteration = suite.get_history(experiment, 0, 'iteration')
  predictions = suite.get_history(experiment, 0, 'predictions')
  truth = suite.get_history(experiment, 0, 'truth')

  resets = suite.get_history(experiment, 0, 'reset') if args.end_of_sequences_only else None
  randoms = suite.get_history(experiment, 0, 'random') if args.end_of_sequences_only else None

  from pylab import rcParams

  rcParams.update({'figure.autolayout': True})
  rcParams.update({'figure.facecolor': 'white'})
  rcParams.update({'ytick.labelsize': 8})
  rcParams.update({'figure.figsize': (12, 6)})

  plotAccuracy(computeAccuracy(predictions, truth, resets=resets, randoms=randoms),
               iteration,
               window=args.window)
  pyplot.show()
