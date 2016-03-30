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
import json

from matplotlib import pyplot
import numpy

from expsuite import PyExperimentSuite


def readExperiment(experiment):
  with open(experiment, "r") as file:
    predictions = []
    predictionsSDR = []
    truths = []
    iterations = []
    resets = []
    randoms = []
    trains = []
    killCell = []
    sequenceCounter = []
    for line in file.readlines():
      dataRec = json.loads(line)
      iterations.append(dataRec['iteration'])

      if 'predictions' in dataRec.keys():
        predictions.append(dataRec['predictions'])
      else:
        predictions.append(None)

      if 'predictionsSDR' in dataRec.keys():
        predictionsSDR.append(dataRec['predictionsSDR'])
      else:
        predictionsSDR.append(None)

      if 'truth' in dataRec.keys():
        truths.append(dataRec['truth'])
      else:
        truths.append(None)

      if 'reset' in dataRec.keys():
        resets.append(dataRec['reset'])
      else:
        resets.append(None)

      if 'random' in dataRec.keys():
        randoms.append(dataRec['random'])
      else:
        randoms.append(None)

      if 'train' in dataRec.keys():
        trains.append(dataRec['train'])
      else:
        trains.append(None)

      if 'killCell' in dataRec.keys():
        killCell.append(dataRec['killCell'])
      else:
        killCell.append(None)

      if 'sequenceCounter' in dataRec.keys():
        sequenceCounter.append(dataRec['sequenceCounter'])
      else:
        sequenceCounter.append(None)

  return {'predictions': predictions,
          'predictionsSDR': predictionsSDR,
          'truths': truths,
          'iterations': iterations,
          'resets': resets,
          'randoms': randoms,
          'trains': trains,
          'killCell': killCell,
          'sequenceCounter': sequenceCounter}

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



def plotAccuracy(results, train, window=100, type="sequences", label=None, hideTraining=True, lineSize=None):
  pyplot.title("High-order prediction")
  pyplot.xlabel("# of sequences seen")
  pyplot.ylabel("High-order prediction accuracy over last {0} tested {1}".format(window, type))

  accuracy = results[0]
  x = results[1]
  movingData = movingAverage(accuracy, min(len(accuracy), window))

  pyplot.plot(x, movingData, label=label, linewidth=lineSize)

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

  if not hideTraining:
    for i in xrange(len(train)):
      if train[i]:
        pyplot.axvline(i, color='orange')

  pyplot.xlim(0, x[-1])
  pyplot.ylim(0, 1.1)



def computeAccuracy(predictions, truths, iterations,
                    resets=None, randoms=None, num=None,
                    sequenceCounter=None):
  accuracy = []
  x = []

  for i in xrange(len(predictions) - 1):
    if num is not None and i > num:
      continue

    if truths[i] is None:
      continue

    # identify the end of sequence
    if resets is not None or randoms is not None:
      if not (resets[i+1] or randoms[i+1]):
        continue

    correct = truths[i] is None or truths[i] in predictions[i]
    accuracy.append(correct)
    if sequenceCounter is not None:
      x.append(sequenceCounter[i])
    else:
      x.append(iterations[i])

  return (accuracy, x)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('experiments', metavar='/path/to/experiment /path/...', nargs='+', type=str)
  parser.add_argument('-w', '--window', type=int, default=100)
  parser.add_argument('-n', '--num', type=int, default=None)
  parser.add_argument('-t', '--training-hide', type=int, nargs='+')
  parser.add_argument('-g', '--graph-labels', type=str, nargs='+')
  parser.add_argument('-s', '--size-of-line', type=float, nargs='+')
  parser.add_argument('-l', '--legend-position', type=int, default=4)
  parser.add_argument('-f', '--full', action='store_true')
  parser.add_argument('-o', '--output', type=str, default=None)

  suite = PyExperimentSuite()
  args = parser.parse_args()

  from pylab import rcParams

  rcParams.update({'figure.autolayout': True})
  rcParams.update({'figure.facecolor': 'white'})
  rcParams.update({'ytick.labelsize': 8})
  rcParams.update({'figure.figsize': (12, 6)})
  rcParams.update({'pdf.fonttype': 42})

  experiments = args.experiments

  for i, experiment in enumerate(experiments):
    iteration = suite.get_history(experiment, 0, 'iteration')
    predictions = suite.get_history(experiment, 0, 'predictions')
    truth = suite.get_history(experiment, 0, 'truth')
    train = suite.get_history(experiment, 0, 'train')

    resets = None if args.full else suite.get_history(experiment, 0, 'reset')
    randoms = None if args.full else suite.get_history(experiment, 0, 'random')
    type = "elements" if args.full else "sequences"

    hideTraining = args.training_hide is not None and len(args.training_hide) > i and args.training_hide[i] > 0
    lineSize = args.size_of_line[i] if args.size_of_line is not None and len(args.size_of_line) > i else 0.8
    label = args.graph_labels[i] if args.graph_labels is not None and len(args.graph_labels) > i else experiment

    plotAccuracy(computeAccuracy(predictions, truth, iteration, resets=resets, randoms=randoms, num=args.num),
                 train,
                 window=args.window,
                 type=type,
                 label=label,
                 hideTraining=hideTraining,
                 lineSize=lineSize)

  if len(experiments) > 1:
    pyplot.legend(loc=args.legend_position)

  if args.output is not None:
    pyplot.savefig(args.output)
  else:
    pyplot.show()
