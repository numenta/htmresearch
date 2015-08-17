#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import operator
import random
import time

from matplotlib import pyplot
import numpy

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.research.monitor_mixin.trace import CountsTrace

from sequence_generator import SequenceGenerator



MIN_ORDER = 6
MAX_ORDER = 7
NUM_PREDICTIONS = 2
NUM_RANDOM = 1
PERTURB_AFTER = 1000

RANDOM_RESERVOIR = 1000
NUM_SYMBOLS = SequenceGenerator.numSymbols(MAX_ORDER, NUM_PREDICTIONS)

MODEL_PARAMS = {
  "model": "CLA",
  "version": 1,
  "predictAheadTime": None,
  "modelParams": {
    "inferenceType": "TemporalMultiStep",
    "sensorParams": {
      "verbosity" : 0,
      "encoders": {
        "element": {
          "fieldname": u"element",
          "name": u"element",
          "type": "SDRCategoryEncoder",
          "categoryList": range(NUM_SYMBOLS + RANDOM_RESERVOIR),
          "n": 2048,
          "w": 41
        }
      },
      "sensorAutoReset" : None,
    },
      "spEnable": False,
      "spParams": {
        "spVerbosity" : 0,
        "globalInhibition": 1,
        "columnCount": 2048,
        "inputWidth": 0,
        "numActiveColumnsPerInhArea": 40,
        "seed": 1956,
        "columnDimensions": 0.5,
        "synPermConnected": 0.1,
        "synPermActiveInc": 0.1,
        "synPermInactiveDec": 0.01,
        "maxBoost": 0.0
    },
    "tpEnable" : True,
    "tpParams": {
      "verbosity": 0,
        "columnCount": 2048,
        "cellsPerColumn": 32,
        "inputWidth": 2048,
        "seed": 1960,
        "temporalImp": "tm_py",
        "newSynapseCount": 20,
        "maxSynapsesPerSegment": 128,
        "maxSegmentsPerCell": 128,
        "initialPerm": 0.21,
        "connectedPerm": 0.50,
        "permanenceInc": 0.1,
        "permanenceDec": 0.1,
        "globalDecay": 0.0,
        "maxAge": 0,
        "minThreshold": 15,
        "activationThreshold": 15,
        "outputType": "normal",
        "pamLength": 1,
      },
      "clParams": {
        "implementation": "cpp",
        "regionName" : "CLAClassifierRegion",
        "clVerbosity" : 0,
        "alpha": 0.0001,
        "steps": "1",
      },
      "trainSPNetOnlyIfRequested": False,
    },
}



def generateSequences():
  sequences = []

  # # Generated sequences
  # generator = SequenceGenerator(seed=42)

  # for order in xrange(MIN_ORDER, MAX_ORDER+1):
  #   sequences += generator.generate(order, NUM_PREDICTIONS)

  # # Subutai's sequences
  # """
  # Make sure to change parameter 'categoryList' above to: "categoryList": range(18)
  # """
  # sequences = [
  #   [0, 1, 2, 3, 4, 5],
  #   [6, 3, 2, 5, 1, 7],
  #   [8, 9, 10, 11, 12, 13],
  #   [14, 1, 2, 3, 15, 16],
  #   [17, 4, 2, 3, 1, 5]
  # ]

  # # Two orders of sequences
  # sequences = [
  #   [4, 2, 5, 0],
  #   [4, 5, 2, 3],
  #   [1, 2, 5, 3],
  #   [1, 5, 2, 0],
  #   [5, 3, 6, 2, 0],
  #   [5, 2, 6, 3, 4],
  #   [1, 3, 6, 2, 4],
  #   [1, 2, 6, 3, 0]
  # ]

  # # Two orders of sequences (easier)
  # # """
  # # Make sure to change parameter 'categoryList' above to: "categoryList": range(13)
  # # """
  # sequences = [
  #   [4, 2, 5, 0],
  #   [4, 5, 2, 3],
  #   [1, 2, 5, 3],
  #   [1, 5, 2, 0],
  #   [11, 9, 12, 8, 6],
  #   [11, 8, 12, 9, 10],
  #   [7, 9, 12, 8, 10],
  #   [7, 8, 12, 9, 6]
  # ]

  # # Two orders of sequences (isolating the problem)
  # sequences = [
  #   [1, 5, 2, 0],
  #   [5, 2, 6, 3, 4]
  # ]
  # random.seed(100) # 100 fails, 300 works (results depend on order of training)

  # # Hardcoded set of sequences
  # sequences = [
  #   [6, 8, 7, 4, 2, 3, 0],
  #   [6, 3, 4, 2, 7, 8, 5],
  #   [1, 8, 7, 4, 2, 3, 5],
  #   [1, 3, 4, 2, 7, 8, 0],
  #   [1, 9, 7, 8, 5, 3, 4, 0],
  #   [1, 4, 3, 5, 8, 7, 9, 6],
  #   [2, 9, 7, 8, 5, 3, 4, 6],
  #   [2, 4, 3, 5, 8, 7, 9, 0]
  # ]

  # Hardcoded set of sequences with multiple predictions (2)
  # Make sure to set NUM_PREDICTIONS = 2 above
  sequences = [
    [4, 8, 3, 10, 9, 6, 1],
    [4, 6, 9, 10, 3, 8, 7],
    [4, 8, 3, 10, 9, 6, 2],
    [4, 6, 9, 10, 3, 8, 0],
    [5, 8, 3, 10, 9, 6, 0],
    [5, 6, 9, 10, 3, 8, 2],
    [5, 8, 3, 10, 9, 6, 7],
    [5, 6, 9, 10, 3, 8, 1],
    [4, 3, 8, 6, 1, 10, 11, 9],
    [4, 11, 10, 1, 6, 8, 3, 7],
    [4, 3, 8, 6, 1, 10, 11, 2],
    [4, 11, 10, 1, 6, 8, 3, 0],
    [5, 3, 8, 6, 1, 10, 11, 0],
    [5, 11, 10, 1, 6, 8, 3, 2],
    [5, 3, 8, 6, 1, 10, 11, 7],
    [5, 11, 10, 1, 6, 8, 3, 9]
  ]

  for sequence in sequences:
    print sequence

  return sequences



def movingAverage(a, n):
  weights = numpy.repeat(1.0, n)/n
  return numpy.convolve(a, weights, 'valid')



def plotMovingAverage(data, window):
  movingData = movingAverage(data, min(len(data), window))
  style = 'ro' if len(data) < window else ''
  pyplot.plot(range(len(movingData)), movingData, style)



def plotAccuracy(correct, window=100):
  pyplot.title("Accuracy over window={0}".format(window))
  plotMovingAverage(correct, window)



def plotTMStats(numPredictedActiveCells, numPredictedInactiveCells, numUnpredictedActiveColumns,
                window=100):
  pyplot.subplot(3, 1, 1)
  pyplot.title("# predicted => active cells over window={0}".format(window))
  plotMovingAverage(numPredictedActiveCells, window)

  pyplot.subplot(3, 1, 2)
  pyplot.title("# predicted => inactive cells over window={0}".format(window))
  plotMovingAverage(numPredictedInactiveCells, window)

  pyplot.subplot(3, 1, 3)
  pyplot.title("# unpredicted => active cells over window={0}".format(window))
  plotMovingAverage(numUnpredictedActiveColumns, window)



def plotTraces(tm, timestamp=int(time.time()), window=500):
  """
  Have to make the following change in NuPIC for this to work:

  --- a/nupic/research/TP_shim.py
  +++ b/nupic/research/TP_shim.py
  @@ -27,10 +27,13 @@ for use with OPF.
   import numpy

   from nupic.research.temporal_memory import TemporalMemory
  +from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  +  TemporalMemoryMonitorMixin)
  +class MonitoredTemporalMemory(TemporalMemoryMonitorMixin, TemporalMemory): pass



  -class TPShim(TemporalMemory):
  +class TPShim(MonitoredTemporalMemory):
  """
  traces = tm.mmGetDefaultTraces()
  traces = [trace for trace in traces if type(trace) is CountsTrace]

  t = len(traces)

  for i in xrange(t):
    trace = traces[i]
    pyplot.subplot(t, 1, i+1)
    pyplot.title(trace.title)
    pyplot.xlim(max(len(trace.data)-window, 0), len(trace.data))
    pyplot.plot(range(len(trace.data)), trace.data)

  pyplot.draw()
  # pyplot.savefig("tm-{0}.png".format(timestamp))



def getEncoderMapping(model):
  encoder = model._getEncoder().encoders[0][1]
  mapping = dict()

  for i in range(NUM_SYMBOLS):
    mapping[i] = set(encoder.encode(i).nonzero()[0])

  return mapping



def classify(mapping, activeColumns, numPredictions=NUM_PREDICTIONS):
  scores = [(len(encoding & activeColumns), i) for i, encoding in mapping.iteritems()]
  random.shuffle(scores)  # break ties randomly
  print sorted(scores, reverse=True)
  return [i for _, i in sorted(scores, reverse=True)[:numPredictions]]



if __name__ == "__main__":
  model = ModelFactory.create(MODEL_PARAMS)
  model.enableInference({"predictedField": "element"})
  shifter = InferenceShifter()
  mapping = getEncoderMapping(model)

  sequences = generateSequences()
  correct = []
  numPredictedActiveCells = []
  numPredictedInactiveCells = []
  numUnpredictedActiveColumns = []

  pyplot.ion()
  pyplot.show()

  from pylab import rcParams

  rcParams.update({'figure.autolayout': True})
  rcParams.update({'figure.facecolor': 'white'})
  rcParams.update({'ytick.labelsize': 8})

  for i in xrange(100000000):
    sequence = random.choice(sequences)

    if i > PERTURB_AFTER:
      sequence = list(reversed(sequence))

    topPredictions = []

    for j, element in enumerate(sequence):
      result = shifter.shift(model.run({"element": element}))
      # print element, result.inferences["multiStepPredictions"][1]
      tm = model._getTPRegion().getSelf()._tfdr

      if j == len(sequence) - 2:
        tm.mmClearHistory()

        # Uncomment to use custom classifier (uses predicted cells to make predictions)
        predictiveColumns = set([tm.columnForCell(cell) for cell in tm.predictiveCells])
        topPredictions = classify(mapping, predictiveColumns)

      if j == len(sequence) - 1:
        # Uncomment to use CLA classifier's predictions
        # bestPredictions = sorted(result.inferences["multiStepPredictions"][1].items(),
        #                          key=operator.itemgetter(1),
        #                          reverse=True)
        # topPredictions = [int(round(a)) for a, b in bestPredictions[:NUM_PREDICTIONS]]

        print "Sequence: ", sequence
        print "Evaluation:", element, topPredictions, element in topPredictions

        correct.append(element in topPredictions)
        numPredictedActiveCells.append(len(tm.mmGetTracePredictedActiveCells().data[0]))
        numPredictedInactiveCells.append(len(tm.mmGetTracePredictedInactiveCells().data[0]))
        numUnpredictedActiveColumns.append(len(tm.mmGetTraceUnpredictedActiveColumns().data[0]))

        if i % 100 == 0:
          rcParams.update({'figure.figsize': (12, 6)})
          pyplot.figure(1)
          pyplot.clf()
          plotAccuracy(correct)
          pyplot.draw()

          # rcParams.update({'figure.figsize': (6, 12)})
          # pyplot.figure(2)
          # pyplot.clf()
          # tm = model._getTPRegion().getSelf()._tfdr
          # plotTraces(tm)

          pyplot.figure(2)
          pyplot.clf()
          plotTMStats(numPredictedActiveCells, numPredictedInactiveCells, numUnpredictedActiveColumns)
          pyplot.draw()

    # Feed noise
    sequence = range(NUM_SYMBOLS, NUM_SYMBOLS + RANDOM_RESERVOIR)
    random.shuffle(sequence)
    sequence = sequence[0:NUM_RANDOM]
    print "Random:", sequence

    for element in sequence:
      model.run({"element": element})

    print
