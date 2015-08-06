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

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.research.monitor_mixin.trace import CountsTrace

from sequence_generator import SequenceGenerator



MAX_ORDER = 4
NUM_PREDICTIONS = 1

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
          "categoryList": range(SequenceGenerator.numSymbols(MAX_ORDER,
                                                             NUM_PREDICTIONS)),
          "n": 2048,
          "w": 41
        }
      },
      "sensorAutoReset" : None,
    },
      "spEnable": True,
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
        "newSynapseCount": 35,
        "maxSynapsesPerSegment": 32,
        "maxSegmentsPerCell": 128,
        "initialPerm": 0.41,
        "permanenceInc": 0.1,
        "permanenceDec" : 0.02,
        "globalDecay": 0.0,
        "maxAge": 0,
        "minThreshold": 30,
        "activationThreshold": 30,
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
  generator = SequenceGenerator(seed=42)
  sequences = []

  for order in xrange(MAX_ORDER-1, MAX_ORDER+1):
    sequences += generator.generate(order, NUM_PREDICTIONS)

  for sequence in sequences:
    print sequence
  return sequences


def plotTraces(tm, timestamp=int(time.time())):
  traces = tm.mmGetDefaultTraces()
  traces = [trace for trace in traces if type(trace) is CountsTrace]

  t = len(traces)

  for i in xrange(t):
    trace = traces[i]
    pyplot.subplot(t, 1, i+1)
    pyplot.title(trace.title)
    pyplot.xlim(max(len(trace.data)-500, 0), len(trace.data))
    pyplot.plot(range(len(trace.data)), trace.data)

  pyplot.draw()
  # pyplot.savefig("tm-{0}.png".format(timestamp))


if __name__ == "__main__":
  model = ModelFactory.create(MODEL_PARAMS)
  model.enableInference({"predictedField": "element"})
  shifter = InferenceShifter()

  sequences = generateSequences()
  numCorrect = 0
  accuracy = []

  pyplot.ion()
  pyplot.show()

  from pylab import rcParams

  rcParams.update({'figure.autolayout': True})
  rcParams.update({'figure.facecolor': 'white'})
  rcParams.update({'ytick.labelsize': 8})

  for i in xrange(10000):
    sequence = random.choice(sequences)

    for j, element in enumerate(sequence):
      result = shifter.shift(model.run({"element": element}))
      # print element, result.inferences["multiStepPredictions"][1]

      if j == len(sequence) - 1:
        bestPredictions = sorted(result.inferences["multiStepPredictions"][1].items(),
                                 key=operator.itemgetter(1),
                                 reverse=True)
        topPredictions = [int(round(a)) for a, b in bestPredictions[:NUM_PREDICTIONS]]

        print "Evaluation:", element, topPredictions, element in topPredictions

        if element in topPredictions:
          numCorrect += 1

        accuracy.append(numCorrect / float(i+1))

        if i % 100 == 0:
          rcParams.update({'figure.figsize': (12, 6)})
          pyplot.figure(1)
          pyplot.clf()
          pyplot.plot(range(len(accuracy)), accuracy)
          pyplot.draw()

          rcParams.update({'figure.figsize': (6, 12)})
          pyplot.figure(2)
          pyplot.clf()
          tm = model._getTPRegion().getSelf()._tfdr
          plotTraces(tm)

    model.resetSequenceStates()
