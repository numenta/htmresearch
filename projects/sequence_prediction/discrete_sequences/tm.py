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
Depends on:
- https://github.com/numenta/nupic/pull/2491
- https://github.com/numenta/nupic/pull/2495
"""

import operator
import os
import pickle
import random
import sys
import time

import numpy

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.research.monitor_mixin.trace import CountsTrace

from sequence_generator.sequence_generator import SequenceGenerator



MIN_ORDER = 6
MAX_ORDER = 7
NUM_PREDICTIONS = [1, 2]
NUM_RANDOM = 1
PERTURB_AFTER = 1000

NUM_SYMBOLS = SequenceGenerator.numSymbols(MAX_ORDER, max(NUM_PREDICTIONS))
RANDOM_START = NUM_SYMBOLS
RANDOM_END = NUM_SYMBOLS + 1000

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
          "categoryList": range(max(RANDOM_END, NUM_SYMBOLS)),
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
        "temporalImp": "monitored_tm_py",
        "newSynapseCount": 20,
        "maxSynapsesPerSegment": 128,
        "maxSegmentsPerCell": 128,
        "initialPerm": 0.21,
        "connectedPerm": 0.50,
        "permanenceInc": 0.1,
        "permanenceDec": 0.1,
        "predictedSegmentDecrement": 0.01,
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



def generateSequences(numPredictions, perturbed=False):
  sequences = []

  # # Generated sequences
  # generator = SequenceGenerator(seed=42)

  # for order in xrange(MIN_ORDER, MAX_ORDER+1):
  #   sequences += generator.generate(order, numPredictions)

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

  if numPredictions == 1:
    # Hardcoded set of sequences
    if perturbed:
      sequences = [
        [6, 8, 7, 4, 2, 3, 5],
        [1, 8, 7, 4, 2, 3, 0],
        [6, 3, 4, 2, 7, 8, 0],
        [1, 3, 4, 2, 7, 8, 5],
        [0, 9, 7, 8, 5, 3, 4, 6],
        [2, 9, 7, 8, 5, 3, 4, 1],
        [0, 4, 3, 5, 8, 7, 9, 1],
        [2, 4, 3, 5, 8, 7, 9, 6]
      ]
    else:
      sequences = [
        [6, 8, 7, 4, 2, 3, 0],
        [1, 8, 7, 4, 2, 3, 5],
        [6, 3, 4, 2, 7, 8, 5],
        [1, 3, 4, 2, 7, 8, 0],
        [0, 9, 7, 8, 5, 3, 4, 1],
        [2, 9, 7, 8, 5, 3, 4, 6],
        [0, 4, 3, 5, 8, 7, 9, 6],
        [2, 4, 3, 5, 8, 7, 9, 1]
      ]

  if numPredictions == 2:
    # Hardcoded set of sequences with multiple predictions (2)
    # Make sure to set NUM_PREDICTIONS = 2 above
    if perturbed:
      sequences = [
        [4, 8, 3, 10, 9, 6, 0],
        [4, 8, 3, 10, 9, 6, 7],
        [5, 8, 3, 10, 9, 6, 1],
        [5, 8, 3, 10, 9, 6, 2],
        [4, 6, 9, 10, 3, 8, 2],
        [4, 6, 9, 10, 3, 8, 1],
        [5, 6, 9, 10, 3, 8, 7],
        [5, 6, 9, 10, 3, 8, 0],
        [4, 3, 8, 6, 1, 10, 11, 0],
        [4, 3, 8, 6, 1, 10, 11, 7],
        [5, 3, 8, 6, 1, 10, 11, 9],
        [5, 3, 8, 6, 1, 10, 11, 2],
        [4, 11, 10, 1, 6, 8, 3, 2],
        [4, 11, 10, 1, 6, 8, 3, 9],
        [5, 11, 10, 1, 6, 8, 3, 7],
        [5, 11, 10, 1, 6, 8, 3, 0]
      ]
    else:
      sequences = [
        [4, 8, 3, 10, 9, 6, 1],
        [4, 8, 3, 10, 9, 6, 2],
        [5, 8, 3, 10, 9, 6, 0],
        [5, 8, 3, 10, 9, 6, 7],
        [4, 6, 9, 10, 3, 8, 7],
        [4, 6, 9, 10, 3, 8, 0],
        [5, 6, 9, 10, 3, 8, 2],
        [5, 6, 9, 10, 3, 8, 1],
        [4, 3, 8, 6, 1, 10, 11, 9],
        [4, 3, 8, 6, 1, 10, 11, 2],
        [5, 3, 8, 6, 1, 10, 11, 0],
        [5, 3, 8, 6, 1, 10, 11, 7],
        [4, 11, 10, 1, 6, 8, 3, 7],
        [4, 11, 10, 1, 6, 8, 3, 0],
        [5, 11, 10, 1, 6, 8, 3, 2],
        [5, 11, 10, 1, 6, 8, 3, 9]
      ]

  print "Sequences generated:"
  for sequence in sequences:
    print sequence
  print

  return sequences



def getEncoderMapping(model):
  encoder = model._getEncoder().encoders[0][1]
  mapping = dict()

  for i in range(NUM_SYMBOLS):
    mapping[i] = set(encoder.encode(i).nonzero()[0])

  return mapping



def classify(mapping, activeColumns, numPredictions):
  scores = [(len(encoding & activeColumns), i) for i, encoding in mapping.iteritems()]
  random.shuffle(scores)  # break ties randomly
  print sorted(scores, reverse=True)
  return [i for _, i in sorted(scores, reverse=True)[:numPredictions]]



class Runner(object):

  def __init__(self, numPredictions):
    self.numPredictions = numPredictions

    self.model = ModelFactory.create(MODEL_PARAMS)
    self.model.enableInference({"predictedField": "element"})
    self.shifter = InferenceShifter()
    self.mapping = getEncoderMapping(self.model)

    self.sequences = generateSequences(self.numPredictions)
    self.correct = []
    self.numPredictedActiveCells = []
    self.numPredictedInactiveCells = []
    self.numUnpredictedActiveColumns = []

    self.i = 0

  def step(self):
    if self.i == PERTURB_AFTER:
      self.sequences = generateSequences(self.numPredictions, perturbed=True)

    sequence = random.choice(self.sequences)

    topPredictions = []

    for j, element in enumerate(sequence):
      result = self.shifter.shift(self.model.run({"element": element}))
      # print element, result.inferences["multiStepPredictions"][1]
      tm = self.model._getTPRegion().getSelf()._tfdr

      if j == len(sequence) - 2:
        tm.mmClearHistory()

        # Uncomment to use custom classifier (uses predicted cells to make predictions)
        predictiveColumns = set([tm.columnForCell(cell) for cell in tm.predictiveCells])
        topPredictions = classify(self.mapping, predictiveColumns, self.numPredictions)

      if j == len(sequence) - 1:
        # Uncomment to use CLA classifier's predictions
        # bestPredictions = sorted(result.inferences["multiStepPredictions"][1].items(),
        #                          key=operator.itemgetter(1),
        #                          reverse=True)
        # topPredictions = [int(round(a)) for a, b in bestPredictions[:self.numPredictions]]

        print "Step (numPredictions={0})".format(self.numPredictions)
        print "Sequence: ", sequence
        print "Evaluation:", element, topPredictions, element in topPredictions

        self.correct.append(element in topPredictions)
        self.numPredictedActiveCells.append(len(tm.mmGetTracePredictedActiveCells().data[0]))
        self.numPredictedInactiveCells.append(len(tm.mmGetTracePredictedInactiveCells().data[0]))
        self.numUnpredictedActiveColumns.append(len(tm.mmGetTraceUnpredictedActiveColumns().data[0]))


    # Feed noise
    sequence = range(RANDOM_START, RANDOM_END)
    random.shuffle(sequence)
    sequence = sequence[0:NUM_RANDOM]
    print "Random:", sequence

    print

    for element in sequence:
      self.model.run({"element": element})

    self.i += 1


  def accuracy(self):
    return self.correct



if __name__ == "__main__":
  outdir = sys.argv[1]

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  runners = []

  for numPredictions in NUM_PREDICTIONS:
    runners.append(Runner(numPredictions))

  for i in iter(int, 1):
    for runner in runners:
      runner.step()

    if i % 100 == 0:
      results = [(runner.numPredictions, runner.accuracy()) for runner in runners]

      with open(os.path.join(outdir, "results_{0}".format(int(time.time()))), 'wb') as outfile:
        pickle.dump(results, outfile)

      tmStats = [(runner.numPredictedActiveCells, runner.numPredictedInactiveCells, runner.numUnpredictedActiveColumns) for runner in runners]

      with open(os.path.join(outdir, "tm_stats_{0}".format(int(time.time()))), 'wb') as outfile:
        pickle.dump(tmStats, outfile)
