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
import json
import operator
import os
import random
import sys
import time
import numpy
from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory
from htmresearch.data.sequence_generator import SequenceGenerator
from htmresearch.algorithms.faulty_temporal_memory import FaultyTemporalMemory
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin)



class MonitoredTemporalMemory(TemporalMemoryMonitorMixin,
                              FaultyTemporalMemory): pass



from htmresearch.algorithms.faulty_temporal_memory_shim import \
  MonitoredFaultyTPShim
from tm import getEncoderMapping, classify, MODEL_PARAMS, generateSequences

MIN_ORDER = 6
MAX_ORDER = 7
NUM_PREDICTIONS = [1]
NUM_RANDOM = 1
KILLCELLS_AFTER = 10000
KILLCELL_PERCENT = list(numpy.arange(10) / 10.0)
NUM_STEPS = 12000

NUM_SYMBOLS = SequenceGenerator.numSymbols(MAX_ORDER, max(NUM_PREDICTIONS))
RANDOM_START = NUM_SYMBOLS
RANDOM_END = NUM_SYMBOLS + 5000



class Runner(object):
  def __init__(self, numPredictions, killCellPercent, resultsDir):
    self.numPredictions = numPredictions

    if not os.path.exists(resultsDir):
      os.makedirs(resultsDir)

    self.resultsFile = open(os.path.join(resultsDir, "0.log"), 'w')

    self.model = ModelFactory.create(MODEL_PARAMS)

    # a hack to use faulty temporal memory instead
    self.model._getTPRegion().getSelf()._tfdr = MonitoredFaultyTPShim(
      numberOfCols=2048,
      cellsPerColumn=32,
      newSynapseCount=32,
      maxSynapsesPerSegment=128,
      maxSegmentsPerCell=128,
      initialPerm=0.21,
      connectedPerm=0.50,
      permanenceInc=0.10,
      permanenceDec=0.10,
      predictedSegmentDecrement=0.01,
      minThreshold=15,
      activationThreshold=15,
      seed=1960,
    )

    self.model.enableInference({"predictedField": "element"})
    self.shifter = InferenceShifter()
    self.mapping = getEncoderMapping(self.model)

    self.sequences = generateSequences(self.numPredictions)
    self.correct = []
    self.numPredictedActiveCells = []
    self.numPredictedInactiveCells = []
    self.numUnpredictedActiveColumns = []

    self.currentSequence = random.choice(self.sequences)
    self.iteration = 0
    self.perturbed = False
    self.randoms = []

    self.killCellPercent = killCellPercent


  def step(self):
    element = self.currentSequence.pop(0)

    randomFlag = (len(self.currentSequence) == 0)
    self.randoms.append(randomFlag)

    killCell = False

    if len(self.currentSequence) == 0:
      if randomFlag:
        self.currentSequence.append(random.randrange(RANDOM_START, RANDOM_END))

      if self.iteration > KILLCELLS_AFTER and not self.perturbed:
        killCell = True
        tm = self.model._getTPRegion().getSelf()._tfdr
        tm.killCells(percent=self.killCellPercent)

        self.model.disableLearning()
        self.perturbed = True

      sequence = random.choice(self.sequences)

      self.currentSequence += sequence

    result = self.shifter.shift(self.model.run({"element": element}))
    tm = self.model._getTPRegion().getSelf()._tfdr

    tm.mmClearHistory()
    # Use custom classifier (uses predicted cells to make predictions)
    predictiveColumns = set(
      [tm.columnForCell(cell) for cell in tm.predictiveCells])
    topPredictions = classify(self.mapping, predictiveColumns,
                              self.numPredictions)

    truth = None if (self.randoms[-1] or
                     len(self.randoms) >= 2 and self.randoms[-2]) else \
    self.currentSequence[0]

    data = {"iteration": self.iteration,
            "current": element,
            "reset": False,
            "random": randomFlag,
            "train": True,
            "predictions": topPredictions,
            "truth": truth,
            "killCell": killCell}

    self.resultsFile.write(json.dumps(data) + '\n')
    self.resultsFile.flush()

    self.iteration += 1



if __name__ == "__main__":
  outdir = sys.argv[1]

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  runners = []

  numPredictions = NUM_PREDICTIONS[0]

  for killCellPercent in KILLCELL_PERCENT:
    resultsDir = os.path.join(outdir, "kill_cell_percent{:1.1f}".format(
      killCellPercent))
    runners.append(Runner(numPredictions, killCellPercent, resultsDir))

  for i in xrange(NUM_STEPS):
    for runner in runners:
      runner.step()
