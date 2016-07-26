#!/usr/bin/env python
# ----------------------------------------------------------------------
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

import math
import numpy as np

from nupic.algorithms import anomaly_likelihood
from nupic.encoders.date import DateEncoder
from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder

from nab.detectors.base import AnomalyDetector

from nupic.bindings.algorithms import SpatialPooler
from nupic.bindings.experimental import ExtendedTemporalMemory


class DistalTimestamps1CellPerColumnDetector(AnomalyDetector):
  """The 'numenta' detector, with the following changes:

  - Use pure Temporal Memory, not the classic TP that uses backtracking.
  - Don't spatial pool the timestamp. Pass it in as distal input.
  - 1 cell per column.
  - Use w=41 in the scalar encoding, rather than w=21, to make up for the
    lost timestamp input to the spatial pooler.
  """
  def __init__(self, *args, **kwargs):
    super(DistalTimestamps1CellPerColumnDetector, self).__init__(*args,
                                                                 **kwargs)

    self.valueEncoder = None
    self.encodedValue = None
    self.timestampEncoder = None
    self.encodedTimestamp = None
    self.activeExternalCells = []
    self.prevActiveExternalCells = []
    self.sp = None
    self.spOutput = None
    self.etm = None
    self.anomalyLikelihood = None


  def getAdditionalHeaders(self):
    """Returns a list of strings."""
    return ["raw_score"]


  def initialize(self):
    rangePadding = abs(self.inputMax - self.inputMin) * 0.2
    minVal = self.inputMin - rangePadding
    maxVal = (self.inputMax + rangePadding
              if self.inputMin != self.inputMax
              else self.inputMin + 1)
    numBuckets = 130.0
    resolution = max(0.001, (maxVal - minVal) / numBuckets)
    self.valueEncoder = RandomDistributedScalarEncoder(resolution,
                                                       w=41,
                                                       seed=42)
    self.encodedValue = np.zeros(self.valueEncoder.getWidth(),
                                 dtype=np.uint32)

    self.timestampEncoder = DateEncoder(timeOfDay=(21,9.49,))
    self.encodedTimestamp = np.zeros(self.timestampEncoder.getWidth(),
                                     dtype=np.uint32)

    inputWidth = self.valueEncoder.getWidth()

    self.sp = SpatialPooler(**{
      "globalInhibition": True,
      "columnDimensions": [2048],
      "inputDimensions": [inputWidth],
      "potentialRadius": inputWidth,
      "numActiveColumnsPerInhArea": 40,
      "seed": 1956,
      "potentialPct": 0.8,
      "maxBoost": 1.0,
      "synPermActiveInc": 0.003,
      "synPermConnected": 0.2,
      "synPermInactiveDec": 0.0005,
    })
    self.spOutput = np.zeros(2048, dtype=np.float32)

    self.etm = ExtendedTemporalMemory(**{
      "activationThreshold": 13,
      "cellsPerColumn": 1,
      "columnDimensions": (2048,),
      "initialPermanence": 0.21,
      "maxSegmentsPerCell": 128,
      "maxSynapsesPerSegment": 32,
      "minThreshold": 10,
      "maxNewSynapseCount": 20,
      "permanenceDecrement": 0.1,
      "permanenceIncrement": 0.1,
      "seed": 1960,
      "formInternalConnections": True
    })

    learningPeriod = math.floor(self.probationaryPeriod / 2.0)
    self.anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
      claLearningPeriod=learningPeriod,
      estimationSamples=self.probationaryPeriod - learningPeriod,
      reestimationPeriod=100
    )


  def handleRecord(self, inputData):
    """Returns a tuple (anomalyScore, rawScore)."""

    self.valueEncoder.encodeIntoArray(inputData["value"],
                                      self.encodedValue)

    self.timestampEncoder.encodeIntoArray(inputData["timestamp"],
                                          self.encodedTimestamp)
    self.prevActiveExternalCells = self.activeExternalCells
    self.activeExternalCells = sorted(self.encodedTimestamp.nonzero()[0])

    self.sp.compute(self.encodedValue, True, self.spOutput)

    activeColumns = set(self.spOutput.nonzero()[0].tolist())
    prevPredictedColumns = set(self.etm.columnForCell(cell)
                               for cell in self.etm.getPredictiveCells())

    rawScore = (len(activeColumns - prevPredictedColumns) /
                float(len(activeColumns)))
    anomalyScore = self.anomalyLikelihood.anomalyProbability(
      inputData["value"], rawScore, inputData["timestamp"])
    logScore = self.anomalyLikelihood.computeLogLikelihood(anomalyScore)

    self.etm.compute(sorted(activeColumns),
                     self.prevActiveExternalCells,
                     self.activeExternalCells)

    return (logScore, rawScore)
