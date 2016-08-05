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
from nupic.encoders.random_distributed_scalar import (
  RandomDistributedScalarEncoder)

from nab.detectors.base import AnomalyDetector

from nupic.bindings.algorithms import SpatialPooler, TemporalMemory


class NumentaTMLowLevelDetector(AnomalyDetector):
  """The 'numentaTM' detector, but not using the CLAModel or network API """
  def __init__(self, *args, **kwargs):
    super(NumentaTMLowLevelDetector, self).__init__(*args, **kwargs)

    self.valueEncoder = None
    self.encodedValue = None
    self.timestampEncoder = None
    self.encodedTimestamp = None
    self.sp = None
    self.spOutput = None
    self.tm = None
    self.anomalyLikelihood = None

    # Set this to False if you want to get results based on raw scores
    # without using AnomalyLikelihood. This will give worse results, but
    # useful for checking the efficacy of AnomalyLikelihood. You will need
    # to re-optimize the thresholds when running with this setting.
    self.useLikelihood = True


  def getAdditionalHeaders(self):
    """Returns a list of strings."""
    return ["raw_score"]


  def initialize(self):

    # Initialize the RDSE with a resolution; calculated from the data min and
    # max, the resolution is specific to the data stream.
    rangePadding = abs(self.inputMax - self.inputMin) * 0.2
    minVal = self.inputMin - rangePadding
    maxVal = (self.inputMax + rangePadding
              if self.inputMin != self.inputMax
              else self.inputMin + 1)
    numBuckets = 130.0
    resolution = max(0.001, (maxVal - minVal) / numBuckets)
    self.valueEncoder = RandomDistributedScalarEncoder(resolution, seed=42)
    self.encodedValue = np.zeros(self.valueEncoder.getWidth(),
                                 dtype=np.uint32)

    # Initialize the timestamp encoder
    self.timestampEncoder = DateEncoder(timeOfDay=(21, 9.49, ))
    self.encodedTimestamp = np.zeros(self.timestampEncoder.getWidth(),
                                     dtype=np.uint32)

    inputWidth = (self.timestampEncoder.getWidth() +
                  self.valueEncoder.getWidth())

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

    self.tm = TemporalMemory(**{
      "activationThreshold": 20,
      "cellsPerColumn": 32,
      "columnDimensions": (2048,),
      "initialPermanence": 0.24,
      "maxSegmentsPerCell": 128,
      "maxSynapsesPerSegment": 128,
      "minThreshold": 13,
      "maxNewSynapseCount": 31,
      "permanenceDecrement": 0.008,
      "permanenceIncrement": 0.04,
      "seed": 1960,
    })

    if self.useLikelihood:
      learningPeriod = math.floor(self.probationaryPeriod / 2.0)
      self.anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
        claLearningPeriod=learningPeriod,
        estimationSamples=self.probationaryPeriod - learningPeriod,
        reestimationPeriod=100
      )


  def handleRecord(self, inputData):
    """Returns a tuple (anomalyScore, rawScore)."""

    # Encode the input data record
    self.valueEncoder.encodeIntoArray(
        inputData["value"], self.encodedValue)
    self.timestampEncoder.encodeIntoArray(
        inputData["timestamp"], self.encodedTimestamp)

    # Run the encoded data through the spatial pooler
    self.sp.compute(np.concatenate((self.encodedTimestamp,
                                    self.encodedValue,)),
                    True, self.spOutput)

    # At the current state, the set of the region's active columns and the set
    # of columns that have previously-predicted cells are used to calculate the
    # raw anomaly score.
    activeColumns = set(self.spOutput.nonzero()[0].tolist())
    prevPredictedColumns = set(self.tm.columnForCell(cell)
                               for cell in self.tm.getPredictiveCells())
    rawScore = (len(activeColumns - prevPredictedColumns) /
                float(len(activeColumns)))

    self.tm.compute(activeColumns)

    if self.useLikelihood:
      # Compute the log-likelihood score
      anomalyScore = self.anomalyLikelihood.anomalyProbability(
        inputData["value"], rawScore, inputData["timestamp"])
      logScore = self.anomalyLikelihood.computeLogLikelihood(anomalyScore)
      return (logScore, rawScore)

    return (rawScore, rawScore)
