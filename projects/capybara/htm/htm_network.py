#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
import htmsanity.nupic.runner as sanity

from nupic.bindings.algorithms import SpatialPooler, TemporalMemory
from nupic.encoders.random_distributed_scalar import (
  RandomDistributedScalarEncoder)



def _computeAnomalyScore(activeColumnsNZ,
                         previouslyPredictiveCellsNZ,
                         nCellsPerColumn):
  """
  The set of the region's active columns and the set of columns that have 
  previously-predicted cells are used to calculate the anomaly score.
  """

  if type(previouslyPredictiveCellsNZ) != np.ndarray:
    previouslyPredictiveCellsNZ = np.array(previouslyPredictiveCellsNZ)

  previouslyPredictiveColumnsNZ = previouslyPredictiveCellsNZ / nCellsPerColumn
  burstingColumnsNZ = np.setdiff1d(activeColumnsNZ,
                                   previouslyPredictiveColumnsNZ)
  return len(burstingColumnsNZ) / float(len(activeColumnsNZ))



def _computePredictedActiveCells(activeCellsNZ,
                                 previouslyPredictiveCellsNZ):
  predictedActive = np.intersect1d(activeCellsNZ, previouslyPredictiveCellsNZ)
  assert len(predictedActive) <= activeCellsNZ
  return predictedActive



class BaseNetwork(object):
  def __init__(self, inputMin=None, inputMax=None, runSanity=False):

    self.inputMin = inputMin
    self.inputMax = inputMax
    self.runSanity = runSanity

    self.encoder = None
    self.encoderOutput = None
    self.sp = None
    self.spOutput = None
    self.spOutputNZ = None
    self.tm = None
    self.anomalyScore = None
    if runSanity:
      self.sanity = None

    self.defaultEncoderResolution = 0.0001
    self.numColumns = 2048
    self.cellsPerColumn = 32

    self.predictedActiveCells = None
    self.previouslyPredictiveCells = None


  def initialize(self):

    # Scalar Encoder
    resolution = self.getEncoderResolution()
    self.encoder = RandomDistributedScalarEncoder(resolution, seed=42)
    self.encoderOutput = np.zeros(self.encoder.getWidth(), dtype=np.uint32)

    # Spatial Pooler
    spInputWidth = self.encoder.getWidth()
    self.spParams = {
      "globalInhibition": True,
      "columnDimensions": [self.numColumns],
      "inputDimensions": [spInputWidth],
      "potentialRadius": spInputWidth,
      "numActiveColumnsPerInhArea": 40,
      "seed": 1956,
      "potentialPct": 0.8,
      "boostStrength": 5.0,
      "synPermActiveInc": 0.003,
      "synPermConnected": 0.2,
      "synPermInactiveDec": 0.0005,
    }
    self.sp = SpatialPooler(**self.spParams)
    self.spOutput = np.zeros(self.numColumns, dtype=np.uint32)

    # Temporal Memory
    self.tmParams = {
      "activationThreshold": 20,
      "cellsPerColumn": self.cellsPerColumn,
      "columnDimensions": (self.numColumns,),
      "initialPermanence": 0.24,
      "maxSegmentsPerCell": 128,
      "maxSynapsesPerSegment": 128,
      "minThreshold": 13,
      "maxNewSynapseCount": 31,
      "permanenceDecrement": 0.008,
      "permanenceIncrement": 0.04,
      "seed": 1960,
    }
    self.tm = TemporalMemory(**self.tmParams)

    # Sanity
    if self.runSanity:
      self.sanity = sanity.SPTMInstance(self.sp, self.tm)


  def handleRecord(self, scalarValue, label=None, skipEncoding=False,
                   learningMode=True):
    """Process one record."""

    if self.runSanity:
      self.sanity.waitForUserContinue()

    # Encode the input data record if it hasn't already been encoded.
    if not skipEncoding:
      self.encodeValue(scalarValue)

    # Run the encoded data through the spatial pooler
    self.sp.compute(self.encoderOutput, learningMode, self.spOutput)
    self.spOutputNZ = self.spOutput.nonzero()[0]

    # WARNING: this needs to happen here, before the TM runs.
    self.previouslyPredictiveCells = self.tm.getPredictiveCells()

    # Run SP output through temporal memory
    self.tm.compute(self.spOutputNZ)
    self.predictedActiveCells = _computePredictedActiveCells(
      self.tm.getActiveCells(), self.previouslyPredictiveCells)

    # Anomaly score
    self.anomalyScore = _computeAnomalyScore(self.spOutputNZ,
                                             self.previouslyPredictiveCells,
                                             self.cellsPerColumn)

    # Run Sanity
    if self.runSanity:
      self.sanity.appendTimestep(self.getEncoderOutputNZ(),
                                 self.getSpOutputNZ(),
                                 self.previouslyPredictiveCells,
                                 {
                                   'value': scalarValue,
                                   'label':label
                                   })


  def encodeValue(self, scalarValue):
    self.encoder.encodeIntoArray(scalarValue, self.encoderOutput)


  def getEncoderResolution(self):
    """
    Compute the Random Distributed Scalar Encoder (RDSE) resolution. It's 
    calculated from the data min and max, specific to the data stream.
    """
    if self.inputMin is None or self.inputMax is None:
      return self.defaultEncoderResolution
    else:
      rangePadding = abs(self.inputMax - self.inputMin) * 0.2
      minVal = self.inputMin - rangePadding
      maxVal = (self.inputMax + rangePadding
                if self.inputMin != self.inputMax
                else self.inputMin + 1)
      numBuckets = 130.0
      return max(self.defaultEncoderResolution, (maxVal - minVal) / numBuckets)


  def getEncoderOutputNZ(self):
    return self.encoderOutput.nonzero()[0]


  def getSpOutputNZ(self):
    return self.spOutputNZ


  def getTmPredictiveCellsNZ(self):
    return self.tm.getPredictiveCells()


  def getTmActiveCellsNZ(self):
    return self.tm.getActiveCells()


  def getTmPredictedActiveCellsNZ(self):
    return self.predictedActiveCells


  def getRawAnomalyScore(self):
    return self.anomalyScore



def example():
  """Simple usage example of BaseNetwork."""

  # Input data params.
  minValue = 0
  amplitude = 10
  maxValue = minValue + amplitude - 1
  numPoints = 1000

  # Create and init network.
  network = BaseNetwork(inputMin=minValue, inputMax=maxValue, runSanity=False)
  network.initialize()

  for i in range(numPoints):
    scalarValue = minValue + (i % amplitude)

    network.encodeValue(scalarValue)
    print 'Scalar Value', scalarValue
    # print 'Encoding', network.getEncoderOutputNZ()

    network.handleRecord(scalarValue, skipEncoding=True)  # already encoded
    # print 'Encoding', network.getEncoderOutputNZ()
    # print 'SP active columns', network.getSpOutputNZ()
    predictedActiveCells = network.getTmPredictedActiveCellsNZ()
    # print 'TM predictive active cells', predictedActiveCells
    print 'Num pred active cells', len(predictedActiveCells)
    print 'Anomaly Score', network.getRawAnomalyScore()
    print ''



if __name__ == '__main__':
  example()
