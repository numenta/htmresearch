# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
A shim for the faulty TP class that transparently implements
FaultyTemporalMemory,

for use with OPF.
"""

import numpy


from htmresearch.algorithms.faulty_temporal_memory import FaultyTemporalMemory
from nupic.algorithms.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin)

class MonitoredFaultyTemporalMemory(TemporalMemoryMonitorMixin,
                              FaultyTemporalMemory): pass

class MonitoredFaultyTPShim(MonitoredFaultyTemporalMemory):
  """
  TP => Temporal Memory shim class.
  """
  def __init__(self,
               numberOfCols=500,
               cellsPerColumn=10,
               initialPerm=0.11,
               connectedPerm=0.50,
               minThreshold=8,
               newSynapseCount=15,
               permanenceInc=0.10,
               permanenceDec=0.10,
               permanenceMax=1.0,
               activationThreshold=12,
               predictedSegmentDecrement=0,
               maxSegmentsPerCell=255,
               maxSynapsesPerSegment=255,
               globalDecay=0.10,
               maxAge=100000,
               pamLength=1,
               verbosity=0,
               outputType="normal",
               seed=42):
    """
    Translate parameters and initialize member variables specific to `TP.py`.
    """
    super(MonitoredFaultyTPShim, self).__init__(
      columnDimensions=(numberOfCols,),
      cellsPerColumn=cellsPerColumn,
      activationThreshold=activationThreshold,
      initialPermanence=initialPerm,
      connectedPermanence=connectedPerm,
      minThreshold=minThreshold,
      maxNewSynapseCount=newSynapseCount,
      permanenceIncrement=permanenceInc,
      permanenceDecrement=permanenceDec,
      predictedSegmentDecrement=predictedSegmentDecrement,
      maxSegmentsPerCell=maxSegmentsPerCell,
      maxSynapsesPerSegment=maxSynapsesPerSegment,
      seed=seed)

    self.infActiveState = {"t": None}


  def compute(self, bottomUpInput, enableLearn, computeInfOutput=None):
    """
    (From `TP.py`)
    Handle one compute, possibly learning.

    @param bottomUpInput     The bottom-up input, typically from a spatial pooler
    @param enableLearn       If true, perform learning
    @param computeInfOutput  If None, default behavior is to disable the inference
                             output when enableLearn is on.
                             If true, compute the inference output
                             If false, do not compute the inference output
    """
    super(MonitoredFaultyTPShim, self).compute(set(bottomUpInput.nonzero()[0]),
                                            learn=enableLearn)
    numberOfCells = self.numberOfCells()

    activeState = numpy.zeros(numberOfCells)
    activeState[self.getCellIndices(self.activeCells)] = 1
    self.infActiveState["t"] = activeState

    output = numpy.zeros(numberOfCells)
    output[self.getCellIndices(self.predictiveCells | self.activeCells)] = 1
    return output


  def topDownCompute(self, topDownIn=None):
    """
    (From `TP.py`)
    Top-down compute - generate expected input given output of the TP

    @param topDownIn top down input from the level above us

    @returns best estimate of the TP input that would have generated bottomUpOut.
    """
    output = numpy.zeros(self.numberOfColumns())
    columns = [self.columnForCell(idx) for idx in self.predictiveCells]
    output[columns] = 1
    return output


  def getActiveState(self):
    activeState = numpy.zeros(self.numberOfCells())
    activeState[self.getCellIndices(self.activeCells)] = 1
    return activeState


  def getPredictedState(self):
    predictedState = numpy.zeros(self.numberOfCells())
    predictedState[self.getCellIndices(self.predictiveCells)] = 1
    return predictedState


  def getLearnActiveStateT(self):
    state = numpy.zeros([self.numberOfColumns(), self.cellsPerColumn])
    return state
