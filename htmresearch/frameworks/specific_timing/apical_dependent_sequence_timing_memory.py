# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
from htmresearch.algorithms.apical_dependent_temporal_memory import (ApicalDependentTemporalMemory)


class ApicalDependentSequenceTimingMemory(ApicalDependentTemporalMemory):
  """
  This is the ApicalDependentSequenceMemory with an extra function called apicalCheck.
  apicalCheck retrieves basally predicted cells with apically active segments for any apical input.

  This is currently used to report apical activity for apical inputs (temporally) preceding the apical
  input used for the apical dependent sequence memory.
  """


  def __init__(self,
               columnCount=2048,
               apicalInputSize=0,
               cellsPerColumn=32,
               activationThreshold=13,
               reducedBasalThreshold=13,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               minThreshold=10,
               sampleSize=20,
               permanenceIncrement=0.1,
               permanenceDecrement=0.1,
               basalPredictedSegmentDecrement=0.0,
               apicalPredictedSegmentDecrement=0.0,
               maxSynapsesPerSegment=-1,
               seed=42):
    """

    :param columnCount: The number of minicolumns
    :param apicalInputSize: The number of bits in the apical input
    :param cellsPerColumn: Number of cells per column
    :param activationThreshold: If the number of active connected synapses on a segment is at least this
                                threshold, the segment is said to be active.
    :param reducedBasalThreshold: The activation threshold of basal (lateral) segments for cells that have
                                  active apical segments. If equal to activationThreshold (default), this
                                  parameter has no effect.
    :param initialPermanence: Initial permanence of a new synapse
    :param connectedPermanence: If the permanence value for a synapse is greater than this value, it is said
                                to be connected.
    :param minThreshold: If the number of potential synapses active on a segment is at least this
                         threshold, it is said to be "matching" and is eligible for learning.
    :param sampleSize: How much of the active SDR to sample with synapses.
    :param permanenceIncrement: Amount by which permanences of synapses are incremented during learning.
    :param permanenceDecrement: Amount by which permanences of synapses are decremented during learning.
    :param basalPredictedSegmentDecrement: Amount by which basal segments are punished for incorrect predictions.
    :param apicalPredictedSegmentDecrement: Amount by which apical segments are punished for incorrect predictions.
    :param maxSynapsesPerSegment: The maximum number of synapses per segment.
    :param seed: Seed for the random number generator.
    """
    params = {
      "columnCount": columnCount,
      "basalInputSize": columnCount * cellsPerColumn,
      "apicalInputSize": apicalInputSize,
      "cellsPerColumn": cellsPerColumn,
      "activationThreshold": activationThreshold,
      "reducedBasalThreshold": reducedBasalThreshold,
      "initialPermanence": initialPermanence,
      "connectedPermanence": connectedPermanence,
      "minThreshold": minThreshold,
      "sampleSize": sampleSize,
      "permanenceIncrement": permanenceIncrement,
      "permanenceDecrement": permanenceDecrement,
      "basalPredictedSegmentDecrement": basalPredictedSegmentDecrement,
      "apicalPredictedSegmentDecrement": apicalPredictedSegmentDecrement,
      "maxSynapsesPerSegment": maxSynapsesPerSegment,
      "seed": seed,
    }

    super(ApicalDependentSequenceTimingMemory, self).__init__(**params)

    self.prevApicalInput = np.empty(0, dtype="uint32")
    self.prevApicalGrowthCandidates = np.empty(0, dtype="uint32")

    self.prevPredictedCells = np.empty(0, dtype="uint32")


  def reset(self):
    """
    Clear all cell and segment activity.
    """
    super(ApicalDependentSequenceTimingMemory, self).reset()

    self.prevApicalInput = np.empty(0, dtype="uint32")
    self.prevApicalGrowthCandidates = np.empty(0, dtype="uint32")

    self.prevPredictedCells = np.empty(0, dtype="uint32")


  def compute(self,
              activeColumns,
              apicalInput=(),
              apicalGrowthCandidates=None,
              learn=True):
    """
    Perform one timestep. Activate the specified columns, using the predictions
    from the previous timestep, then learn. Then form a new set of predictions
    using the new active cells and the apicalInput.

    @param activeColumns (numpy array)
    List of active columns

    @param apicalInput (numpy array)
    List of active input bits for the apical dendrite segments

    @param apicalGrowthCandidates (numpy array or None)
    List of bits that the active cells may grow new apical synapses to
    If None, the apicalInput is assumed to be growth candidates.

    @param learn (bool)
    Whether to grow / reinforce / punish synapses
    """
    activeColumns = np.asarray(activeColumns)
    apicalInput = np.asarray(apicalInput)

    if apicalGrowthCandidates is None:
      apicalGrowthCandidates = apicalInput
    apicalGrowthCandidates = np.asarray(apicalGrowthCandidates)

    self.prevPredictedCells = self.predictedCells

    self.activateCells(activeColumns, self.activeCells, self.prevApicalInput,
                       self.winnerCells, self.prevApicalGrowthCandidates, learn)

    self.depolarizeCells(self.activeCells, apicalInput, learn)

    self.prevApicalInput = apicalInput.copy()
    self.prevApicalGrowthCandidates = apicalGrowthCandidates.copy()


  def apicalCheck(self, apicalInput):
    """
    Return 'recent' apically predicted cells for each tick  of apical timer
    - finds active apical segments corresponding to predicted basal segment,

    @param apicalInput (numpy array)
    List of active input bits for the apical dendrite segments
    """
    # Calculate predictions for this timestep
    (activeApicalSegments, matchingApicalSegments,
     apicalPotentialOverlaps) = self._calculateSegmentActivity(
       self.apicalConnections, apicalInput, self.connectedPermanence,
       self.activationThreshold, self.minThreshold, self.reducedBasalThreshold)

    apicallySupportedCells = self.apicalConnections.mapSegmentsToCells(
      activeApicalSegments)

    predictedCells = np.intersect1d(
      self.basalConnections.mapSegmentsToCells(self.activeBasalSegments),
      apicallySupportedCells)

    return predictedCells


  def getPredictedCells(self):
    """
    @return (numpy array)
    The prediction from the previous timestep
    """
    return self.prevPredictedCells


  def getNextPredictedCells(self):
    """
    @return (numpy array)
    The prediction for the next timestep
    """
    return self.predictedCells


  def getNextBasalPredictedCells(self):
    """
    @return (numpy array)
    Cells with active basal segments
    """
    return np.unique(
      self.basalConnections.mapSegmentsToCells(self.activeBasalSegments))


  def getNextApicalPredictedCells(self):
    """
    @return (numpy array)
    Cells with active apical segments
    """
    return np.unique(
      self.apicalConnections.mapSegmentsToCells(self.activeApicalSegments))
