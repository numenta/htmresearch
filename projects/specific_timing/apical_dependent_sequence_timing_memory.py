"""An adaptation of the ApicalDependentSequenceMemory"""

import numpy as np
from htmresearch.algorithms.apical_dependent_temporal_memory import (ApicalDependentTemporalMemory)



class ApicalDependentSequenceTimingMemory(ApicalDependentTemporalMemory):
  """
  Sequence memory with apical dependence.
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
