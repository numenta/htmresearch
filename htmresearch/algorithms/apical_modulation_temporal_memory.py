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

"""An implementation of TemporalMemory"""

import numpy as np

from htmresearch.support import numpy_helpers as np2
from nupic.bindings.math import Random, SparseMatrixConnections



class ApicalModulationTemporalMemory(object):
  """
  TemporalMemory with basal and apical connections, and with the ability to
  connect to external cells.

  Basal connections are used to implement traditional Temporal Memory.

  If apical segments are active, the threshold for lateral/basal segment
  activation is lowered.

  This TemporalMemory is unaware of whether its basalInput or apicalInput are
  from internal or external cells. They are just cell numbers. The caller knows
  what these cell numbers mean, but the TemporalMemory doesn't. This allows the
  same code to work for various algorithms.

  To implement sequence memory, use

    basalInputSize = columnCount*cellsPerColumn

  and call compute like this:

    tm.compute(activeColumns, tm.getActiveCells(), tm.getWinnerCells())

  """

  def __init__(self,
               columnCount=2048,
               basalInputSize=0,
               apicalInputSize=0,
               cellsPerColumn=32,
               activationThreshold=13,
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
    @param columnCount (int)
    The number of minicolumns

    @param basalInputSize (sequence)
    The number of bits in the basal input

    @param apicalInputSize (int)
    The number of bits in the apical input

    @param cellsPerColumn (int)
    Number of cells per column

    @param activationThreshold (int)
    If the number of active connected synapses on a segment is at least this
    threshold, the segment is said to be active.

    @param initialPermanence (float)
    Initial permanence of a new synapse

    @param connectedPermanence (float)
    If the permanence value for a synapse is greater than this value, it is said
    to be connected.

    @param minThreshold (int)
    If the number of potential synapses active on a segment is at least this
    threshold, it is said to be "matching" and is eligible for learning.

    @param sampleSize (int)
    How much of the active SDR to sample with synapses.

    @param permanenceIncrement (float)
    Amount by which permanences of synapses are incremented during learning.

    @param permanenceDecrement (float)
    Amount by which permanences of synapses are decremented during learning.

    @param basalPredictedSegmentDecrement (float)
    Amount by which segments are punished for incorrect predictions.

    @param apicalPredictedSegmentDecrement (float)
    Amount by which segments are punished for incorrect predictions.

    @param maxSynapsesPerSegment
    The maximum number of synapses per segment.

    @param seed (int)
    Seed for the random number generator.
    """

    self.columnCount = columnCount
    self.cellsPerColumn = cellsPerColumn
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.minThreshold = minThreshold
    self.sampleSize = sampleSize
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.basalPredictedSegmentDecrement = basalPredictedSegmentDecrement
    self.apicalPredictedSegmentDecrement = apicalPredictedSegmentDecrement
    self.activationThreshold = activationThreshold
    self.maxSynapsesPerSegment = maxSynapsesPerSegment

    self.basalConnections = SparseMatrixConnections(columnCount*cellsPerColumn,
                                                    basalInputSize)
    self.apicalConnections = SparseMatrixConnections(columnCount*cellsPerColumn,
                                                     apicalInputSize)
    self.rng = Random(seed)
    self.activeCells = ()
    self.winnerCells = ()
    self.predictedCells = ()
    self.activeBasalSegments = ()
    self.activeApicalSegments = ()


  def reset(self):
    """
    Clear all cell and segment activity. This has no effect on the subsequent
    predictions or activity.
    """

    self.activeCells = ()
    self.winnerCells = ()
    self.predictedCells = ()
    self.activeBasalSegments = ()
    self.activeApicalSegments = ()


  def compute(self,
              activeColumns,
              basalInput,
              apicalInput=(),
              basalGrowthCandidates=None,
              apicalGrowthCandidates=None,
              learn=True):
    """
    Perform one timestep. Use the basal and apical input to form a set of
    predictions, then activate the specified columns.

    @param activeColumns (numpy array)
    List of active columns

    @param basalInput (numpy array)
    List of active input bits for the basal dendrite segments

    @param apicalInput (numpy array)
    List of active input bits for the apical dendrite segments

    @param basalGrowthCandidates (numpy array or None)
    List of bits that the active cells may grow new basal synapses to.
    If None, the basalInput is assumed to be growth candidates.

    @param apicalGrowthCandidates (numpy array or None)
    List of bits that the active cells may grow new apical synapses to
    If None, the apicalInput is assumed to be growth candidates.

    @param learn (bool)
    Whether to grow / reinforce / punish synapses
    """

    if basalGrowthCandidates is None:
      basalGrowthCandidates = basalInput

    if apicalGrowthCandidates is None:
      apicalGrowthCandidates = apicalInput





    # Calculate predictions for this timestep
    (activeApicalSegments,
     matchingApicalSegments,
     apicalPotentialOverlaps) = self._calculateSegmentActivity(
       self.apicalConnections, apicalInput, self.connectedPermanence,
       self.activationThreshold, self.minThreshold)

    cellsWithActiveApicalSegments = self.apicalConnections.mapSegmentsToCells(
      activeApicalSegments)
    if learn:
        cellsWithActiveApicalSegments = ()
    #if len(cellsWithActiveApicalSegments) > 0:
    #    raise(Exception("Ok"))

    (activeBasalSegments,
     matchingBasalSegments,
     basalPotentialOverlaps) = self._calculateSegmentActivity(
       self.basalConnections, basalInput, self.connectedPermanence,
       self.activationThreshold, self.minThreshold, apicallyActiveCells=cellsWithActiveApicalSegments)


    predictedCells = self._calculatePredictedCells(activeBasalSegments,
                                                   activeApicalSegments)




    # Calculate active cells
    (correctPredictedCells,
     burstingColumns) = np2.setCompare(predictedCells, activeColumns,
                                       predictedCells / self.cellsPerColumn,
                                       rightMinusLeft=True)
    newActiveCells = np.concatenate((correctPredictedCells,
                                     np2.getAllCellsInColumns(
                                       burstingColumns, self.cellsPerColumn)))

    # Calculate learning
    (learningActiveBasalSegments,
     learningMatchingBasalSegments,
     basalSegmentsToPunish,
     newBasalSegmentCells,
     learningCells) = self._calculateBasalLearning(
       activeColumns, burstingColumns, correctPredictedCells,
       activeBasalSegments, matchingBasalSegments, basalPotentialOverlaps)

    (learningActiveApicalSegments,
     learningMatchingApicalSegments,
     apicalSegmentsToPunish,
     newApicalSegmentCells) = self._calculateApicalLearning(
       learningCells, activeColumns, activeApicalSegments,
       matchingApicalSegments, apicalPotentialOverlaps)

    # Learn
    if learn:
      # Learn on existing segments
      for learningSegments in (learningActiveBasalSegments,
                               learningMatchingBasalSegments):
        self._learn(self.basalConnections, self.rng, learningSegments,
                    basalInput, basalGrowthCandidates, basalPotentialOverlaps,
                    self.initialPermanence, self.sampleSize,
                    self.permanenceIncrement, self.permanenceDecrement,
                    self.maxSynapsesPerSegment)

      for learningSegments in (learningActiveApicalSegments,
                               learningMatchingApicalSegments):

        self._learn(self.apicalConnections, self.rng, learningSegments,
                    apicalInput, apicalGrowthCandidates,
                    apicalPotentialOverlaps, self.initialPermanence,
                    self.sampleSize, self.permanenceIncrement,
                    self.permanenceDecrement, self.maxSynapsesPerSegment)

      # Punish incorrect predictions
      if self.basalPredictedSegmentDecrement != 0.0:
        self.basalConnections.adjustActiveSynapses(
          basalSegmentsToPunish, basalInput, -self.basalPredictedSegmentDecrement)

      if self.apicalPredictedSegmentDecrement != 0.0:
        self.apicalConnections.adjustActiveSynapses(
          apicalSegmentsToPunish, apicalInput, -self.apicalPredictedSegmentDecrement)

      # Grow new segments
      if len(basalGrowthCandidates) > 0:
        self._learnOnNewSegments(self.basalConnections, self.rng,
                                 newBasalSegmentCells, basalGrowthCandidates,
                                 self.initialPermanence, self.sampleSize,
                                 self.maxSynapsesPerSegment)

      if len(apicalGrowthCandidates) > 0:
        self._learnOnNewSegments(self.apicalConnections, self.rng,
                                 newApicalSegmentCells, apicalGrowthCandidates,
                                 self.initialPermanence, self.sampleSize,
                                 self.maxSynapsesPerSegment)

    # Save the results
    newActiveCells.sort()
    learningCells.sort()
    self.activeCells = newActiveCells
    self.winnerCells = learningCells
    self.predictedCells = predictedCells
    self.activeBasalSegments = activeBasalSegments
    self.activeApicalSegments = activeApicalSegments


  def _calculateBasalLearning(self,
                              activeColumns,
                              burstingColumns,
                              correctPredictedCells,
                              activeBasalSegments,
                              matchingBasalSegments,
                              basalPotentialOverlaps):
    """
    Basic Temporal Memory learning. Correctly predicted cells always have
    active basal segments, and we learn on these segments. In bursting
    columns, we either learn on an existing basal segment, or we grow a new one.

    The only influence apical dendrites have on basal learning is: the apical
    dendrites influence which cells are considered "predicted". So an active
    apical dendrite can prevent some basal segments in active columns from
    learning.

    @param correctPredictedCells (numpy array)
    @param burstingColumns (numpy array)
    @param activeBasalSegments (numpy array)
    @param matchingBasalSegments (numpy array)
    @param basalPotentialOverlaps (numpy array)

    @return (tuple)
    - learningActiveBasalSegments (numpy array)
      Active basal segments on correct predicted cells

    - learningMatchingBasalSegments (numpy array)
      Matching basal segments selected for learning in bursting columns

    - basalSegmentsToPunish (numpy array)
      Basal segments that should be punished for predicting an inactive column

    - newBasalSegmentCells (numpy array)
      Cells in bursting columns that were selected to grow new basal segments

    - learningCells (numpy array)
      Cells that have learning basal segments or are selected to grow a basal
      segment
    """

    # Correctly predicted columns
    learningActiveBasalSegments = self.basalConnections.filterSegmentsByCell(
      activeBasalSegments, correctPredictedCells)

    cellsForMatchingBasal = self.basalConnections.mapSegmentsToCells(
      matchingBasalSegments)
    matchingCells = np.unique(cellsForMatchingBasal)

    (matchingCellsInBurstingColumns,
     burstingColumnsWithNoMatch) = np2.setCompare(
       matchingCells, burstingColumns, matchingCells / self.cellsPerColumn,
       rightMinusLeft=True)

    learningMatchingBasalSegments = self._chooseBestSegmentPerColumn(
      self.basalConnections, matchingCellsInBurstingColumns,
      matchingBasalSegments, basalPotentialOverlaps, self.cellsPerColumn)
    newBasalSegmentCells = self._getCellsWithFewestSegments(
      self.basalConnections, self.rng, burstingColumnsWithNoMatch,
      self.cellsPerColumn)

    learningCells = np.concatenate(
      (correctPredictedCells,
       self.basalConnections.mapSegmentsToCells(learningMatchingBasalSegments),
       newBasalSegmentCells))

    # Incorrectly predicted columns
    correctMatchingBasalMask = np.in1d(
      cellsForMatchingBasal / self.cellsPerColumn, activeColumns)

    basalSegmentsToPunish = matchingBasalSegments[~correctMatchingBasalMask]

    return (learningActiveBasalSegments,
            learningMatchingBasalSegments,
            basalSegmentsToPunish,
            newBasalSegmentCells,
            learningCells)


  def _calculateApicalLearning(self,
                               learningCells,
                               activeColumns,
                               activeApicalSegments,
                               matchingApicalSegments,
                               apicalPotentialOverlaps):
    """
    Calculate apical learning for each learning cell.

    The set of learning cells was determined completely from basal segments.
    Do all apical learning on the same cells.

    Learn on any active segments on learning cells. For cells without active
    segments, learn on the best matching segment. For cells without a matching
    segment, grow a new segment.

    @param learningCells (numpy array)
    @param correctPredictedCells (numpy array)
    @param activeApicalSegments (numpy array)
    @param matchingApicalSegments (numpy array)
    @param apicalPotentialOverlaps (numpy array)

    @return (tuple)
    - learningActiveApicalSegments (numpy array)
      Active apical segments on correct predicted cells

    - learningMatchingApicalSegments (numpy array)
      Matching apical segments selected for learning in bursting columns

    - apicalSegmentsToPunish (numpy array)
      Apical segments that should be punished for predicting an inactive column

    - newApicalSegmentCells (numpy array)
      Cells in bursting columns that were selected to grow new apical segments
    """

    # Cells with active apical segments
    learningActiveApicalSegments = self.apicalConnections.filterSegmentsByCell(
      activeApicalSegments, learningCells)

    # Cells with matching apical segments
    learningCellsWithoutActiveApical = np.setdiff1d(
      learningCells,
      self.apicalConnections.mapSegmentsToCells(learningActiveApicalSegments))
    cellsForMatchingApical = self.apicalConnections.mapSegmentsToCells(
      matchingApicalSegments)
    learningCellsWithMatchingApical = np.intersect1d(
      learningCellsWithoutActiveApical, cellsForMatchingApical)
    learningMatchingApicalSegments = self._chooseBestSegmentPerCell(
      self.apicalConnections, learningCellsWithMatchingApical,
      matchingApicalSegments, apicalPotentialOverlaps)

    # Cells that need to grow an apical segment
    newApicalSegmentCells = np.setdiff1d(learningCellsWithoutActiveApical,
                                         learningCellsWithMatchingApical)

    # Incorrectly predicted columns
    correctMatchingApicalMask = np.in1d(
      cellsForMatchingApical / self.cellsPerColumn, activeColumns)

    apicalSegmentsToPunish = matchingApicalSegments[~correctMatchingApicalMask]

    return (learningActiveApicalSegments,
            learningMatchingApicalSegments,
            apicalSegmentsToPunish,
            newApicalSegmentCells)


  @staticmethod
  def _calculateSegmentActivity(connections, activeInput, connectedPermanence,
                                activationThreshold, minThreshold, apicallyActiveCells=()):
    """
    Calculate the active and matching segments for this timestep.

    @param connections (SparseMatrixConnections)
    @param activeInput (numpy array)

    @return (tuple)
    - activeSegments (numpy array)
      Dendrite segments with enough active connected synapses to cause a
      dendritic spike

    - matchingSegments (numpy array)
      Dendrite segments with enough active potential synapses to be selected for
      learning in a bursting column

    - potentialOverlaps (numpy array)
      The number of active potential synapses for each segment.
      Includes counts for active, matching, and nonmatching segments.
    """

    # Without tie-breaker:
    # You need to lower the threshold A LOT (.5 or less) to produce increases in the number of predictions...
    # And when you do that, performance decreases to the level of NoFB
    # .75: no visible effect, either on prediction size or performance !
    # Another problem is that w/o tiebreaker, noise doesn't just reduce the number of predictions - sometimes
    # it generates too many predictions ! Presumably from wrongly-bursting columns...?

    # Active
    overlaps = connections.computeActivity(activeInput, connectedPermanence)
    outrightActiveSegments = np.flatnonzero(overlaps >= activationThreshold)
    # We have apicallyActiveCells, but we want segments from the apicallyActiveCells... Or more precisely,
    # conditionallyActiveSegments = np.intersect1d(np.flatnonzero(overlaps >= activationThreshold * 0.75), apicallyActiveCells)  # BUGG !!!
    conditionallyActiveSegments = np.flatnonzero((overlaps < activationThreshold) & (overlaps >= activationThreshold * 0.75))
    cellsOfCASegments = connections.mapSegmentsToCells(conditionallyActiveSegments)
    apicallyActiveSegments = conditionallyActiveSegments[np.in1d(cellsOfCASegments, apicallyActiveCells)] # CA segments from apically active cells

    activeSegments = np.concatenate((outrightActiveSegments, apicallyActiveSegments))

    # To cancel all the apical influence (for debugging):
    # activeSegments = outrightActiveSegments

    #if len(apicallyActiveSegments > 0):
    #    raise("OK")
    # Matching
    potentialOverlaps = connections.computeActivity(activeInput)
    matchingSegments = np.flatnonzero(potentialOverlaps >= minThreshold)

    return (activeSegments,
            matchingSegments,
            potentialOverlaps)


  def _calculatePredictedCells(self, activeBasalSegments, activeApicalSegments):
    """
    Calculate the predicted cells, given the set of active segments.

    An active basal segment is enough to predict a cell.
    An active apical segment is *not* enough to predict a cell.

    When a cell has both types of segments active, other cells in its minicolumn
    must also have both types of segments to be considered predictive.

    @param activeBasalSegments (numpy array)
    @param activeApicalSegments (numpy array)

    @return (numpy array)
    """

    cellsForBasalSegments = self.basalConnections.mapSegmentsToCells(
      activeBasalSegments)
    cellsForApicalSegments = self.apicalConnections.mapSegmentsToCells(
      activeApicalSegments)

    fullyDepolarizedCells = np.intersect1d(cellsForBasalSegments,
                                           cellsForApicalSegments)
    partlyDepolarizedCells = np.setdiff1d(cellsForBasalSegments,
                                          fullyDepolarizedCells)

    inhibitedMask = np.in1d(partlyDepolarizedCells / self.cellsPerColumn,
                            fullyDepolarizedCells / self.cellsPerColumn)
    predictedCells = np.append(fullyDepolarizedCells,
                               partlyDepolarizedCells[~inhibitedMask])



    # If uncommented: Only apical facilitation, no apical tie-breaker:
    #predictedCells = cellsForBasalSegments

    return predictedCells


  @staticmethod
  def _learn(connections, rng, learningSegments, activeInput, growthCandidates,
             potentialOverlaps, initialPermanence, sampleSize,
             permanenceIncrement, permanenceDecrement, maxSynapsesPerSegment):
    """
    Adjust synapse permanences, grow new synapses, and grow new segments.

    @param learningActiveSegments (numpy array)
    @param learningMatchingSegments (numpy array)
    @param activeInput (numpy array)
    @param growthCandidates (numpy array)
    @param potentialOverlaps (numpy array)
    """

    # Learn on existing segments
    connections.adjustSynapses(learningSegments, activeInput,
                               permanenceIncrement, -permanenceDecrement)

    # Grow new synapses. Calculate "maxNew", the maximum number of synapses to
    # grow per segment. "maxNew" might be a number or it might be a list of
    # numbers.
    if sampleSize == -1:
      maxNew = len(growthCandidates)
    else:
      maxNew = sampleSize - potentialOverlaps[learningSegments]

    # NOTE: The following line undoes the previous four, and fixes some buggy behavior! It's a
    # kludge until we figure out what's going on.
    # NOTE 2: There is still some buggy behavior even when this line is on (e.g. no prediction
    # error when swapping ends of sequences, weird oscillations in predictions/active cells, etc.)
    # maxNew = 20

    if maxSynapsesPerSegment != -1:
      synapseCounts = connections.mapSegmentsToSynapseCounts(
        learningSegments)
      numSynapsesToReachMax = maxSynapsesPerSegment - synapseCounts
      maxNew = np.where(maxNew <= numSynapsesToReachMax,
                        maxNew, numSynapsesToReachMax)

    connections.growSynapsesToSample(learningSegments, growthCandidates,
                                     maxNew, initialPermanence, rng)


  @staticmethod
  def _learnOnNewSegments(connections, rng, newSegmentCells, growthCandidates,
                          initialPermanence, sampleSize, maxSynapsesPerSegment):

    numNewSynapses = len(growthCandidates)

    if sampleSize != -1:
      numNewSynapses = min(numNewSynapses, sampleSize)

    if maxSynapsesPerSegment != -1:
      numNewSynapses = min(numNewSynapses, maxSynapsesPerSegment)

    # numNewSynapses = min(20, len(growthCandidates))

    newSegments = connections.createSegments(newSegmentCells)
    connections.growSynapsesToSample(newSegments, growthCandidates,
                                     numNewSynapses, initialPermanence,
                                     rng)


  @classmethod
  def _chooseBestSegmentPerCell(cls,
                                connections,
                                cells,
                                allMatchingSegments,
                                potentialOverlaps):
    """
    For each specified cell, choose its matching segment with largest number
    of active potential synapses. When there's a tie, the first segment wins.

    @param connections (SparseMatrixConnections)
    @param cells (numpy array)
    @param allMatchingSegments (numpy array)
    @param potentialOverlaps (numpy array)

    @return (numpy array)
    One segment per cell
    """

    candidateSegments = connections.filterSegmentsByCell(allMatchingSegments,
                                                         cells)

    # Narrow it down to one pair per cell.
    onePerCellFilter = np2.argmaxMulti(potentialOverlaps[candidateSegments],
                                       connections.mapSegmentsToCells(
                                         candidateSegments))
    learningSegments = candidateSegments[onePerCellFilter]

    return learningSegments


  @classmethod
  def _chooseBestSegmentPerColumn(cls, connections, matchingCells,
                                  allMatchingSegments, potentialOverlaps,
                                  cellsPerColumn):
    """
    For all the columns covered by 'matchingCells', choose the column's matching
    segment with largest number of active potential synapses. When there's a
    tie, the first segment wins.

    @param connections (SparseMatrixConnections)
    @param matchingCells (numpy array)
    @param allMatchingSegments (numpy array)
    @param potentialOverlaps (numpy array)
    """

    candidateSegments = connections.filterSegmentsByCell(allMatchingSegments,
                                                         matchingCells)

    # Narrow it down to one segment per column.
    cellScores = potentialOverlaps[candidateSegments]
    columnsForCandidates = (connections.mapSegmentsToCells(candidateSegments) /
                            cellsPerColumn)
    onePerColumnFilter = np2.argmaxMulti(cellScores, columnsForCandidates)

    learningSegments = candidateSegments[onePerColumnFilter]

    return learningSegments


  @classmethod
  def _getCellsWithFewestSegments(cls, connections, rng, columns,
                                  cellsPerColumn):
    """
    For each column, get the cell that has the fewest total basal segments.
    Break ties randomly.

    @param connections (SparseMatrixConnections)
    @param rng (Random)
    @param columns (numpy array) Columns to check

    @return (numpy array)
    One cell for each of the provided columns
    """
    candidateCells = np2.getAllCellsInColumns(columns, cellsPerColumn)

    # Arrange the segment counts into one row per minicolumn.
    segmentCounts = np.reshape(connections.getSegmentCounts(candidateCells),
                               newshape=(len(columns),
                                         cellsPerColumn))

    # Filter to just the cells that are tied for fewest in their minicolumn.
    minSegmentCounts = np.amin(segmentCounts, axis=1, keepdims=True)
    candidateCells = candidateCells[np.flatnonzero(segmentCounts ==
                                                   minSegmentCounts)]

    # Filter to one cell per column, choosing randomly from the minimums.
    # To do the random choice, add a random offset to each index in-place, using
    # casting to floor the result.
    (_,
     onePerColumnFilter,
     numCandidatesInColumns) = np.unique(candidateCells / cellsPerColumn,
                                         return_index=True, return_counts=True)

    offsetPercents = np.empty(len(columns), dtype="float32")
    rng.initializeReal32Array(offsetPercents)

    np.add(onePerColumnFilter,
           offsetPercents*numCandidatesInColumns,
           out=onePerColumnFilter,
           casting="unsafe")

    return candidateCells[onePerColumnFilter]


  def getActiveCells(self):
    """
    @return (numpy array)
    Active cells
    """
    return self.activeCells


  def getPredictedActiveCells(self):
    """
    @return (numpy array)
    Active cells that were correctly predicted
    """
    return np.intersect1d(self.activeCells, self.predictedCells)


  def getWinnerCells(self):
    """
    @return (numpy array)
    Cells that were selected for learning
    """
    return self.winnerCells


  def getPredictedCells(self):
    """
    @return (numpy array)
    Cells that were predicted for this timestep
    """
    return self.predictedCells


  def getActiveBasalSegments(self):
    """
    @return (numpy array)
    Active basal segments for this timestep
    """
    return self.activeBasalSegments


  def getActiveApicalSegments(self):
    """
    @return (numpy array)
    Matching basal segments for this timestep
    """
    return self.activeApicalSegments


  def numberOfColumns(self):
    """ Returns the number of columns in this layer.

    @return (int) Number of columns
    """
    return self.columnCount


  def numberOfCells(self):
    """
    Returns the number of cells in this layer.

    @return (int) Number of cells
    """
    return self.numberOfColumns() * self.cellsPerColumn


  def getCellsPerColumn(self):
    """
    Returns the number of cells per column.

    @return (int) The number of cells per column.
    """
    return self.cellsPerColumn


  def getActivationThreshold(self):
    """
    Returns the activation threshold.
    @return (int) The activation threshold.
    """
    return self.activationThreshold


  def setActivationThreshold(self, activationThreshold):
    """
    Sets the activation threshold.
    @param activationThreshold (int) activation threshold.
    """
    self.activationThreshold = activationThreshold


  def getInitialPermanence(self):
    """
    Get the initial permanence.
    @return (float) The initial permanence.
    """
    return self.initialPermanence


  def setInitialPermanence(self, initialPermanence):
    """
    Sets the initial permanence.
    @param initialPermanence (float) The initial permanence.
    """
    self.initialPermanence = initialPermanence


  def getMinThreshold(self):
    """
    Returns the min threshold.
    @return (int) The min threshold.
    """
    return self.minThreshold


  def setMinThreshold(self, minThreshold):
    """
    Sets the min threshold.
    @param minThreshold (int) min threshold.
    """
    self.minThreshold = minThreshold


  def getSampleSize(self):
    """
    Gets the sampleSize.
    @return (int)
    """
    return self.sampleSize


  def setSampleSize(self, sampleSize):
    """
    Sets the sampleSize.
    @param sampleSize (int)
    """
    self.sampleSize = sampleSize


  def getPermanenceIncrement(self):
    """
    Get the permanence increment.
    @return (float) The permanence increment.
    """
    return self.permanenceIncrement


  def setPermanenceIncrement(self, permanenceIncrement):
    """
    Sets the permanence increment.
    @param permanenceIncrement (float) The permanence increment.
    """
    self.permanenceIncrement = permanenceIncrement


  def getPermanenceDecrement(self):
    """
    Get the permanence decrement.
    @return (float) The permanence decrement.
    """
    return self.permanenceDecrement


  def setPermanenceDecrement(self, permanenceDecrement):
    """
    Sets the permanence decrement.
    @param permanenceDecrement (float) The permanence decrement.
    """
    self.permanenceDecrement = permanenceDecrement


  def getBasalPredictedSegmentDecrement(self):
    """
    Get the predicted segment decrement.
    @return (float) The predicted segment decrement.
    """
    return self.basalPredictedSegmentDecrement


  def setBasalPredictedSegmentDecrement(self, predictedSegmentDecrement):
    """
    Sets the predicted segment decrement.
    @param predictedSegmentDecrement (float) The predicted segment decrement.
    """
    self.basalPredictedSegmentDecrement = basalPredictedSegmentDecrement


  def getApicalPredictedSegmentDecrement(self):
    """
    Get the predicted segment decrement.
    @return (float) The predicted segment decrement.
    """
    return self.apicalPredictedSegmentDecrement


  def setApicalPredictedSegmentDecrement(self, predictedSegmentDecrement):
    """
    Sets the predicted segment decrement.
    @param predictedSegmentDecrement (float) The predicted segment decrement.
    """
    self.apicalPredictedSegmentDecrement = apicalPredictedSegmentDecrement


  def getConnectedPermanence(self):
    """
    Get the connected permanence.
    @return (float) The connected permanence.
    """
    return self.connectedPermanence


  def setConnectedPermanence(self, connectedPermanence):
    """
    Sets the connected permanence.
    @param connectedPermanence (float) The connected permanence.
    """
    self.connectedPermanence = connectedPermanence
