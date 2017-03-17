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


import operator

import numpy as np

from htmresearch.support import numpy_helpers as np2
from nupic.bindings.math import Random, SparseMatrixConnections


EMPTY_UINT_ARRAY = np.array((), dtype="uint32")


class TemporalMemory(object):
  """
  TemporalMemory with basal and apical connections, and with the ability to
  connect to external cells.

  Basal connections are used to implement traditional Temporal Memory.

  The apical connections are used for further disambiguation. If multiple cells
  in a minicolumn have active basal segments, each of those cells is predicted,
  unless one of them also has an active apical segment, in which case only the
  cells with active basal and apical segments are predicted.

  In other words, the apical connections have no effect unless the basal input
  is a union of SDRs (e.g. from bursting minicolumns).

  This TemporalMemory is unaware of whether its basalInput or apicalInput are
  from internal or external cells. They are just cell numbers. The caller knows
  what these cell numbers mean, but the TemporalMemory doesn't. This allows the
  same code to work for various algorithms.

  To implement sequence memory, use

    basalInputDimensions=(numColumns*cellsPerColumn,)

  and call compute like this:

    tm.compute(activeColumns, tm.getActiveCells(), tm.getWinnerCells())

  """

  def __init__(self,
               columnDimensions=(2048,),
               basalInputDimensions=(),
               apicalInputDimensions=(),
               cellsPerColumn=32,
               activationThreshold=13,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               minThreshold=10,
               sampleSize=20,
               permanenceIncrement=0.1,
               permanenceDecrement=0.1,
               predictedSegmentDecrement=0.0,
               maxNewSynapseCount=None,
               maxSynapsesPerSegment=-1,
               maxSegmentsPerCell=None,
               seed=42):

    self.columnDimensions = columnDimensions
    self.numColumns = self._numPoints(columnDimensions)
    self.basalInputDimensions = basalInputDimensions
    self.apicalInputDimensions = apicalInputDimensions

    self.cellsPerColumn = cellsPerColumn
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.minThreshold = minThreshold

    self.sampleSize = sampleSize
    if maxNewSynapseCount is not None:
      print "Parameter 'maxNewSynapseCount' is deprecated. Use 'sampleSize'."
      self.sampleSize = maxNewSynapseCount

    if maxSegmentsPerCell is not None:
      print "Warning: ignoring parameter 'maxSegmentsPerCell'"

    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.predictedSegmentDecrement = predictedSegmentDecrement
    self.activationThreshold = activationThreshold
    self.maxSynapsesPerSegment = maxSynapsesPerSegment

    self.basalConnections = SparseMatrixConnections(
      self.numColumns*cellsPerColumn, self._numPoints(basalInputDimensions))
    self.apicalConnections = SparseMatrixConnections(
      self.numColumns*cellsPerColumn, self._numPoints(apicalInputDimensions))
    self.rng = Random(seed)
    self.activeCells = EMPTY_UINT_ARRAY
    self.winnerCells = EMPTY_UINT_ARRAY
    self.prevPredictedCells = EMPTY_UINT_ARRAY
    self.activeBasalSegments = EMPTY_UINT_ARRAY
    self.activeApicalSegments = EMPTY_UINT_ARRAY


  def reset(self):
    self.activeCells = EMPTY_UINT_ARRAY
    self.winnerCells = EMPTY_UINT_ARRAY
    self.prevPredictedCells = EMPTY_UINT_ARRAY
    self.activeBasalSegments = EMPTY_UINT_ARRAY
    self.activeApicalSegments = EMPTY_UINT_ARRAY


  def compute(self,
              activeColumns,
              basalInput,
              apicalInput=EMPTY_UINT_ARRAY,
              basalGrowthCandidates=None,
              apicalGrowthCandidates=None,
              learn=True):
    """
    @param activeColumns (numpy array)
    @param basalInput (numpy array)
    @param basalGrowthCandidates (numpy array)
    @param apicalInput (numpy array)
    @param apicalGrowthCandidates (numpy array)
    @param learn (bool)
    """
    if basalGrowthCandidates is None:
      basalGrowthCandidates = basalInput

    if apicalGrowthCandidates is None:
      apicalGrowthCandidates = apicalInput

    # Calculate predictions for this timestep
    (activeBasalSegments,
     matchingBasalSegments,
     basalPotentialOverlaps) = self._calculateSegmentActivity(
       self.basalConnections, basalInput, self.connectedPermanence,
       self.activationThreshold, self.minThreshold)

    (activeApicalSegments,
     matchingApicalSegments,
     apicalPotentialOverlaps) = self._calculateSegmentActivity(
       self.apicalConnections, apicalInput, self.connectedPermanence,
       self.activationThreshold, self.minThreshold)

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
      if self.predictedSegmentDecrement != 0.0:
        self.basalConnections.adjustActiveSynapses(
          basalSegmentsToPunish, basalInput, -self.predictedSegmentDecrement)
        self.apicalConnections.adjustActiveSynapses(
          apicalSegmentsToPunish, apicalInput, -self.predictedSegmentDecrement)

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
    self.prevPredictedCells = predictedCells
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
    apical dendrite can keep some basal segments in active columns from
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
                                activationThreshold, minThreshold):
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

    # Active
    overlaps = connections.computeActivity(activeInput, connectedPermanence)
    activeSegments = np.flatnonzero(overlaps >= activationThreshold)

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


  @staticmethod
  def _numPoints(dimensions):
    """
    Get the number of discrete points in a set of dimensions.

    @param dimensions (sequence of integers)
    @return (int)
    """
    if len(dimensions) == 0:
      return 0
    else:
      return reduce(operator.mul, dimensions, 1)


  def getActiveCells(self):
    return self.activeCells


  def getWinnerCells(self):
    return self.winnerCells


  def getPreviouslyPredictedCells(self):
    return self.prevPredictedCells


  def getActiveBasalSegments(self):
    return self.activeBasalSegments


  def getActiveApicalSegments(self):
    return self.activeApicalSegments
