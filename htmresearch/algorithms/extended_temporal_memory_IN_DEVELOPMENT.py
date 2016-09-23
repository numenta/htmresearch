# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014-2016, Numenta, Inc.  Unless you have an agreement
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
Extended Temporal Memory implementation in Python.
"""

from collections import defaultdict
import itertools
import operator


from nupic.bindings.math import Random
from nupic.research.connections import Connections, binSearch
from nupic.support.group_by import groupby2

EPSILON = 0.00001 # constant error threshold to check equality of permanences to
                  # other floats
MIN_PREDICTIVE_THRESHOLD = 2



class ExtendedTemporalMemory(object):
  """ Class implementing the Temporal Memory algorithm. """

  def __init__(self,
               columnDimensions=(2048,),
               basalInputDimensions=(),
               apicalInputDimensions=(),
               cellsPerColumn=32,
               activationThreshold=13,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               minThreshold=10,
               maxNewSynapseCount=20,
               permanenceIncrement=0.10,
               permanenceDecrement=0.10,
               predictedSegmentDecrement=0.0,
               formInternalBasalConnections=True,
               learnOnOneCell=False,
               maxSegmentsPerCell=255,
               maxSynapsesPerSegment=255,
               seed=42,
               checkInputs=True):
    """
    @param columnDimensions (list)
    Dimensions of the column space

    @param basalInputDimensions (list)
    Dimensions of the external basal input.

    @param apicalInputDimensions (list)
    Dimensions of the external apical input.

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

    @param maxNewSynapseCount (int)
    The maximum number of synapses added to a segment during learning.

    @param permanenceIncrement (float)
    Amount by which permanences of synapses are incremented during learning.

    @param permanenceDecrement (float)
    Amount by which permanences of synapses are decremented during learning.

    @param predictedSegmentDecrement (float)
    Amount by which segments are punished for incorrect predictions.

    @param formInternalBasalConnections (boolean)
    If True, the winner cell for each column will be fixed between resets.

    @param learnOnOneCell (boolean)
    If True, the winner cell for each column will be fixed between resets.

    @param maxSegmentsPerCell
    The maximum number of segments per cell.

    @param maxSynapsesPerSegment
    The maximum number of synapses per segment.

    @param seed (int)
    Seed for the random number generator.

    """
    # Error checking
    if not len(columnDimensions):
      raise ValueError("Number of column dimensions must be greater than 0")

    if cellsPerColumn <= 0:
      raise ValueError("Number of cells per column must be greater than 0")

    if minThreshold > activationThreshold:
      raise ValueError(
        "The min threshold can't be greater than the activation threshold")

    self.columnDimensions = columnDimensions
    self._numColumns = _numPoints(columnDimensions)

    self.basalInputDimensions = basalInputDimensions
    self._numBasalInputs = _numPoints(basalInputDimensions)

    self.apicalInputDimensions = apicalInputDimensions
    self._numApicalInputs = _numPoints(apicalInputDimensions)

    self.cellsPerColumn = cellsPerColumn
    self.activationThreshold = activationThreshold
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.minThreshold = minThreshold
    self.maxNewSynapseCount = maxNewSynapseCount
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.predictedSegmentDecrement = predictedSegmentDecrement
    self.formInternalBasalConnections = formInternalBasalConnections
    self.learnOnOneCell = learnOnOneCell
    self.checkInputs = checkInputs

    self.basalConnections = Connections(
      self.numberOfCells(),
      maxSegmentsPerCell=maxSegmentsPerCell,
      maxSynapsesPerSegment=maxSynapsesPerSegment)
    self.apicalConnections = Connections(self.numberOfCells(),
      maxSegmentsPerCell=maxSegmentsPerCell,
      maxSynapsesPerSegment=maxSynapsesPerSegment)
    self._random = Random(seed)

    self.activeCells = []
    self.winnerCells = []
    self.activeBasalSegments = []
    self.matchingBasalSegments = []
    self.activeApicalSegments = []
    self.matchingApicalSegments = []

    self.numActiveConnectedSynapsesForBasalSegment = []
    self.numActivePotentialSynapsesForBasalSegment = []
    self.numActiveConnectedSynapsesForApicalSegment = []
    self.numActivePotentialSynapsesForApicalSegment = []

    self.chosenCellForColumn = {}


  # ==============================
  # Main methods
  # ==============================


  def compute(self,
              activeColumns,
              activeCellsExternalBasal=(),
              activeCellsExternalApical=(),
              reinforceCandidatesExternalBasal=(),
              reinforceCandidatesExternalApical=(),
              growthCandidatesExternalBasal=(),
              growthCandidatesExternalApical=(),
              learn=True):
    """
    Perform one time step of the Temporal Memory algorithm.

    This method calls activateCells, then calls depolarizeCells. Using the
    TemporalMemory via its compute method ensures that you'll always be able to
    call getPredictiveCells to get predictions for the next time step.

    @param activeColumns (list)
    Sorted list of active columns.

    @param activeCellsExternalBasal (list)
    Sorted list of active external cells for activating basal dendrites at the
    end of this time step.

    @param activeCellsExternalApical (list)
    Sorted list of active external cells for activating apical dendrites at the
    end of this time step.

    @param reinforceCandidatesExternalBasal (list)
    Sorted list of external cells. Any learning basal dendrite segments will use
    this list to decide which synapses to reinforce and which synapses to
    punish. Typically this list should be the 'activeCellsExternalBasal' from
    the prevous time step.

    @param reinforceCandidatesExternalApical (list)
    Sorted list of external cells. Any learning apical dendrite segments will use
    this list to decide which synapses to reinforce and which synapses to
    punish. Typically this list should be the 'activeCellsExternalApical' from
    the prevous time step.

    @param growthCandidatesExternalBasal (list)
    Sorted list of external cells. Any learning basal dendrite segments can grow
    synapses to cells in this list. Typically this list should be a subset of
    the 'activeCellsExternalBasal' from the prevous time step.

    @param growthCandidatesExternalApical (list)
    Sorted list of external cells. Any learning apical dendrite segments can grow
    synapses to cells in this list. Typically this list should be a subset of
    the 'activeCellsExternalApical' from the prevous time step.

    @param learn (bool)
    Whether or not learning is enabled

    """
    self.activateCells(activeColumns,
                       reinforceCandidatesExternalBasal,
                       reinforceCandidatesExternalApical,
                       growthCandidatesExternalBasal,
                       growthCandidatesExternalApical,
                       learn)
    self.depolarizeCells(activeCellsExternalBasal,
                         activeCellsExternalApical,
                         learn)


  def activateCells(self,
                    activeColumns,
                    reinforceCandidatesExternalBasal=(),
                    reinforceCandidatesExternalApical=(),
                    growthCandidatesExternalBasal=(),
                    growthCandidatesExternalApical=(),
                    learn=True):
    """
    Calculate the active cells, using the current active columns and
    dendrite segments. Grow and reinforce synapses.

    @param activeColumns (list)
    A sorted list of active column indices.

    @param reinforceCandidatesExternalBasal (list)
    Sorted list of external cells. Any learning basal dendrite segments will use
    this list to decide which synapses to reinforce and which synapses to
    punish. Typically this list should be the 'activeCellsExternalBasal' from
    the prevous time step.

    @param reinforceCandidatesExternalApical (list)
    Sorted list of external cells. Any learning apical dendrite segments will use
    this list to decide which synapses to reinforce and which synapses to
    punish. Typically this list should be the 'activeCellsExternalApical' from
    the prevous time step.

    @param growthCandidatesExternalBasal (list)
    Sorted list of external cells. Any learning basal dendrite segments can grow
    synapses to cells in this list. Typically this list should be a subset of
    the 'activeCellsExternalBasal' from the previous 'depolarizeCells'.

    @param growthCandidatesExternalApical (list)
    Sorted list of external cells. Any learning apical dendrite segments can grow
    synapses to cells in this list. Typically this list should be a subset of
    the 'activeCellsExternalApical' from the previous 'depolarizeCells'.

    @param learn (bool)
    If true, reinforce / punish / grow synapses.

    """

    if self.checkInputs:
      assert self._isSortedWithoutDuplicates(activeColumns)
      assert self._isSortedWithoutDuplicates(reinforceCandidatesExternalBasal)
      assert self._isSortedWithoutDuplicates(reinforceCandidatesExternalApical)
      assert self._isSortedWithoutDuplicates(growthCandidatesExternalBasal)
      assert self._isSortedWithoutDuplicates(growthCandidatesExternalApical)
      assert all(c >= 0 and c < self._numColumns
                 for c in activeColumns)
      assert all(c >= 0 and c < self._numBasalInputs
                 for c in reinforceCandidatesExternalBasal)
      assert all(c >= 0 and c < self._numApicalInputs
                 for c in reinforceCandidatesExternalApical)
      assert all(c >= 0 and c < self._numBasalInputs
                 for c in growthCandidatesExternalBasal)
      assert all(c >= 0 and c < self._numApicalInputs
                 for c in growthCandidatesExternalApical)

    newActiveCells = []
    newWinnerCells = []

    segToCol = lambda segment: int(segment.cell / self.cellsPerColumn)

    for columnData in groupby2(activeColumns, _identity,
                               self.activeBasalSegments, segToCol,
                               self.matchingBasalSegments, segToCol,
                               self.activeApicalSegments, segToCol,
                               self.matchingApicalSegments, segToCol):
      (column,
       activeColumns,
       columnActiveBasalSegments,
       columnMatchingBasalSegments,
       columnActiveApicalSegments,
       columnMatchingApicalSegments) = groupbyExpand(columnData)

      isActiveColumn = len(activeColumns) > 0

      if isActiveColumn:
        maxPredictiveScore = 0

        for cellData in groupby2(columnActiveBasalSegments, _cellForSegment,
                                 columnActiveApicalSegments, _cellForSegment):
          (cell,
           cellActiveBasalSegments,
           cellActiveApicalSegments) = groupbyExpand(cellData)

          maxPredictiveScore = max(maxPredictiveScore,
                                   self._predictiveScore(cellActiveBasalSegments,
                                                         cellActiveApicalSegments))

        if maxPredictiveScore >= MIN_PREDICTIVE_THRESHOLD:
          cellsToAdd = self.activatePredictedColumn(
            column,
            columnActiveBasalSegments,
            columnMatchingBasalSegments,
            columnActiveApicalSegments,
            columnMatchingApicalSegments,
            maxPredictiveScore,
            self.activeCells,
            reinforceCandidatesExternalBasal,
            reinforceCandidatesExternalApical,
            self.winnerCells,
            growthCandidatesExternalBasal,
            growthCandidatesExternalApical,
            learn)

          newActiveCells += cellsToAdd
          newWinnerCells += cellsToAdd
        else:
          (cellsToAdd,
           winnerCell) = self.burstColumn(
             column,
             columnActiveBasalSegments,
             columnMatchingBasalSegments,
             columnActiveApicalSegments,
             columnMatchingApicalSegments,
             self.activeCells,
             reinforceCandidatesExternalBasal,
             reinforceCandidatesExternalApical,
             self.winnerCells,
             growthCandidatesExternalBasal,
             growthCandidatesExternalApical,
             learn)

          newActiveCells += cellsToAdd
          newWinnerCells.append(winnerCell)
      else:
        if learn:
          self.punishPredictedColumn(
            columnActiveBasalSegments,
            columnMatchingBasalSegments,
            columnActiveApicalSegments,
            columnMatchingApicalSegments,
            self.activeCells,
            reinforceCandidatesExternalBasal,
            reinforceCandidatesExternalApical,
            self.winnerCells,
            growthCandidatesExternalBasal,
            growthCandidatesExternalApical)

    self.activeCells = newActiveCells
    self.winnerCells = newWinnerCells


  def depolarizeCells(self,
                      activeCellsExternalBasal=(),
                      activeCellsExternalApical=(),
                      learn=True):
    """
    Calculate dendrite segment activity, using the current active cells.

    @param activeCellsExternalBasal (list)
    Sorted list of active external cells for activating basal dendrites.

    @param activeCellsExternalApical (list)
    Sorted list of active external cells for activating apical dendrites.

    @param learn (bool)
    If true, segment activations will be recorded. This information is used
    during segment cleanup.

    """

    if self.checkInputs:
      assert all(c >= 0 and c < self._numBasalInputs
                 for c in activeCellsExternalBasal)
      assert all(c >= 0 and c < self._numApicalInputs
                 for c in activeCellsExternalApical)

    (self.activeBasalSegments,
     self.matchingBasalSegments,
     self.numActiveConnectedSynapsesForBasalSegment,
     self.numActivePotentialSynapsesForBasalSegment) = self._calculateExcitations(
      self.basalConnections, self.activeCells, activeCellsExternalBasal,
      self.connectedPermanence, self.activationThreshold, self.minThreshold,
      learn)

    (self.activeApicalSegments,
     self.matchingApicalSegments,
     self.numActiveConnectedSynapsesForApicalSegment,
     self.numActivePotentialSynapsesForApicalSegment) = self._calculateExcitations(
      self.apicalConnections, self.activeCells, activeCellsExternalApical,
      self.connectedPermanence, self.activationThreshold, self.minThreshold,
      learn)


  def reset(self):
    """
    Indicates the start of a new sequence. Clears any predictions and makes sure
    synapses don't grow to the currently active cells in the next time step.
    """
    self.activeCells = []
    self.winnerCells = []
    self.activeBasalSegments = []
    self.matchingBasalSegments = []
    self.activeApicalSegments = []
    self.matchingApicalSegments = []
    self.chosenCellForColumn = {}


  # ==============================
  # Extension points
  # These methods are designed to be overridden.
  # ==============================


  def activatePredictedColumn(self,
                              column,
                              columnActiveBasalSegments,
                              columnMatchingBasalSegments,
                              columnActiveApicalSegments,
                              columnMatchingApicalSegments,
                              predictiveThreshold,
                              reinforceCandidatesInternal,
                              reinforceCandidatesExternalBasal,
                              reinforceCandidatesExternalApical,
                              growthCandidatesInternal,
                              growthCandidatesExternalBasal,
                              growthCandidatesExternalApical,
                              learn):
    """
    @param column (int)
    Index of column.

    @param columnMatchingBasalSegments (tuple)
    Active basal segments in this column.

    @param columnMatchingBasalSegments (tuple)
    Matching basal segments in this column.

    @param columnMatchingApicalSegments (tuple)
    Active apical segments in this column.

    @param columnMatchingApicalSegments (tuple)
    Matching apical segments in this column.

    @param predictiveThreshold (int)
    The minimum predictive score required for a cell to become active.

    @param learn (bool)
    If true, grow and reinforce synapses.

    @return cellsToAdd (list)
    A list of predicted cells that will be added to active cells and winner
    cells.

    """

    return self._activatePredictedColumn(
      self.basalConnections, self.apicalConnections, self._random,
      columnActiveBasalSegments, columnMatchingBasalSegments,
      columnActiveApicalSegments, columnMatchingApicalSegments,
      predictiveThreshold,
      reinforceCandidatesInternal,
      reinforceCandidatesExternalBasal,
      reinforceCandidatesExternalApical,
      growthCandidatesInternal,
      growthCandidatesExternalBasal,
      growthCandidatesExternalApical,
      self.numActivePotentialSynapsesForBasalSegment,
      self.numActivePotentialSynapsesForApicalSegment,
      self.maxNewSynapseCount, self.initialPermanence,
      self.permanenceIncrement, self.permanenceDecrement,
      self.formInternalBasalConnections, learn)


  def burstColumn(self, column,
                  columnActiveBasalSegments, columnMatchingBasalSegments,
                  columnActiveApicalSegments, columnMatchingApicalSegments,
                  reinforceCandidatesInternal,
                  reinforceCandidatesExternalBasal,
                  reinforceCandidatesExternalApical,
                  growthCandidatesInternal,
                  growthCandidatesExternalBasal,
                  growthCandidatesExternalApical,
                  learn):
    """
    @param column (int)
    Index of column.

    @param columnMatchingBasalSegments (tuple)
    Active basal segments in this column.

    @param columnMatchingBasalSegments (tuple)
    Matching basal segments in this column.

    @param columnMatchingApicalSegments (tuple)
    Active apical segments in this column.

    @param columnMatchingApicalSegments (tuple)
    Matching apical segments in this column.

    @param learn (bool)
    Whether or not learning is enabled.

    @return (tuple) Contains:
                      `cells`         (iter),
                      `winnerCell`    (int),

    """

    return self._burstColumn(
      self.basalConnections, self.apicalConnections, self._random,
      self.chosenCellForColumn,
      column,
      columnActiveBasalSegments, columnMatchingBasalSegments,
      columnActiveApicalSegments, columnMatchingApicalSegments,
      reinforceCandidatesInternal,
      reinforceCandidatesExternalBasal,
      reinforceCandidatesExternalApical,
      growthCandidatesInternal,
      growthCandidatesExternalBasal,
      growthCandidatesExternalApical,
      self.numActivePotentialSynapsesForBasalSegment,
      self.numActivePotentialSynapsesForApicalSegment,
      self.cellsPerColumn, self.maxNewSynapseCount, self.initialPermanence,
      self.permanenceIncrement, self.permanenceDecrement,
      self.formInternalBasalConnections, self.learnOnOneCell, learn)


  def punishPredictedColumn(self,
                            columnActiveBasalSegments,
                            columnMatchingBasalSegments,
                            columnActiveApicalSegments,
                            columnMatchingApicalSegments,
                            reinforceCandidatesInternal,
                            reinforceCandidatesExternalBasal,
                            reinforceCandidatesExternalApical,
                            growthCandidatesInternal,
                            growthCandidatesExternalBasal,
                            growthCandidatesExternalApical):
    """
    @param columnMatchingBasalSegments (tuple)
    Active basal segments in this column.

    @param columnMatchingBasalSegments (tuple)
    Matching basal segments in this column.

    @param columnMatchingApicalSegments (tuple)
    Active apical segments in this column.

    @param columnMatchingApicalSegments (tuple)
    Matching apical segments in this column.

    """
    # Punish basal segments.
    self._punishPredictedColumn(self.basalConnections,
                                columnMatchingBasalSegments,
                                reinforceCandidatesInternal,
                                reinforceCandidatesExternalBasal,
                                self.predictedSegmentDecrement)

    # Don't punish apical segments.


  # ==============================
  # Helper methods
  #
  # These class methods use the following parameter ordering convention:
  #
  # 1. Output / mutated params
  # 2. Traditional parameters to the method, i.e. the ones that would still
  #    exist if this were in instance method.
  # 3. Model state (not mutated)
  # 4. Model parameters (including "learn")
  # ==============================


  @classmethod
  def _activatePredictedColumn(cls, basalConnections, apicalConnections, rng,
                               columnActiveBasalSegments,
                               columnMatchingBasalSegments,
                               columnActiveApicalSegments,
                               columnMatchingApicalSegments,
                               predictiveThreshold,
                               reinforceCandidatesInternal,
                               reinforceCandidatesExternalBasal,
                               reinforceCandidatesExternalApical,
                               growthCandidatesInternal,
                               growthCandidatesExternalBasal,
                               growthCandidatesExternalApical,
                               numActivePotentialSynapsesForBasalSegment,
                               numActivePotentialSynapsesForApicalSegment,
                               maxNewSynapseCount, initialPermanence,
                               permanenceIncrement, permanenceDecrement,
                               formInternalBasalConnections, learn):

    cellsToAdd = []

    for cellData in groupby2(columnActiveBasalSegments, _cellForSegment,
                             columnMatchingBasalSegments, _cellForSegment,
                             columnActiveApicalSegments, _cellForSegment,
                             columnMatchingApicalSegments, _cellForSegment):
      (cell,
       cellActiveBasalSegments,
       cellMatchingBasalSegments,
       cellActiveApicalSegments,
       cellMatchingApicalSegments) = groupbyExpand(cellData)

      if cls._predictiveScore(cellActiveBasalSegments,
                              cellActiveApicalSegments) >= predictiveThreshold:
        cellsToAdd.append(cell)

        if learn:
          # Basal learning.
          growthCandidatesInternalBasal = (
            growthCandidatesInternal if formInternalBasalConnections
            else tuple()
          )
          cls._learnOnCell(basalConnections, rng,
                           cell,
                           cellActiveBasalSegments, cellMatchingBasalSegments,
                           reinforceCandidatesInternal,
                           reinforceCandidatesExternalBasal,
                           growthCandidatesInternalBasal,
                           growthCandidatesExternalBasal,
                           numActivePotentialSynapsesForBasalSegment,
                           maxNewSynapseCount, initialPermanence,
                           permanenceIncrement, permanenceDecrement)

          # Apical learning.
          growthCandidatesInternalApical = tuple()
          cls._learnOnCell(apicalConnections, rng,
                           cell,
                           cellActiveApicalSegments, cellMatchingApicalSegments,
                           reinforceCandidatesInternal,
                           reinforceCandidatesExternalApical,
                           growthCandidatesInternalApical,
                           growthCandidatesExternalApical,
                           numActivePotentialSynapsesForApicalSegment,
                           maxNewSynapseCount, initialPermanence,
                           permanenceIncrement, permanenceDecrement)

    return cellsToAdd


  @classmethod
  def _burstColumn(cls, basalConnections, apicalConnections, rng,
                   chosenCellForColumn,
                   column,
                   columnActiveBasalSegments, columnMatchingBasalSegments,
                   columnActiveApicalSegments, columnMatchingApicalSegments,
                   reinforceCandidatesInternal,
                   reinforceCandidatesExternalBasal,
                   reinforceCandidatesExternalApical,
                   growthCandidatesInternal,
                   growthCandidatesExternalBasal,
                   growthCandidatesExternalApical,
                   numActivePotentialSynapsesForBasalSegment,
                   numActivePotentialSynapsesForApicalSegment,
                   cellsPerColumn, maxNewSynapseCount, initialPermanence,
                   permanenceIncrement, permanenceDecrement,
                   formInternalBasalConnections, learnOnOneCell, learn):
    start = cellsPerColumn * column
    cells = xrange(start, start + cellsPerColumn)

    # Calculate the winner cell.
    if learnOnOneCell and column in chosenCellForColumn:
      winnerCell = chosenCellForColumn[column]
    else:
      if len(columnMatchingBasalSegments) > 0:
        numActive = lambda s: numActivePotentialSynapsesForBasalSegment[s.flatIdx]
        bestBasalSegment = max(columnMatchingBasalSegments, key=numActive)
        winnerCell = bestBasalSegment.cell

        # Mini optimization: don't search for the best basal segment twice.
        columnMatchingBasalSegments = (bestBasalSegment,)
      else:
        winnerCell = cls._getLeastUsedCell(rng, cells, basalConnections)

      if learnOnOneCell:
        chosenCellForColumn[column] = winnerCell

    if learn:
      # Basal learning.
      cellActiveBasalSegments = [s for s in columnActiveBasalSegments
                                 if s.cell == winnerCell]
      cellMatchingBasalSegments = [s for s in columnMatchingBasalSegments
                                   if s.cell == winnerCell]
      growthCandidatesInternalBasal = (
        growthCandidatesInternal if formInternalBasalConnections else tuple()
      )
      cls._learnOnCell(basalConnections, rng,
                       winnerCell,
                       cellActiveBasalSegments, cellMatchingBasalSegments,
                       reinforceCandidatesInternal,
                       reinforceCandidatesExternalBasal,
                       growthCandidatesInternalBasal,
                       growthCandidatesExternalBasal,
                       numActivePotentialSynapsesForBasalSegment,
                       maxNewSynapseCount, initialPermanence,
                       permanenceIncrement, permanenceDecrement)

      # Apical learning.
      cellActiveApicalSegments = [s for s in columnActiveApicalSegments
                                  if s.cell == winnerCell]
      cellMatchingApicalSegments = [s for s in columnMatchingApicalSegments
                                    if s.cell == winnerCell]
      growthCandidatesInternalApical = tuple()
      cls._learnOnCell(apicalConnections, rng,
                       winnerCell,
                       cellActiveApicalSegments, cellMatchingApicalSegments,
                       reinforceCandidatesInternal,
                       reinforceCandidatesExternalApical,
                       growthCandidatesInternalApical,
                       growthCandidatesExternalApical,
                       numActivePotentialSynapsesForApicalSegment,
                       maxNewSynapseCount, initialPermanence,
                       permanenceIncrement, permanenceDecrement)

    return cells, winnerCell


  @classmethod
  def _punishPredictedColumn(cls, connections, columnMatchingSegments,
                             reinforceCandidatesInternal,
                             reinforceCandidatesExternal,
                             predictedSegmentDecrement):
    if predictedSegmentDecrement > 0.0 and len(columnMatchingSegments) > 0:
      for segment in columnMatchingSegments:
        cls._adaptSegment(connections, segment,
                          reinforceCandidatesInternal,
                          reinforceCandidatesExternal,
                          -predictedSegmentDecrement, 0.0)


  @classmethod
  def _learnOnCell(cls, connections, rng,
                   cell,
                   cellActiveSegments, cellMatchingSegments,
                   reinforceCandidatesInternal,
                   reinforceCandidatesExternal,
                   growthCandidatesInternal,
                   growthCandidatesExternal,
                   numActivePotentialSynapsesForSegment,
                   maxNewSynapseCount, initialPermanence,
                   permanenceIncrement, permanenceDecrement):
    if len(cellActiveSegments) > 0:
      # Learn on every active segment.
      for segment in cellActiveSegments:
        cls._adaptSegment(connections, segment,
                          reinforceCandidatesInternal,
                          reinforceCandidatesExternal,
                          permanenceIncrement, permanenceDecrement)

        active = numActivePotentialSynapsesForSegment[segment.flatIdx]
        nGrowDesired = maxNewSynapseCount - active

        if nGrowDesired > 0:
          cls._growSynapses(connections, rng, segment, nGrowDesired,
                            growthCandidatesInternal,
                            growthCandidatesExternal,
                            initialPermanence)
    elif len(cellMatchingSegments) > 0:
      # No active segments.
      # Learn on the best matching segment.
      numActive = lambda s: numActivePotentialSynapsesForSegment[s.flatIdx]
      bestMatchingSegment = max(cellMatchingSegments, key=numActive)

      cls._adaptSegment(connections, bestMatchingSegment,
                        reinforceCandidatesInternal,
                        reinforceCandidatesExternal,
                        permanenceIncrement, permanenceDecrement)

      nGrowDesired = maxNewSynapseCount - numActive(bestMatchingSegment)

      if nGrowDesired > 0:
        cls._growSynapses(connections, rng, bestMatchingSegment, nGrowDesired,
                          growthCandidatesInternal,
                          growthCandidatesExternal,
                          initialPermanence)
    else:
      # No matching segments.
      # Grow a new segment and learn on it.
      nGrowExact = min(maxNewSynapseCount,
                       len(growthCandidatesInternal) + len(growthCandidatesExternal))
      if nGrowExact > 0:
        segment = connections.createSegment(cell)
        cls._growSynapses(connections, rng, segment, nGrowExact,
                          growthCandidatesInternal,
                          growthCandidatesExternal,
                          initialPermanence)


  @classmethod
  def _calculateExcitations(cls, connections, activeCells, activeExternalCells,
                            connectedPermanence, activationThreshold,
                            minThreshold, learn):

    numCells = connections.numCells
    allActiveCells = itertools.chain(activeCells,
                                     (c + numCells
                                      for c in activeExternalCells))

    (numActiveConnected,
     numActivePotential) = connections.computeActivity(allActiveCells,
                                                       connectedPermanence)

    activeSegments = list(
      connections.segmentForFlatIdx(i)
      for i in xrange(len(numActiveConnected))
      if numActiveConnected[i] >= activationThreshold
    )

    matchingSegments = list(
      connections.segmentForFlatIdx(i)
      for i in xrange(len(numActivePotential))
      if numActivePotential[i] >= minThreshold
    )

    if learn:
      for segment in activeSegments:
        connections.recordSegmentActivity(segment)
      connections.startNewIteration()

    return (sorted(activeSegments, key = connections.segmentPositionSortKey),
            sorted(matchingSegments, key = connections.segmentPositionSortKey),
            numActiveConnected,
            numActivePotential)


  @classmethod
  def _getLeastUsedCell(cls, rng, cells, connections):
    leastUsedCells = []
    minNumSegments = float("inf")
    for cell in cells:
      numSegments = connections.numSegments(cell)

      if numSegments < minNumSegments:
        minNumSegments = numSegments
        leastUsedCells = []

      if numSegments == minNumSegments:
        leastUsedCells.append(cell)

    i = rng.getUInt32(len(leastUsedCells))
    return leastUsedCells[i]


  @classmethod
  def _growSynapses(cls, connections, rng, segment, nDesiredNewSynapes,
                    growthCandidatesInternal, growthCandidatesExternal,
                    initialPermanence):
    numCells = connections.numCells
    candidates = list(growthCandidatesInternal)
    candidates.extend(cell + numCells for cell in growthCandidatesExternal)

    for synapse in connections.synapsesForSegment(segment):
      i = binSearch(candidates, synapse.presynapticCell)
      if i != -1:
        del candidates[i]

    nActual = min(nDesiredNewSynapes, len(candidates))

    for _ in range(nActual):
      i = rng.getUInt32(len(candidates))
      connections.createSynapse(segment, candidates[i], initialPermanence)
      del candidates[i]


  @classmethod
  def _adaptSegment(cls, connections, segment,
                    reinforceCandidatesInternal, reinforceCandidatesExternal,
                    permanenceIncrement, permanenceDecrement):
    numCells = connections.numCells

    # Destroying a synapse modifies the set that we're iterating through.
    synapsesToDestroy = []

    for synapse in connections.synapsesForSegment(segment):
      permanence = synapse.permanence
      presynapticCell = synapse.presynapticCell

      if presynapticCell < numCells:
        isActive = -1 != binSearch(reinforceCandidatesInternal,
                                   presynapticCell)
      else:
        isActive = -1 != binSearch(reinforceCandidatesExternal,
                                   presynapticCell - numCells)

      if isActive:
        permanence += permanenceIncrement
      else:
        permanence -= permanenceDecrement

      # Keep permanence within min/max bounds.
      permanence = max(0.0, min(1.0, permanence))

      if permanence < EPSILON:
        synapsesToDestroy.append(synapse)
      else:
        connections.updateSynapsePermanence(synapse, permanence)

    for synapse in synapsesToDestroy:
      connections.destroySynapse(synapse)

    if connections.numSynapses(segment) == 0:
      connections.destroySegment(segment)


  @staticmethod
  def _isSortedWithoutDuplicates(iterable):
    """
    Returns True if the input is sorted and contains no duplicates.

    @param iterable (iter)

    @return (bool)
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return all(itertools.imap(operator.lt, a, b))


  @classmethod
  def _predictiveScore(cls, activeBasalSegmentsOnCell,
                       activeApicalSegmentsOnCell):
    score = 0

    if len(activeBasalSegmentsOnCell) > 0:
      score += 2

    if len(activeApicalSegmentsOnCell) > 0:
      score += 1

    return score


  def columnForCell(self, cell):
    """ Returns the index of the column that a cell belongs to.

    @param cell (int) Cell index

    @return (int) Column index
    """
    self._validateCell(cell)

    return int(cell / self.cellsPerColumn)


  def cellsForColumn(self, column):
    """ Returns the indices of cells that belong to a column.

    @param column (int) Column index

    @return (list) Cell indices
    """
    self._validateColumn(column)

    start = self.cellsPerColumn * column
    end = start + self.cellsPerColumn
    return range(start, end)


  def numberOfColumns(self):
    """ Returns the number of columns in this layer.

    @return (int) Number of columns
    """
    return self._numColumns


  def numberOfCells(self):
    """ Returns the number of cells in this layer.

    @return (int) Number of cells
    """
    return self.numberOfColumns() * self.cellsPerColumn


  def getActiveCells(self):
    """ Returns the indices of the active cells.

    @return (list) Indices of active cells.
    """
    return self.getCellIndices(self.activeCells)


  def getPredictiveCells(self):
    """ Returns the indices of the predictive cells.

    @return (list) Indices of predictive cells.
    """

    predictiveCells = []

    segToCol = lambda segment: int(segment.cell / self.cellsPerColumn)

    for columnData in groupby2(self.activeBasalSegments, segToCol,
                               self.activeApicalSegments, segToCol):
      (column,
       columnActiveBasalSegments,
       columnActiveApicalSegments) = groupbyExpand(columnData)

      maxPredictiveScore = 0

      for cellData in groupby2(columnActiveBasalSegments, _cellForSegment,
                               columnActiveApicalSegments, _cellForSegment):
        (cell,
         cellActiveBasalSegments,
         cellActiveApicalSegments) = groupbyExpand(cellData)

        maxPredictiveScore = max(maxPredictiveScore,
                                 self._predictiveScore(cellActiveBasalSegments,
                                                       cellActiveApicalSegments))

      if maxPredictiveScore >= MIN_PREDICTIVE_THRESHOLD:
        for cellData in groupby2(columnActiveBasalSegments, _cellForSegment,
                                 columnActiveApicalSegments, _cellForSegment):
          (cell,
           cellActiveBasalSegments,
           cellActiveApicalSegments) = groupbyExpand(cellData)

          if self._predictiveScore(cellActiveBasalSegments,
                                   cellActiveApicalSegments) >= maxPredictiveScore:
            predictiveCells.append(cell)

    return predictiveCells


  def getWinnerCells(self):
    """ Returns the indices of the winner cells.

    @return (list) Indices of winner cells.
    """
    return self.getCellIndices(self.winnerCells)


  def getCellsPerColumn(self):
    """ Returns the number of cells per column.

    @return (int) The number of cells per column.
    """
    return self.cellsPerColumn


  def getColumnDimensions(self):
    """
    Returns the dimensions of the columns.
    @return (tuple) Column dimensions
    """
    return self.columnDimensions


  def getBasalInputDimensions(self):
    """
    Returns the dimensions of the external basal input.
    @return (tuple) External basal input dimensions
    """
    return self.basalInputDimensions


  def getApicalInputDimensions(self):
    """
    Returns the dimensions of the external apical input.
    @return (tuple) External apical input dimensions
    """
    return self.apicalInputDimensions


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


  def getMaxNewSynapseCount(self):
    """
    Returns the max new synapse count.
    @return (int) The max new synapse count.
    """
    return self.maxNewSynapseCount


  def setMaxNewSynapseCount(self, maxNewSynapseCount):
    """
    Sets the max new synapse count.
    @param maxNewSynapseCount (int) Max new synapse count.
    """
    self.maxNewSynapseCount = maxNewSynapseCount


  def getFormInternalBasalConnections(self):
    """
    Returns whether to form internal connections between cells.
    @return (bool) the formInternalBasalConnections parameter
    """
    return self.formInternalBasalConnections


  def setFormInternalBasalConnections(self, formInternalBasalConnections):
    """
    Sets whether to form internal connections between cells.
    @param formInternalBasalConnections (bool)
    """
    self.formInternalBasalConnections = formInternalBasalConnections


  def getLearnOnOneCell(self):
    """
    Returns whether to always choose the same cell when bursting a column until
    the next reset occurs.
    @return (bool) the learnOnOneCell parameter
    """
    return self.learnOnOneCell


  def setLearnOnOneCell(self, learnOnOneCell):
    """
    Sets whether to always choose the same cell when bursting a column until the
    next reset occurs.
    @param learnOnOneCell (bool)
    """
    self.learnOnOneCell = learnOnOneCell


  def setMaxNewSynapseCount(self, maxNewSynapseCount):
    """
    Sets the max new synapse count.
    @param maxNewSynapseCount (int) Max new synapse count.
    """
    self.maxNewSynapseCount = maxNewSynapseCount


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


  def getPredictedSegmentDecrement(self):
    """
    Get the predicted segment decrement.
    @return (float) The predicted segment decrement.
    """
    return self.predictedSegmentDecrement


  def setPredictedSegmentDecrement(self, predictedSegmentDecrement):
    """
    Sets the predicted segment decrement.
    @param predictedSegmentDecrement (float) The predicted segment decrement.
    """
    self.predictedSegmentDecrement = predictedSegmentDecrement


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


  def __eq__(self, other):
    """
    Equality operator for ExtendedTemporalMemory instances.
    Checks if two instances are functionally identical
    (might have different internal state).

    @param other (TemporalMemory) TemporalMemory instance to compare to
    """
    if self.columnDimensions != other.columnDimensions:
      return False
    if self.cellsPerColumn != other.cellsPerColumn:
      return False
    if self.activationThreshold != other.activationThreshold:
      return False
    if abs(self.initialPermanence - other.initialPermanence) > EPSILON:
      return False
    if abs(self.connectedPermanence - other.connectedPermanence) > EPSILON:
      return False
    if self.minThreshold != other.minThreshold:
      return False
    if self.maxNewSynapseCount != other.maxNewSynapseCount:
      return False
    if abs(self.permanenceIncrement - other.permanenceIncrement) > EPSILON:
      return False
    if abs(self.permanenceDecrement - other.permanenceDecrement) > EPSILON:
      return False
    if abs(self.predictedSegmentDecrement -
           other.predictedSegmentDecrement) > EPSILON:
      return False

    if self.connections != other.connections:
      return False
    if self.activeCells != other.activeCells:
      return False
    if self.winnerCells != other.winnerCells:
      return False

    if self.matchingSegments != other.matchingSegments:
      return False
    if self.activeSegments != other.activeSegments:
      return False

    return True


  def __ne__(self, other):
    """
    Non-equality operator for ExtendedTemporalMemory instances.
    Checks if two instances are not functionally identical
    (might have different internal state).

    @param other (ExtendedTemporalMemory)
    ETM instance to compare to.
    """
    return not self.__eq__(other)


  def _validateColumn(self, column):
    """ Raises an error if column index is invalid.

    @param column (int) Column index
    """
    if column >= self.numberOfColumns() or column < 0:
      raise IndexError("Invalid column")


  def _validateCell(self, cell):
    """ Raises an error if cell index is invalid.

    @param cell (int) Cell index
    """
    if cell >= self.numberOfCells() or cell < 0:
      raise IndexError("Invalid cell")


  @classmethod
  def getCellIndices(cls, cells):
    """ Returns the indices of the cells passed in.

    @param cells (list) cells to find the indices of
    """
    return [cls.getCellIndex(c) for c in cells]


  @staticmethod
  def getCellIndex(cell):
    """ Returns the index of the cell

    @param cell (int) cell to find the index of
    """
    return cell


def _identity(x):
  return x


def _cellForSegment(segment):
  return segment.cell

def _numPoints(dimensions):
  if len(dimensions) == 0:
    return 0
  else:
    return reduce(operator.mul, dimensions, 1)


def groupbyExpand(groupbyIteration):
  """
  Convert iterators and 'None's from a groupby2 iteration into simple tuples.

  This lets you avoid the following hassles:

  - groupby2 returns iterators. Iterators are single-use.
  - groupby2 returns values that don't support `len`.
  - groupby2 returns iterators *or* None. It's messy to handle this in the
    caller.

  groupbyExpand has a performance trade-off. It copies the sequences into
  tuples, using more memory. Using groupby2 + groupbyExpand is still a fast way
  of computing the intersection and differences of an arbitrary number of sorted
  lists.

  @param groupbyIteration (tuple)
  One yielded return value from groupby2.

  @returns (list)
  A copy of groupbyIteration, but with its iterators converted to tuples.
  """
  expanded = list(groupbyIteration)
  for i in xrange(1, len(expanded)):
    expanded[i] = tuple(expanded[i]) if expanded[i] is not None else ()
  return expanded
