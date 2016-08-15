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

from htmresearch.algorithms.connections_phases import Connections
from htmresearch.algorithms.temporal_memory_phases import TemporalMemory



class ExtendedTemporalMemory(TemporalMemory):
  """
  Class implementing the Temporal Memory algorithm with the added ability of
  being able to learn from both internal and external cell activation. This
  class has an option to learn on a single cell within a column and not
  look for a new winner cell until a reset() is called.
  """

  # ==============================
  # Main functions
  # ==============================


  def __init__(self,
               learnOnOneCell=True,
               **kwargs):
    """
    @param learnOnOneCell (boolean) If True, the winner cell for each column will be fixed between resets.
    """

    super(ExtendedTemporalMemory, self).__init__(**kwargs)

    self.activeExternalCells = set()
    self.learnOnOneCell = learnOnOneCell
    self.chosenCellForColumn = dict()

    self.unpredictedActiveColumns = set()
    self.predictedActiveCells = set()

    self.activeApicalCells = set()
    self.apicalConnections = Connections(self.numberOfCells())
    self.activeApicalSegments = set()
    self.matchingApicalSegments = set()


  def compute(self,
              activeColumns,
              activeExternalCells=None,
              activeApicalCells=None,
              formInternalConnections=True,
              learn=True):
    """
    Feeds input record through TM, performing inference and learning.
    Updates member variables with new state.

    @param activeColumns           (set)     Indices of active columns in `t`
    @param activeExternalCells     (set)     Indices of active external inputs in `t`
    @param formInternalConnections (boolean) Flag to determine whether to form connections with
                                             internal cells within this temporal memory
    """

    if activeExternalCells is None:
      activeExternalCells = set()

    if activeApicalCells is None:
      activeApicalCells = set()

    activeExternalCells = self._reindexActiveCells(activeExternalCells)
    activeApicalCells = self._reindexActiveCells(activeApicalCells)

    (activeCells,
     winnerCells,
     activeSegments,
     activeApicalSegments,
     predictiveCells,
     predictedActiveColumns,
     matchingSegments,
     matchingApicalSegments,
     matchingCells,
     chosenCellForColumn) = self.computeFn(activeColumns,
                                           activeExternalCells,
                                           activeApicalCells,
                                           self.activeExternalCells,
                                           self.activeApicalCells,
                                           self.predictiveCells,
                                           self.activeSegments,
                                           self.activeApicalSegments,
                                           self.activeCells,
                                           self.winnerCells,
                                           self.matchingSegments,
                                           self.matchingApicalSegments,
                                           self.matchingCells,
                                           self.connections,
                                           self.apicalConnections,
                                           formInternalConnections,
                                           self.learnOnOneCell,
                                           self.chosenCellForColumn,
                                           learn=learn)

    self.activeExternalCells = activeExternalCells
    self.activeApicalCells = activeApicalCells

    self.unpredictedActiveColumns = activeColumns - predictedActiveColumns
    self.predictedActiveCells = self.predictiveCells & activeCells

    self.activeCells = activeCells
    self.winnerCells = winnerCells
    self.activeSegments = activeSegments
    self.activeApicalSegments = activeApicalSegments
    self.predictiveCells = predictiveCells
    self.chosenCellForColumn = chosenCellForColumn
    self.matchingSegments = matchingSegments
    self.matchingApicalSegments = matchingApicalSegments
    self.matchingCells = matchingCells


  def computeFn(self,
                activeColumns,
                activeExternalCells,
                activeApicalCells,
                prevActiveExternalCells,
                prevActiveApicalCells,
                prevPredictiveCells,
                prevActiveSegments,
                prevActiveApicalSegments,
                prevActiveCells,
                prevWinnerCells,
                prevMatchingSegments,
                prevMatchingApicalSegments,
                prevMatchingCells,
                connections,
                apicalConnections,
                formInternalConnections,
                learnOnOneCell,
                chosenCellForColumn,
                learn=True):
    """
    'Functional' version of compute.
    Returns new state.

    @param activeColumns                   (set)         Indices of active columns in `t`
    @param activeExternalCells             (set)         Indices of active external cells in `t`
    @param activeApicalCells               (set)         Indices of active apical cells in `t`
    @param prevActiveExternalCells         (set)         Indices of active external cells in `t-1`
    @param prevActiveApicalCells           (set)         Indices of active apical cells in `t-1`
    @param prevPredictiveCells             (set)         Indices of predictive cells in `t-1`
    @param prevActiveSegments              (set)         Indices of active segments in `t-1`
    @param prevActiveApicalSegments        (set)         Indices of active apical segments in `t-1`
    @param prevActiveCells                 (set)         Indices of active cells in `t-1`
    @param prevWinnerCells                 (set)         Indices of winner cells in `t-1`
    @param prevMatchingSegments            (set)         Indices of matching segments in `t-1`
    @param prevMatchingApicalSegments      (set)         Indices of matching apical segments in `t-1`
    @param prevMatchingCells               (set)         Indices of matching cells in `t-1`
    @param connections                     (Connections) Connectivity of layer
    @param apicalConnections               (Connections) Apical connectivity of layer
    @param formInternalConnections         (boolean)     Flag to determine whether to form connections
                                                         with internal cells within this temporal memory
    @param learnOnOneCell                  (boolean)     If True, the winner cell for each column will
                                                         be fixed between resets.
    @param chosenCellForColumn             (dict)        The current winner cell for each column, if
                                                         it exists.

    @return (tuple) Contains:
                      `activeCells`               (set),
                      `winnerCells`               (set),
                      `activeSegments`            (set),
                      `activeApicalSegments`      (set),
                      `predictiveCells`           (set),
                      'predictedActiveColumns'    (set),
                      'matchingSegments'          (set),
                      'matchingApicalSegments'    (set),
                      'matchingCells'             (set),
                      'chosenCellForColumn'       (dict)
    """
    activeCells = set()
    winnerCells = set()

    (_activeCells,
     _winnerCells,
     predictedActiveColumns,
     predictedInactiveCells) = self.activateCorrectlyPredictiveCells(
       prevPredictiveCells,
       prevMatchingCells,
       activeColumns)

    activeCells.update(_activeCells)
    winnerCells.update(_winnerCells)

    (_activeCells,
     _winnerCells,
     learningSegments,
     apicalLearningSegments,
     chosenCellForColumn) = self.burstColumns(
       activeColumns,
       predictedActiveColumns,
       prevActiveCells | prevActiveExternalCells,
       prevActiveApicalCells,
       prevWinnerCells,
       learnOnOneCell,
       chosenCellForColumn,
       connections,
       apicalConnections)

    activeCells.update(_activeCells)
    winnerCells.update(_winnerCells)

    if learn:
      prevCellActivity = prevActiveExternalCells
      self.learnOnApicalSegments(prevActiveApicalSegments,
                                 apicalLearningSegments,
                                 prevActiveApicalCells,
                                 winnerCells,
                                 apicalConnections,
                                 predictedInactiveCells,
                                 prevMatchingApicalSegments)


      if formInternalConnections:
        prevCellActivity.update(prevWinnerCells)

      self.learnOnSegments(prevActiveSegments,
                           learningSegments,
                           prevActiveCells | prevActiveExternalCells,
                           winnerCells,
                           prevCellActivity,
                           connections,
                           predictedInactiveCells,
                           prevMatchingSegments)

    allActiveCells = activeCells | activeExternalCells
    (activeSegments,
    predictiveDistalCells,
    matchingSegments,
    matchingDistalCells) = self.computePredictiveCells(allActiveCells,
                                                       connections)

    (activeApicalSegments,
    predictiveApicalCells,
    matchingApicalSegments,
    matchingApicalCells) = self.computePredictiveCells(activeApicalCells,
                                                       apicalConnections)

    matchingCells = matchingDistalCells | matchingApicalCells

    predictiveCells = self.calculatePredictiveCells(predictiveDistalCells,
                                                    predictiveApicalCells)

    return (activeCells,
            winnerCells,
            activeSegments,
            activeApicalSegments,
            predictiveCells,
            predictedActiveColumns,
            matchingSegments,
            matchingApicalSegments,
            matchingCells,
            chosenCellForColumn)


  def reset(self):
    super(ExtendedTemporalMemory, self).reset()

    self.activeExternalCells = set()
    self.chosenCellForColumn = dict()

    self.unpredictedActiveColumns = set()
    self.predictedActiveCells = set()

    self.activeApicalCells = set()
    self.activeApicalSegments = set()
    self.matchingApicalSegments = set()


  def burstColumns(self,
                   activeColumns,
                   predictedActiveColumns,
                   prevActiveCells,
                   prevActiveApicalCells,
                   prevWinnerCells,
                   learnOnOneCell,
                   chosenCellForColumn,
                   connections,
                   apicalConnections):
    """
    Phase 2: Burst unpredicted columns.

    Pseudocode:

      - for each unpredicted active column
        - mark all cells as active
        - If learnOnOneCell, keep the old best matching cell if it exists
        - mark the best matching cell as winner cell
          - (learning)
            - if it has no matching segment
              - (optimization) if there are prev winner cells
                - add a segment to it
            - mark the segment as learning

    @param activeColumns                   (set)         Indices of active columns in `t`
    @param predictedActiveColumns          (set)         Indices of predicted => active columns in `t`
    @param prevActiveCells                 (set)         Indices of active cells in `t-1`
    @param prevActiveApicalCells           (set)         Indices of ext active cells in `t-1`
    @param prevWinnerCells                 (set)         Indices of winner cells in `t-1`
    @param learnOnOneCell                  (boolean)     If True, the winner cell for each column will
                                                         be fixed between resets.
    @param chosenCellForColumn             (dict)        The current winner cell for each column,
                                                         if it exists.
    @param connections                     (Connections) Connectivity of layer
    @param apicalConnections               (Connections) External connectivity of layer

    @return (tuple) Contains:
                      `activeCells`      (set),
                      `winnerCells`      (set),
                      `learningSegments` (set),
                      `apicalLearningSegments` (set)
    """
    activeCells = set()
    winnerCells = set()
    learningSegments = set()
    apicalLearningSegments = set()

    unpredictedActiveColumns = activeColumns - predictedActiveColumns

    for column in unpredictedActiveColumns:
      cells = self.cellsForColumn(column)
      activeCells.update(cells)

      if learnOnOneCell and (column in chosenCellForColumn):
        chosenCell = chosenCellForColumn[column]
        cells = set([chosenCell])

      (bestCell,
       bestSegment,
       bestApicalSegment) = self.bestMatchingCell(cells,
                                                  prevActiveCells,
                                                  prevActiveApicalCells,
                                                  connections,
                                                  apicalConnections)
      winnerCells.add(bestCell)

      if bestSegment is None and len(prevWinnerCells):
        bestSegment = connections.createSegment(bestCell)

      if bestApicalSegment is None:
        bestApicalSegment = apicalConnections.createSegment(bestCell)

      if bestSegment is not None:
        learningSegments.add(bestSegment)

      if bestApicalSegment is not None:
        apicalLearningSegments.add(bestApicalSegment)

      chosenCellForColumn[column] = bestCell

    return (activeCells, winnerCells, learningSegments, apicalLearningSegments,
            chosenCellForColumn)


  def learnOnApicalSegments(self,
                            prevActiveSegments,
                            learningSegments,
                            prevActiveCells,
                            winnerCells,
                            connections,
                            predictedInactiveCells,
                            prevMatchingSegments):
    """
    Phase 3: Perform learning by adapting segments.

    Pseudocode:

      - (learning) for each prev active or learning segment
        - if learning segment or from winner cell
          - strengthen active synapses
          - weaken inactive synapses
        - if learning segment
          - add some synapses to the segment
            - subsample from prev winner cells

    @param prevActiveSegments           (set)         Indices of active segments in `t-1`
    @param learningSegments             (set)         Indices of learning segments in `t`
    @param prevActiveCells              (set)         Indices of active cells in `t-1`
    @param winnerCells                  (set)         Indices of winner cells in `t`
    @param connections                  (Connections) Connectivity of layer
    @param predictedInactiveCells       (set)         Indices of predicted inactive cells
    @param prevMatchingSegments         (set)         Indices of segments with
    """
    for winnerCell in winnerCells:
      winnerSegments = connections.segmentsForCell(winnerCell)
      if len(winnerSegments & (prevActiveSegments | learningSegments)) == 0:
        maxActiveSynapses = 0
        winnerSegment = None
        for segment in winnerSegments:
          activeSynapses = TemporalMemory.activeSynapsesForSegment(
              segment,
              prevActiveCells,
              connections)
          numActiveSynapses = len(activeSynapses)
          if numActiveSynapses > maxActiveSynapses:
            maxActiveSynapses = numActiveSynapses
            winnerSegment = segment
        if winnerSegment is not None:
          learningSegments.add(winnerSegment)

    for segment in prevActiveSegments | learningSegments:
      isLearningSegment = segment in learningSegments
      isFromWinnerCell = connections.cellForSegment(segment) in winnerCells

      activeSynapses = self.activeSynapsesForSegment(
        segment, prevActiveCells, connections)

      if isLearningSegment or isFromWinnerCell:
        self.adaptSegment(segment, activeSynapses, connections,
                          self.permanenceIncrement,
                          self.permanenceDecrement)

      if isLearningSegment:
        n = self.maxNewSynapseCount - len(activeSynapses)

        for presynapticCell in self.pickCellsToLearnOn(n,
                                                       segment,
                                                       prevActiveCells,
                                                       connections):
          connections.createSynapse(segment,
                                    presynapticCell,
                                    self.initialPermanence)


  def bestMatchingCell(self, cells, activeCells, activeApicalCells, connections, apicalConnections):
    """
    Gets the cell with the best matching segment
    (see `TM.bestMatchingSegment`) that has the largest number of active
    synapses of all best matching segments.

    If none were found, pick the least used cell (see `TM.leastUsedCell`).

    @param cells                       (set)         Indices of cells
    @param activeCells                 (set)         Indices of active cells
    @param activeApicalCells           (set)         Indices of active apical cells
    @param connections                 (Connections) Connectivity of layer
    @param apicalConnections           (Connections) Apical connectivity of layer

    @return (tuple) Contains:
                      `cell`                (int),
                      `bestSegment`         (int),
                      `bestApicalSegment` (int)
    """
    maxSynapses = 0
    bestCell = None
    bestSegment = None
    bestApicalSegment = None

    for cell in cells:
      segment, numActiveSynapses = self.bestMatchingSegment(
        cell, activeCells, connections)

      apicalSegment, apicalNumActiveSynapses = self.bestMatchingSegment(
        cell, activeApicalCells, apicalConnections)

      if segment is not None and numActiveSynapses > maxSynapses:
        maxSynapses = numActiveSynapses
        bestCell = cell
        bestSegment = segment
        bestApicalSegment = apicalSegment

    if bestCell is None:
      bestCell = self.leastUsedCell(cells, connections)

    return bestCell, bestSegment, bestApicalSegment


  def activeCellsIndices(self):
    """
    @return (set) Set of indices.
    """
    return self.activeCells


  def predictedActiveCellsIndices(self):
    """
    @return (set) Set of indices.
    """
    return self.predictedActiveCells


  def _reindexActiveCells(self, activeCells):
    """
    Move sensorimotor or apical input indices to outside the range of valid
    cell indices

    @params activeCells (set) Indices of active external cells in `t`
    """
    numCells = self.numberOfCells()
    return set([index + numCells for index in activeCells])


  def calculatePredictiveCells(self, predictiveDistalCells,
                               predictiveApicalCells):

    columnPredThresh = 2

    cellPredictiveScores = defaultdict(int)

    for candidate in predictiveApicalCells:
      cellPredictiveScores[candidate] += 1

    for candidate in predictiveDistalCells:
      cellPredictiveScores[candidate] += 2

    columnThresholds = defaultdict(int)

    for candidate in cellPredictiveScores:
      column = self.columnForCell(candidate)
      score = cellPredictiveScores[candidate]
      columnThresholds[column] = max(score, columnThresholds[column])

    predictiveCells = set()

    for candidate in cellPredictiveScores:
      column = self.columnForCell(candidate)
      if (columnThresholds[column] >= columnPredThresh) and (
            cellPredictiveScores[candidate] >= columnThresholds[column]):
        predictiveCells.add(candidate)

    return predictiveCells
