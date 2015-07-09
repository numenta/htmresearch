# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
General Temporal Memory implementation in Python.
"""

from collections import defaultdict, namedtuple

from nupic.research.temporal_memory import TemporalMemory
from nupic.research.connections import Connections


class GeneralTemporalMemory(TemporalMemory):
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
               externalActivationThreshold=26,
               **kwargs):
    """
    @param learnOnOneCell (boolean) If True, the winner cell for each column will be fixed between resets.
    """

    super(GeneralTemporalMemory, self).__init__(**kwargs)

    self.activeExternalCells = set()
    self.learnOnOneCell = learnOnOneCell
    self.chosenCellForColumn = dict()

    self.unpredictedActiveColumns = set()
    self.predictedActiveCells = set()

    self.externalConnections = Connections(self.numberOfCells())
    self.externalActivationThreshold = externalActivationThreshold
    self.activeExternalSegments = set()
    self.matchingExternalSegments = set()


  def compute(self,
              activeColumns,
              activeExternalCells=None,
              formInternalConnections=True,
              learn=True):
    """
    Feeds input record through TM, performing inference and learning.
    Updates member variables with new state.

    @param activeColumns           (set)     Indices of active columns in `t`
    @param activeExternalCells     (set)     Indices of active external inputs in `t`
    @param formInternalConnections (boolean) Flag to determine whether to form connections with internal cells within this temporal memory
    """

    if not activeExternalCells:
      activeExternalCells = set()

    activeExternalCells = self._reindexActiveExternalCells(activeExternalCells)

    (activeCells,
     winnerCells,
     activeSegments,
     activeExternalSegments,
     predictiveCells,
     predictedColumns,
     matchingSegments,
     matchingExternalSegments,
     matchingCells,
     chosenCellForColumn) = self.computeFn(activeColumns,
                                           activeExternalCells,
                                           self.activeExternalCells,
                                           self.predictiveCells,
                                           self.activeSegments,
                                           self.activeExternalSegments,
                                           self.activeCells,
                                           self.winnerCells,
                                           self.matchingSegments,
                                           self.matchingExternalSegments,
                                           self.matchingCells,
                                           self.connections,
                                           self.externalConnections,
                                           formInternalConnections,
                                           self.learnOnOneCell,
                                           self.chosenCellForColumn,
                                           learn=learn)

    self.activeExternalCells = activeExternalCells

    self.unpredictedActiveColumns = activeColumns - predictedColumns
    self.predictedActiveCells = self.predictiveCells & activeCells

    self.activeCells = activeCells
    self.winnerCells = winnerCells
    self.activeSegments = activeSegments
    self.activeExternalSegments = activeExternalSegments
    self.predictiveCells = predictiveCells
    self.chosenCellForColumn = chosenCellForColumn
    self.matchingSegments = matchingSegments
    self.matchingExternalSegments = matchingExternalSegments
    self.matchingCells = matchingCells

  def computeFn(self,
                activeColumns,
                activeExternalCells,
                prevActiveExternalCells,
                prevPredictiveCells,
                prevActiveSegments,
                prevActiveExternalSegments,
                prevActiveCells,
                prevWinnerCells,
                prevMatchingSegments,
                prevMatchingExternalSegments,
                prevMatchingCells,
                connections,
                externalConnections,
                formInternalConnections,
                learnOnOneCell,
                chosenCellForColumn,
                learn=True):
    """
    'Functional' version of compute.
    Returns new state.

    @param activeColumns                   (set)         Indices of active columns in `t`
    @param activeExternalCells             (set)         Indices of active external cells in `t`
    @param prevActiveExternalCells         (set)         Indices of active external cells in `t-1`
    @param prevPredictiveCells             (set)         Indices of predictive cells in `t-1`
    @param prevActiveSegments              (set)         Indices of active segments in `t-1`
    @param prevActiveExternalSegments      (set)         Indices of active external segments in `t-1`
    @param prevActiveCells                 (set)         Indices of active cells in `t-1`
    @param prevWinnerCells                 (set)         Indices of winner cells in `t-1`
    @param prevMatchingSegments            (set)         Indices of matching segments in `t-1`
    @param prevMatchingExternalSegments    (set)         Indices of matching external segments in `t-1`
    @param prevMatchingCells               (set)         Indices of matching cells in `t-1`
    @param connections                     (Connections) Connectivity of layer
    @param externalConnections             (Connections) External connectivity of layer
    @param formInternalConnections         (boolean)     Flag to determine whether to form connections with internal cells within this temporal memory
    @param learnOnOneCell                  (boolean)     If True, the winner cell for each column will be fixed between resets.
    @param chosenCellForColumn             (dict)        The current winner cell for each column, if it exists.

    @return (tuple) Contains:
                      `activeCells`               (set),
                      `winnerCells`               (set),
                      `activeApicalSegments`      (set),
                      `activeDistalSegments`      (set),
                      `predictiveCells`           (set),
                      'predictedColumns'          (set),
                      'matchingApicalSegments'    (set),
                      'matchingDistalSegments'    (set),
                      'matchingCells'             (set),
                      'chosenCellForColumn'       (dict)
    """
    activeCells = set()
    winnerCells = set()

    (_activeCells,
     _winnerCells,
     predictedColumns,
     predictedInactiveCells) = self.activateCorrectlyPredictiveCells(
       prevPredictiveCells,
       prevMatchingCells,
       activeColumns)

    activeCells.update(_activeCells)
    winnerCells.update(_winnerCells)

    (_activeCells,
     _winnerCells,
     learningSegments,
     externalLearningSegments,
     chosenCellForColumn) = self.burstColumns(
       activeColumns,
       predictedColumns,
       prevActiveCells,
       prevActiveExternalCells,
       prevWinnerCells,
       learnOnOneCell,
       chosenCellForColumn,
       connections,
       externalConnections)

    activeCells.update(_activeCells)
    winnerCells.update(_winnerCells)

    if learn:
      self.learnOnExternalSegments(prevActiveExternalSegments,
                           externalLearningSegments,
                           prevActiveExternalCells,
                           winnerCells,
                           externalConnections,
                           predictedInactiveCells,
                           prevMatchingExternalSegments)


      if formInternalConnections:
        self.learnOnSegments(prevActiveSegments,
                             learningSegments,
                             prevActiveCells,
                             winnerCells,
                             prevWinnerCells,
                             connections,
                             predictedInactiveCells,
                             prevMatchingSegments)

    (activeDistalSegments,
    predictiveDistalCells,
    matchingDistalSegments,
    matchingDistalCells) = self.computePredictiveCells(
    activeCells, connections)

    (activeApicalSegments,
    predictiveApicalCells,
    matchingApicalSegments,
    matchingApicalCells) = self.computeExternalPredictiveCells(
    activeExternalCells, externalConnections)

    matchingCells = matchingDistalCells | matchingApicalCells

    # There is probably a way to speed up the below
    cellPredictiveScores = defaultdict(int)

    for candidate in predictiveApicalCells:
      cellPredictiveScores[candidate] += 1

    for candidate in predictiveDistalCells:
      cellPredictiveScores[candidate] += 1

    columnThresholds = defaultdict(int)

    for candidate in cellPredictiveScores:
      column = self.columnForCell(candidate)
      score = cellPredictiveScores[candidate]
      columnThresholds[column] = max(score, columnThresholds[column])

    predictiveCells = set()

    for candidate in cellPredictiveScores:
      column = self.columnForCell(candidate)
      if cellPredictiveScores[candidate] >= columnThresholds[column]:
        predictiveCells.add(candidate)

    return (activeCells,
            winnerCells,
            activeDistalSegments,
            activeApicalSegments,
            predictiveCells,
            predictedColumns,
            matchingDistalSegments,
            matchingApicalSegments,
            matchingCells,
            chosenCellForColumn)


  def reset(self):
    super(GeneralTemporalMemory, self).reset()

    self.activeExternalCells = set()
    self.chosenCellForColumn = dict()

    self.unpredictedActiveColumns = set()
    self.predictedActiveCells = set()


  def burstColumns(self,
                   activeColumns,
                   predictedColumns,
                   prevActiveCells,
                   prevActiveExternalCells,
                   prevWinnerCells,
                   learnOnOneCell,
                   chosenCellForColumn,
                   connections,
                   externalConnections):
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
    @param predictedColumns                (set)         Indices of predicted columns in `t`
    @param prevActiveCells                 (set)         Indices of active cells in `t-1`
    @param prevActiveExternalCells         (set)         Indices of ext active cells in `t-1`
    @param prevWinnerCells                 (set)         Indices of winner cells in `t-1`
    @param learnOnOneCell                  (boolean)     If True, the winner cell for each column will be fixed between resets.
    @param chosenCellForColumn             (dict)        The current winner cell for each column, if it exists.
    @param connections                     (Connections) Connectivity of layer
    @param externalConnections             (Connections) External connectivity of layer

    @return (tuple) Contains:
                      `activeCells`      (set),
                      `winnerCells`      (set),
                      `learningSegments` (set),
                      `externalLearningSegments` (set)
    """
    activeCells = set()
    winnerCells = set()
    learningSegments = set()
    externalLearningSegments = set()

    unpredictedColumns = activeColumns - predictedColumns

    for column in unpredictedColumns:
      cells = self.cellsForColumn(column)
      activeCells.update(cells)

      if learnOnOneCell and (column in chosenCellForColumn):
        chosenCell = chosenCellForColumn[column]
        cells = set([chosenCell])

      (bestCell,
       bestSegment,
       bestExternalSegment) = self.bestMatchingCell(cells,
                                            prevActiveCells,
                                            prevActiveExternalCells,
                                            connections,
                                            externalConnections)
      winnerCells.add(bestCell)

      if bestSegment is None and len(prevWinnerCells):
        bestSegment = connections.createSegment(bestCell)

      if bestExternalSegment is None:
        bestExternalSegment = externalConnections.createSegment(bestCell)

      if bestSegment is not None:
        learningSegments.add(bestSegment)

      if bestExternalSegment is not None:
        externalLearningSegments.add(bestExternalSegment)

      chosenCellForColumn[column] = bestCell

    return activeCells, winnerCells, learningSegments, externalLearningSegments, chosenCellForColumn

  def learnOnExternalSegments(self,
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
          learningSegments |= winnerSegments

    for segment in prevActiveSegments | learningSegments:
      isLearningSegment = segment in learningSegments
      isFromWinnerCell = connections.cellForSegment(segment) in winnerCells

      activeSynapses = self.activeSynapsesForSegment(
        segment, prevActiveCells, connections)

      if isLearningSegment or isFromWinnerCell:
        self.adaptSegment(segment, activeSynapses, connections,
                          self.permanenceIncrement/2.0, 0)

      if isLearningSegment:
        n = self.maxNewSynapseCount - len(activeSynapses)

        for presynapticCell in self.pickCellsToLearnOn(n,
                                                       segment,
                                                       prevActiveCells,
                                                       connections):
          connections.createSynapse(segment,
                                    presynapticCell,
                                    self.initialPermanence)

  def computeExternalPredictiveCells(self, activeCells, connections):
    """
    Phase 4: Compute predictive cells due to lateral input
    on distal dendrites.

    Pseudocode:

      - for each distal dendrite segment with activity >= activationThreshold
        - mark the segment as active
        - mark the cell as predictive

      - if predictedSegmentDecrement > 0
        - for each distal dendrite segment with unconnected
          activity >=  minThreshold
          - mark the segment as matching
          - mark the cell as matching

    Forward propagates activity from active cells to the synapses that touch
    them, to determine which synapses are active.

    @param activeCells (set)         Indices of active cells in `t`
    @param connections (Connections) Connectivity of layer

    @return (tuple) Contains:
                      `activeSegments`  (set),
                      `predictiveCells` (set),
                      `matchingSegments` (set),
                      `matchingCells`    (set)
    """
    numActiveConnectedSynapsesForSegment = defaultdict(int)
    numActiveSynapsesForSegment = defaultdict(int)
    activeSegments = set()
    predictiveCells = set()

    matchingSegments = set()
    matchingCells = set()

    for cell in activeCells:
      for synapseData in connections.synapsesForPresynapticCell(cell).values():
        segment = synapseData.segment
        permanence = synapseData.permanence

        if permanence >= self.connectedPermanence:
          numActiveConnectedSynapsesForSegment[segment] += 1

          if (numActiveConnectedSynapsesForSegment[segment] >=
              self.externalActivationThreshold):
            activeSegments.add(segment)
            predictiveCells.add(connections.cellForSegment(segment))

        if permanence > 0 and self.predictedSegmentDecrement > 0:
          numActiveSynapsesForSegment[segment] += 1

          if numActiveSynapsesForSegment[segment] >= self.minThreshold:
            matchingSegments.add(segment)
            matchingCells.add(connections.cellForSegment(segment))

    return activeSegments, predictiveCells, matchingSegments, matchingCells

  def bestMatchingCell(self, cells, activeCells, activeExternalCells, connections, externalConnections):
    """
    Gets the cell with the best matching segment
    (see `TM.bestMatchingSegment`) that has the largest number of active
    synapses of all best matching segments.

    If none were found, pick the least used cell (see `TM.leastUsedCell`).

    @param cells                       (set)         Indices of cells
    @param activeCells                 (set)         Indices of active cells
    @param activeExternalCells         (set)         Indices of active external cells
    @param connections                 (Connections) Connectivity of layer
    @param externalConnections         (Connections) External connectivity of layer

    @return (tuple) Contains:
                      `cell`                (int),
                      `bestSegment`         (int),
                      `bestExternalSegment` (int)
    """
    maxSynapses = 0
    bestCell = None
    bestSegment = None
    bestExternalSegment = None

    for cell in cells:
      segment, numActiveSynapses = self.bestMatchingSegment(
        cell, activeCells, connections)

      externalSegment, externalNumActiveSynapses = self.bestMatchingSegment(
        cell, activeExternalCells, externalConnections)

      if segment is not None and numActiveSynapses > maxSynapses:
        maxSynapses = numActiveSynapses
        bestCell = cell
        bestSegment = segment
        bestExternalSegment = externalSegment

    if bestCell is None:
      bestCell = self.leastUsedCell(cells, connections)

    return bestCell, bestSegment, bestExternalSegment


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


  def _reindexActiveExternalCells(self, activeExternalCells):
    """
    Move sensorimotor input indices to outside the range of valid cell indices

    @params activeExternalCells (set) Indices of active external cells in `t`
    """
    numCells = self.numberOfCells()
    return set([index + numCells for index in activeExternalCells])
