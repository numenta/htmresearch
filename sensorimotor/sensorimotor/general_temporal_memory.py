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

from nupic.research.temporal_memory import TemporalMemory



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
     predictiveCells,
     predictedColumns,
     matchingSegments,
     matchingCells,
     chosenCellForColumn) = self.computeFn(activeColumns,
                                           activeExternalCells,
                                           self.activeExternalCells,
                                           self.predictiveCells,
                                           self.activeSegments,
                                           self.activeCells,
                                           self.winnerCells,
                                           self.matchingSegments,
                                           self.matchingCells,
                                           self.connections,
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
    self.predictiveCells = predictiveCells
    self.chosenCellForColumn = chosenCellForColumn
    self.matchingSegments = matchingSegments
    self.matchingCells = matchingCells

  def computeFn(self,
                activeColumns,
                activeExternalCells,
                prevActiveExternalCells,
                prevPredictiveCells,
                prevActiveSegments,
                prevActiveCells,
                prevWinnerCells,
                prevMatchingSegments,
                prevMatchingCells,
                connections,
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
    @param prevActiveCells                 (set)         Indices of active cells in `t-1`
    @param prevWinnerCells                 (set)         Indices of winner cells in `t-1`
    @param prevMatchingSegments            (set)         Indices of matching segments in `t-1`
    @param prevMatchingCells               (set)         Indices of matching cells in `t-1`
    @param connections                     (Connections) Connectivity of layer
    @param formInternalConnections         (boolean)     Flag to determine whether to form connections with internal cells within this temporal memory
    @param learnOnOneCell                  (boolean)     If True, the winner cell for each column will be fixed between resets.
    @param chosenCellForColumn             (dict)        The current winner cell for each column, if it exists.

    @return (tuple) Contains:
                      `activeCells`               (set),
                      `winnerCells`               (set),
                      `activeSegments`            (set),
                      `predictiveCells`           (set),
                      'predictedColumns'          (set),
                      'matchingSegments'          (set),
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
     chosenCellForColumn) = self.burstColumns(
       activeColumns,
       predictedColumns,
       prevActiveCells | prevActiveExternalCells,
       prevWinnerCells,
       learnOnOneCell,
       chosenCellForColumn,
       connections)

    activeCells.update(_activeCells)
    winnerCells.update(_winnerCells)

    if learn:
      prevCellActivity = prevActiveExternalCells

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

    (activeSegments,
     predictiveCells,
     matchingSegments,
     matchingCells) = self.computePredictiveCells(
       activeCells | activeExternalCells, connections)

    return (activeCells,
            winnerCells,
            activeSegments,
            predictiveCells,
            predictedColumns,
            matchingSegments,
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
                   prevWinnerCells,
                   learnOnOneCell,
                   chosenCellForColumn,
                   connections):
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
    @param prevWinnerCells                 (set)         Indices of winner cells in `t-1`
    @param learnOnOneCell                  (boolean)     If True, the winner cell for each column will be fixed between resets.
    @param chosenCellForColumn             (dict)        The current winner cell for each column, if it exists.
    @param connections                     (Connections) Connectivity of layer

    @return (tuple) Contains:
                      `activeCells`      (set),
                      `winnerCells`      (set),
                      `learningSegments` (set)
    """
    activeCells = set()
    winnerCells = set()
    learningSegments = set()

    unpredictedColumns = activeColumns - predictedColumns

    for column in unpredictedColumns:
      cells = self.cellsForColumn(column)
      activeCells.update(cells)

      if learnOnOneCell and (column in chosenCellForColumn):
        chosenCell = chosenCellForColumn[column]
        cells = set([chosenCell])

      (bestCell,
       bestSegment) = self.bestMatchingCell(cells,
                                            prevActiveCells,
                                            connections)
      winnerCells.add(bestCell)

      if bestSegment is None and len(prevWinnerCells):
        bestSegment = connections.createSegment(bestCell)

      if bestSegment is not None:
        learningSegments.add(bestSegment)

      chosenCellForColumn[column] = bestCell

    return activeCells, winnerCells, learningSegments, chosenCellForColumn


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
