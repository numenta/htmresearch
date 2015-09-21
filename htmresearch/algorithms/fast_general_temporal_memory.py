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
General Temporal Memory implementation in Python.
"""

from nupic.bindings.algorithms import Connections
from nupic.research.fast_temporal_memory import FastTemporalMemory, ConnectionsCell

from sensorimotor.general_temporal_memory import GeneralTemporalMemory



class FastGeneralTemporalMemory(GeneralTemporalMemory, FastTemporalMemory):
  """
  An implementation of GeneralTemporalMemory using C++ Connections data
  structure (from FastTemporalMemory) for more optimized performance.
  Note that while this implementation is faster than GeneralTemporalMemory,
  it is harder to debug because of the C++ code. In contrast,
  GeneralTemporalMemory is solely implemented in Python and will be easier to
  debug.
  """

  def __init__(self, *args, **kwargs):
    super(FastGeneralTemporalMemory, self).__init__(*args, **kwargs)
    self.apicalConnections = Connections(self.numberOfCells())


  def burstColumns(self,
                   activeColumns,
                   predictedColumns,
                   prevActiveCells,
                   prevActiveApicalCells,
                   prevWinnerCells,
                   learnOnOneCell,
                   chosenCellForColumn,
                   connections,
                   apicalConnections):
    """
    Phase 2: Burst unpredicted columns.

    TODO: Add functionality for feedback.

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

      bestSegment = connections.mostActiveSegmentForCells(
        list(cells), list(prevActiveCells), self.minThreshold)

      if bestSegment is None:
        bestCell = self.leastUsedCell(cells, connections)
        if len(prevWinnerCells):
          bestSegment = connections.createSegment(bestCell)
      else:
        # TODO: For some reason, bestSegment.cell is garbage-collected after
        # this function returns. So we have to use the below hack. Figure out
        # why and clean up.
        bestCell = ConnectionsCell(bestSegment.cell.idx)

      winnerCells.add(bestCell)

      if bestSegment is not None:
        learningSegments.add(bestSegment)

      chosenCellForColumn[column] = bestCell

    return activeCells, winnerCells, learningSegments, set(), chosenCellForColumn


  def activeCellsIndices(self):
    """
    @return (set) Set of indices.
    """
    return self._indicesForCells(self.activeCells)


  def predictedActiveCellsIndices(self):
    """
    @return (set) Set of indices.
    """
    return self._indicesForCells(self.predictedActiveCells)


  @staticmethod
  def _indicesForCells(cells):
    return set([cell.idx for cell in cells])


  def _reindexActiveExternalCells(self, activeExternalCells):
    """
    Move sensorimotor input indices to outside the range of valid cell indices

    @params activeExternalCells (set) Indices of active external cells in `t`
    """
    numCells = self.numberOfCells()

    def increaseIndexByNumberOfCells(index):
      return ConnectionsCell(index + numCells)

    return set(map(increaseIndexByNumberOfCells, activeExternalCells))
