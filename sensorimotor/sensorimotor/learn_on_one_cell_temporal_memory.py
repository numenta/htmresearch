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
Temporal Memory subclass with Learn-On-One-Cell learning rule implementation in
Python.
"""

from sensorimotor.general_temporal_memory import GeneralTemporalMemory


class LearnOnOneCellTemporalMemory(GeneralTemporalMemory):
  """
  Class implementing the Temporal Memory algorithm with a change in the learning
  rule. This class prefers to learn on a single cell within a column and does
  not look for a new cell to learn with until a reset() is called.
  """

  # ==============================
  # Main functions
  # ==============================

  def __init__(self, **kwargs):
    super(LearnOnOneCellTemporalMemory, self).__init__(**kwargs)

    self.chosenCellForColumn = dict()


  def reset(self):
    """
    Indicates the start of a new sequence. Resets sequence state of the TM.
    """
    super(LearnOnOneCellTemporalMemory, self).reset()

    self.chosenCellForColumn = dict()

  # ==============================
  # Helper functions
  # ==============================

  def burstColumns(self,
                   activeColumns,
                   predictedColumns,
                   prevActiveSynapsesForSegment,
                   connections):
    """
    Phase 2: Burst unpredicted columns.

    Pseudocode:

      - for each unpredicted active column
        - mark all cells as active
        - keep the old best matching cell if it exists
        - mark the best matching cell as winner cell
          - (learning)
            - if it has no matching segment
              - (optimization) if there are prev winner cells
                - add a segment to it
            - mark the segment as learning

    @param activeColumns                (set)         Indices of active columns
                                                      in `t`
    @param predictedColumns             (set)         Indices of predicted
                                                      columns in `t`
    @param prevActiveSynapsesForSegment (dict)        Mapping from segments to
                                                      active synapses in `t-1`,
                                                      see
                                                      `TM.computeActiveSynapses`
    @param connections                  (Connections) Connectivity of layer

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
      cells = connections.cellsForColumn(column)
      activeCells.update(cells)

      if column in self.chosenCellForColumn:
        chosenCell = self.chosenCellForColumn[column]
        cells = set([chosenCell])

      (bestCell,
       bestSegment) = self.getBestMatchingCell(cells,
                                               prevActiveSynapsesForSegment,
                                               connections)
      winnerCells.add(bestCell)

      if bestSegment == None:
        # TODO: (optimization) Only do this if there are prev winner cells
        bestSegment = connections.createSegment(bestCell)

      learningSegments.add(bestSegment)

      self.chosenCellForColumn[column] = bestCell

    return (activeCells, winnerCells, learningSegments)
