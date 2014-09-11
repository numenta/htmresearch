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
  being able to learn from both internal and external cell activation.
  """

  # ==============================
  # Main functions
  # ==============================


  def __init__(self,
             **kwargs):
    super(GeneralTemporalMemory, self).__init__(**kwargs)

    self.activeExternalCells = set()


  def compute(self,
              activeColumns,
              activeExternalCells=None,
              formInternalConnections=True,
              learn=True):
    """
    Feeds input record through TM, performing inference and learning.
    Updates member variables with new state.

    @param activeColumns            (set)     Indices of active columns in `t`

    @param activeExternalCells      (set)     Indices of active external inputs
                                              in `t`

    @param formInternalConnections  (boolean) Flag to determine whether to form
                                              connections with internal cells
                                              within this temporal memory
    """

    if not activeExternalCells:
      activeExternalCells = set()

    activeExternalCells = self.reindexActiveExternalCells(activeExternalCells)

    (activeCells,
     winnerCells,
     activeSynapsesForSegment,
     activeSegments,
     predictiveCells) = self.computeFn(activeColumns,
                                       activeExternalCells,
                                       self.activeExternalCells,
                                       self.predictiveCells,
                                       self.activeSegments,
                                       self.activeSynapsesForSegment,
                                       self.winnerCells,
                                       self.connections,
                                       formInternalConnections,
                                       learn=learn)


    self.activeColumns = activeColumns
    self.prevPredictiveCells = self.predictiveCells
    self.activeExternalCells = activeExternalCells

    self.activeCells = activeCells
    self.winnerCells = winnerCells
    self.activeSynapsesForSegment = activeSynapsesForSegment
    self.activeSegments = activeSegments
    self.predictiveCells = predictiveCells


  def computeFn(self,
                activeColumns,
                activeExternalCells,
                prevActiveExternalCells,
                prevPredictiveCells,
                prevActiveSegments,
                prevActiveSynapsesForSegment,
                prevWinnerCells,
                connections,
                formInternalConnections,
                learn=True):
    """
    'Functional' version of compute.
    Returns new state.

    @param activeColumns                (set)         Indices of active columns
                                                      in `t`

    @param activeExternalCells          (set)         Indices of active external
                                                      cells in `t`

    @param prevActiveExternalCells      (set)         Indices of active external
                                                      cells i `t-1`

    @param prevPredictiveCells          (set)         Indices of predictive
                                                      cells in `t-1`
    @param prevActiveSegments           (set)         Indices of active segments
                                                      in `t-1`
    @param prevActiveSynapsesForSegment (dict)        Mapping from segments to
                                                      active synapses in `t-1`,
                                                      see
                                                      `TM.computeActiveSynapses`
    @param prevWinnerCells              (set)         Indices of winner cells
                                                      in `t-1`
    @param connections                  (Connections) Connectivity of layer

    @param formInternalConnections      (boolean)     Flag to determine whether
                                                      to form connections with
                                                      internal cells within this
                                                      temporal memory

    @return (tuple) Contains:
                      `activeCells`               (set),
                      `winnerCells`               (set),
                      `activeSynapsesForSegment`  (dict),
                      `activeSegments`            (set),
                      `predictiveCells`           (set)
    """
    activeCells = set()
    winnerCells = set()

    (_activeCells,
     _winnerCells,
     predictedColumns) = self.activateCorrectlyPredictiveCells(
       prevPredictiveCells,
       activeColumns,
       connections)

    activeCells.update(_activeCells)
    winnerCells.update(_winnerCells)

    (_activeCells,
     _winnerCells,
     learningSegments) = self.burstColumns(activeColumns,
                                           predictedColumns,
                                           prevActiveSynapsesForSegment,
                                           connections)

    activeCells.update(_activeCells)
    winnerCells.update(_winnerCells)

    if learn:
      prevCellActivity = prevActiveExternalCells

      if formInternalConnections:
        prevCellActivity.update(prevWinnerCells)

      self.learnOnSegments(prevActiveSegments,
                           learningSegments,
                           prevActiveSynapsesForSegment,
                           winnerCells,
                           prevCellActivity,
                           connections)

    activeSynapsesForSegment = self.computeActiveSynapses(
                                              activeExternalCells | activeCells,
                                              connections)

    (activeSegments,
     predictiveCells) = self.computePredictiveCells(activeSynapsesForSegment,
                                                    connections)

    return (activeCells,
            winnerCells,
            activeSynapsesForSegment,
            activeSegments,
            predictiveCells)


  def reset(self):
    super(GeneralTemporalMemory, self).reset()

    self.activeExternalCells = set()

  # ==============================
  # Helper functions
  # ==============================

  def reindexActiveExternalCells(self, activeExternalCells):
    """
    Move sensorimotor input indices to outside the range of valid cell indices

    @params activeExternalCells     (set)   Indices of active external cells in
                                            `t`
    """
    numCells = self.connections.numberOfCells()

    def increaseIndexByNumberOfCells(index):
      return index + numCells

    return set(map(increaseIndexByNumberOfCells, activeExternalCells))
