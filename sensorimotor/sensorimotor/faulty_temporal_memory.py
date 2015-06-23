# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
Faulty Temporal Memory implementation in Python.
"""
import numpy
from collections import defaultdict
from sensorimotor.general_temporal_memory import GeneralTemporalMemory
# from sensorimotor.orphan_temporal_memory import OrphanTemporalMemory

class FaultyTemporalMemory(GeneralTemporalMemory):
  """
  Class implementing a fallible Temporal Memory class. This class allows the
  user to kill a certain number of cells. The dead cells cannot become active,
  will no longer participate in predictions, and cannot become winning cells.
  While admittedly cruel and cold hearted, this feature  enables us to test the
  robustness of the Temporal Memory algorithm under such situations. Such is the
  price of progress.

  And by the way, we're not actually killing anything real.
  """

  # ==============================
  # Main functions
  # ==============================


  def __init__(self,
               **kwargs):
    super(FaultyTemporalMemory, self).__init__(**kwargs)
    self.deadCells = set()
    self.zombiePermutation = None    # Contains the order in which cells
                                      # will be killed
    self.numDead = 0


  def killCells(self, percent = 0.05):
    """
    Changes the percentage of cells that are now considered dead. The first
    time you call this method a permutation list is set up. Calls change the
    number of cells considered dead.
    """
    if self.zombiePermutation is None:
      self.zombiePermutation = numpy.random.permutation(self.numberOfCells())

    self.numDead = round(percent * self.numberOfCells())
    if self.numDead > 0:
      self.deadCells = set(self.zombiePermutation[0:self.numDead])
    else:
      self.deadCells = set()

    print "Total number of dead cells=",len(self.deadCells)


  # def computeFn(self,
  #               activeColumns,
  #               activeExternalCells,
  #               prevActiveExternalCells,
  #               prevPredictiveCells,
  #               prevActiveSegments,
  #               prevActiveCells,
  #               prevMatchingSegments,
  #               prevMatchingCells,
  #               prevWinnerCells,
  #               connections,
  #               formInternalConnections,
  #               learnOnOneCell,
  #               chosenCellForColumn,
  #               learn=True):
  #   """
  #   'Functional' version of compute.
  #   Returns new state.
  #
  #   @param activeColumns            (set)         Indices of active columns in `t`
  #   @param activeExternalCells      (set)         Indices of active external cells in `t`
  #   @param prevActiveExternalCells  (set)         Indices of active external cells in `t-1`
  #   @param prevPredictiveCells      (set)         Indices of predictive cells in `t-1`
  #   @param prevActiveSegments       (set)         Indices of active segments in `t-1`
  #   @param prevMatchingSegments     (set)         Indices of matching segments in `t-1`
  #   @param prevMatchingCells        (set)         Indices of matching cells in `t-1`
  #   @param prevActiveCells          (set)         Indices of active cells in `t-1`
  #   @param prevWinnerCells          (set)         Indices of winner cells in `t-1`
  #   @param connections              (Connections) Connectivity of layer
  #   @param formInternalConnections  (boolean)     Flag to determine whether to form connections with internal cells within this temporal memory
  #   @param learnOnOneCell           (boolean)     If True, the winner cell for each column will be fixed between resets.
  #   @param chosenCellForColumn      (dict)        The current winner cell for each column, if it exists.
  #
  #   @return (tuple) Contains:
  #                     `activeCells`               (set),
  #                     `winnerCells`               (set),
  #                     `activeSegments`            (set),
  #                     `predictiveCells`           (set)
  #   """
  #   activeCells = set()
  #   winnerCells = set()
  #
  #   (_activeCells,
  #    _winnerCells,
  #    predictedColumns,
  #    orphanCells) = self.activateCorrectlyPredictiveCells(
  #      prevPredictiveCells,
  #      prevMatchingCells,
  #      activeColumns)
  #
  #   activeCells.update(_activeCells)
  #   winnerCells.update(_winnerCells)
  #
  #   (_activeCells,
  #    _winnerCells,
  #    learningSegments,
  #    chosenCellForColumn) = self.burstColumns(
  #      activeColumns,
  #      predictedColumns,
  #      prevActiveCells | prevActiveExternalCells,
  #      prevWinnerCells,
  #      learnOnOneCell,
  #      chosenCellForColumn,
  #      connections)
  #
  #   activeCells.update(_activeCells)
  #   winnerCells.update(_winnerCells)
  #
  #   if learn:
  #     prevCellActivity = prevActiveExternalCells
  #
  #     if formInternalConnections:
  #       prevCellActivity.update(prevWinnerCells)
  #
  #     self.learnOnSegments(prevActiveSegments,
  #                          learningSegments,
  #                          prevActiveCells | prevActiveExternalCells,
  #                          winnerCells,
  #                          prevCellActivity,
  #                          connections,
  #                          orphanCells,
  #                          prevMatchingSegments)
  #
  #   (activeSegments,
  #    predictiveCells,
  #    matchingSegments,
  #    matchingCells) = self.computePredictiveCells(
  #      activeCells | activeExternalCells, connections)
  #
  #   return (activeCells,
  #           winnerCells,
  #           activeSegments,
  #           predictiveCells,
  #           predictedColumns,
  #           chosenCellForColumn,
  #           matchingSegments,
  #           matchingCells)


  def computePredictiveCells(self, activeCells, connections):
    """
    Phase 4: Compute predictive and matching cells due to lateral input
    on distal dendrites.

    Pseudocode:

      - for each distal dendrite segment with activity >= activationThreshold
        - mark the segment as active
        - mark the cell as predictive

      - for each distal dendrite segment with unconnected
        activity >=  minThreshold
        - mark the segment as matching
        - mark the cell as matching

    Forward propagates activity from active cells to the synapses that touch
    them, to determine which synapses are active.

    @param activeCells (set)         Indices of active cells in `t`
    @param connections (Connections) Connectivity of layer

    @return (tuple) Contains:
                      `activeSegments`   (set),
                      `predictiveCells`  (set),
                      `matchingSegments` (set),
                      `matchingCells`    (set)
    """
    numActiveConnectedSynapsesForSegment = defaultdict(lambda: 0)
    numActiveSynapsesForSegment = defaultdict(lambda: 0)
    activeSegments = set()
    predictiveCells = set()

    matchingSegments = set()
    matchingCells = set()

    for cell in activeCells:
      for synapseData in connections.synapsesForPresynapticCell(cell).values():
        segment = synapseData.segment
        permanence = synapseData.permanence
        postSynapticCell = connections.cellForSegment(segment)

        if permanence > 0 and not(postSynapticCell in self.deadCells):

          # Measure whether segment has sufficient weak activity, defined as
          # total active synapses >= minThreshold. A synapse is active if it
          # exists ( permanence > 0) and does not have to be connected.
          numActiveSynapsesForSegment[segment] += 1

          # Measure whether a segment has sufficient connected active
          # synapses to cause the cell to enter predicted mode.
          if permanence >= self.connectedPermanence:
            numActiveConnectedSynapsesForSegment[segment] += 1

            if (numActiveConnectedSynapsesForSegment[segment] >=
                self.activationThreshold):
              activeSegments.add(segment)
              predictiveCells.add(connections.cellForSegment(segment))

          if numActiveSynapsesForSegment[segment] >= self.minThreshold:
            matchingSegments.add(segment)
            matchingCells.add(connections.cellForSegment(segment))

    return activeSegments, predictiveCells, matchingSegments, matchingCells


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
      cells = self.cellsForColumn(column) - self.deadCells
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


  #########################################################################
  #
  # Debugging routines


  def printDeadCells(self):
    """
    Print statistics for the dead cells
    """
    columnCasualties = numpy.zeros(self.numberOfColumns())
    for cell in self.deadCells:
      col = self.columnForCell(cell)
      columnCasualties[col] += 1
    for col in range(self.numberOfColumns()):
      print col,columnCasualties[col]


  def printSegmentsForCell(self, cell):
    segments = self.connections.segmentsForCell(cell)

    print "Segments for cell",cell
    for segment in segments:
      self.printSegment(segment, self.connections)


  def printSegment(self, segment, connections):
    cell = connections.cellForSegment(segment)
    synapses = connections.synapsesForSegment(segment)
    print "segment id=",segment
    print "   cell=",cell
    print "   col =",self.columnForCell(cell)
    print "   synapses=",
    for synapse in synapses:
      synapseData = connections.dataForSynapse(synapse)
      permanence = synapseData.permanence
      presynapticCell = synapseData.presynapticCell
      print "%d:%g" % (presynapticCell,permanence),
    print


