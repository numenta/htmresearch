# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
Faulty Temporal Memory implementation in Python.
"""
import numpy
from collections import defaultdict
from temporal_memory_phases import TemporalMemory

class FaultyTemporalMemory(TemporalMemory):
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

    numSegmentDeleted = 0
    for cell in self.deadCells:
      segmentsPerCell = list(self.connections.segmentsForCell(cell))
      for segment in segmentsPerCell:
        self.connections.destroySegment(segment)
        numSegmentDeleted += 1

    print "Total number of segments removed=", numSegmentDeleted


  def activateCorrectlyPredictiveCells(self,
                                       prevPredictiveCells,
                                       prevMatchingCells,
                                       activeColumns):
    """
    Phase 1: Activate the correctly predictive cells.

    Pseudocode:

      - for each prev predictive cell
        - if in active column
          - mark it as active
          - mark it as winner cell
          - mark column as predicted => active
        - if not in active column
          - mark it as an predicted but inactive cell

    @param prevPredictiveCells (set) Indices of predictive cells in `t-1`
    @param activeColumns       (set) Indices of active columns in `t`

    @return (tuple) Contains:
                      `activeCells`               (set),
                      `winnerCells`               (set),
                      `predictedActiveColumns`    (set),
                      `predictedInactiveCells`    (set)
    """
    activeCells = set()
    winnerCells = set()
    predictedActiveColumns = set()
    predictedInactiveCells = set()

    prevPredictiveCells = prevPredictiveCells - self.deadCells
    prevMatchingCells = prevMatchingCells - self.deadCells

    for cell in prevPredictiveCells:
      column = self.columnForCell(cell)

      if column in activeColumns:
        activeCells.add(cell)
        winnerCells.add(cell)
        predictedActiveColumns.add(column)

    if self.predictedSegmentDecrement > 0:
      for cell in prevMatchingCells:
        column = self.columnForCell(cell)

        if column not in activeColumns:
          predictedInactiveCells.add(cell)

    return (activeCells,
            winnerCells,
            predictedActiveColumns,
            predictedInactiveCells)


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
      assert(not (cell in self.deadCells))
      for synapseData in connections.synapsesForPresynapticCell(cell).values():
        segment = synapseData.segment
        permanence = synapseData.permanence
        postSynapticCell = connections.cellForSegment(segment)

        if not(postSynapticCell in self.deadCells):
          if permanence >= self.connectedPermanence:
            numActiveConnectedSynapsesForSegment[segment] += 1

            if (numActiveConnectedSynapsesForSegment[segment] >=
                self.activationThreshold):
              activeSegments.add(segment)
              predictiveCells.add(connections.cellForSegment(segment))

          if permanence > 0 and self.predictedSegmentDecrement > 0:
            numActiveSynapsesForSegment[segment] += 1

            if numActiveSynapsesForSegment[segment] >= self.minThreshold:
              matchingSegments.add(segment)
              matchingCells.add(connections.cellForSegment(segment))


    return activeSegments, predictiveCells, matchingSegments, matchingCells


  def burstColumns(self,
                   activeColumns,
                   predictedActiveColumns,
                   prevActiveCells,
                   prevWinnerCells,
                   connections):
    """
    Phase 2: Burst unpredicted columns.

    Pseudocode:

      - for each unpredicted active column
        - mark all cells as activex
        - If learnOnOneCell, keep the old best matching cell if it exists
        - mark the best matching cell as winner cell
          - (learning)
            - if it has no matching segment
              - (optimization) if there are prev winner cells
                - add a segment to it
            - mark the segment as learning

    @param activeColumns                   (set)        Indices of active
                                                        columns in `t`
    @param predictedActiveColumns          (set)        Indices of predicted
                                                        columns in `t`
    @param prevActiveCells                 (set)        Indices of active cells
                                                        in `t-1`
    @param prevWinnerCells                 (set)        Indices of winner cells
                                                        in `t-1`
    @param connections                     (Connections) Connectivity of layer

    @return (tuple) Contains:
                      `activeCells`      (set),
                      `winnerCells`      (set),
                      `learningSegments` (set)
    """
    activeCells = set()
    winnerCells = set()
    learningSegments = set()

    unpredictedActiveColumns = activeColumns - predictedActiveColumns

    for column in unpredictedActiveColumns:
      cells = self.cellsForColumn(column) - self.deadCells
      if len(cells) == 0:
        continue

      activeCells.update(cells)

      # if learnOnOneCell and (column in chosenCellForColumn):
      #   chosenCell = chosenCellForColumn[column]
      #   cells = set([chosenCell])

      (bestCell,
       bestSegment) = self.bestMatchingCell(cells,
                                            prevActiveCells,
                                            connections)
      winnerCells.add(bestCell)

      if bestSegment is None and len(prevWinnerCells):
        bestSegment = connections.createSegment(bestCell)

      if bestSegment is not None:
        learningSegments.add(bestSegment)

      # chosenCellForColumn[column] = bestCell

    return activeCells, winnerCells, learningSegments


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


