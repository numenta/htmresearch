# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015-2016, Numenta, Inc.  Unless you have an agreement
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
from nupic.research.temporal_memory import TemporalMemory



class FaultyTemporalMemory(TemporalMemory):
  """
  Class implementing a fallible Temporal Memory class. This class allows the
  user to kill a certain number of cells. The dead cells cannot become active,
  will no longer participate in predictions, and cannot become winning cells.
  """

  def __init__(self,
               **kwargs):
    super(FaultyTemporalMemory, self).__init__(**kwargs)
    self.deadCells = set()
    self.zombiePermutation = None # Contains the order in which cells
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

    for cellIdx in self.deadCells:
      # Destroy segments.  self.destroySegment() takes care of deleting synapses
      for segment in self.connections.segmentsForCell(cellIdx):
        self.connections.destroySegment(segment)
        numSegmentDeleted += 1

    print "Total number of segments removed=", numSegmentDeleted


  def burstColumn(self, connections, random, column, columnMatchingSegments,
                  prevActiveCells, prevWinnerCells,
                  numActivePotentialSynapsesForSegment, cellsPerColumn,
                  maxNewSynapseCount, initialPermanence, permanenceIncrement,
                  permanenceDecrement, learn):
    """ Originally copied from temporal_memory.py and augmented to ignore dead
    cells.  See TemporalMemory.burstColumns() for original implementation.
    """
    start = cellsPerColumn * column
    cells = xrange(start, start + cellsPerColumn)

    # Strip out destroyed cells
    cells = [cellIdx
             for cellIdx
             in cells
             if cellIdx not in self.deadCells]

    if columnMatchingSegments is not None:
      numActive = lambda s: numActivePotentialSynapsesForSegment[s.flatIdx]
      bestMatchingSegment = max(columnMatchingSegments, key=numActive)
      winnerCell = bestMatchingSegment.cell

      if learn:
        self.adaptSegment(connections, bestMatchingSegment, prevActiveCells,
                          permanenceIncrement, permanenceDecrement)

        nGrowDesired = maxNewSynapseCount - numActive(bestMatchingSegment)

        if nGrowDesired > 0:
          self.growSynapses(connections, random, bestMatchingSegment,
                            nGrowDesired, prevWinnerCells, initialPermanence)
    else:
      winnerCell = self.leastUsedCell(random, cells, connections)
      if learn:
        nGrowExact = min(maxNewSynapseCount, len(prevWinnerCells))
        if nGrowExact > 0:
          segment = connections.createSegment(winnerCell)
          self.growSynapses(connections, random, segment, nGrowExact,
                            prevWinnerCells, initialPermanence)

    return cells, winnerCell


  def printDeadCells(self):
    """
    Print statistics for the dead cells
    """
    columnCasualties = numpy.zeros(self.numberOfColumns())
    for cell in self.deadCells:
      col = self.columnForCell(cell)
      columnCasualties[col] += 1
    for col in range(self.numberOfColumns()):
      print col, columnCasualties[col]


  def printSegmentsForCell(self, cell):
    segments = self.connections.segmentsForCell(cell)

    print "Segments for cell", cell
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