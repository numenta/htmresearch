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
from nupic.research.temporal_memory import  TemporalMemory
from nupic.research.connections import CellData, Connections



class FaultyCellData(CellData):
  """ CellData subclass that adds a `destroyed` property that evaluates to True
  when a cell is "killed"
  """


  __slots__ = CellData.__slots__ + ["__destroyed"]


  def __init__(self):
    super(FaultyCellData, self).__init__()
    self.__destroyed = False


  @property
  def destroyed(self):
    return self.__destroyed


  def destroy(self):
    self.__destroyed = True # Destruction is permanent!



class FaultyConnections(Connections):
  def __init__(self,
               numCells,
               maxSegmentsPerCell=255,
               maxSynapsesPerSegment=255):
    super(FaultyConnections, self).__init__(
      numCells=numCells,
      maxSegmentsPerCell=maxSegmentsPerCell,
      maxSynapsesPerSegment=maxSynapsesPerSegment)

    # Redo _cells assignment to use FaultyCellData, which supports destruction
    # of cells
    self._cells = [FaultyCellData() for _ in xrange(numCells)]


  @property
  def cells(self):
    # Expose otherwise internal _cells as a property
    return self._cells


  def destroyCell(self, cellIdx):

    # Destroy cell
    self.cells[cellIdx].destroy()

    # Destroy segments.  self.destroySegment() takes care of deleting synapses
    segmentCount = 0
    for segment in self.segmentsForCell(cellIdx):
      self.destroySegment(segment)
      segmentCount += 1

    return segmentCount


  def segmentsForCell(self, cell):
    # Completely ignore segments for destroyed cells
    if self.cells[cell].destroyed:
      return iter([])

    return super(FaultyConnections, self).segmentsForCell(cell)


  def synapsesForSegment(self, segment):
    # Completely ignore synapses on destroyed cells and destroyed segments
    if self.cells[segment.cell].destroyed or segment._destroyed:
      return iter([])

    # Return only synapses connected to non-destroyed presynaptic cells
    return (synapse
            for synapse
            in super(FaultyConnections, self).synapsesForSegment(segment)
            if not self.cells[synapse.presynapticCell].destroyed)


  def createSynapse(self, segment, presynapticCell, permanence):
    # Do NOT create synapses on dead cells/segments or to dead presynaptic cells
    if (self.cells[segment.cell].destroyed or
        self.cells[presynapticCell].destroyed or
        segment._destroyed):
      return

    return (super(FaultyConnections, self)
            .createSynapse(segment, presynapticCell, permanence))



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
    self.zombiePermutation = None    # Contains the order in which cells
                                      # will be killed
    self.numDead = 0


  @staticmethod
  def connectionsFactory(*args, **kwargs):
    """
    Override connectionsFactory() in base class to return a Connections subclass
    that supports cell destruction.

    See Connections for constructor signature and usage

    @return: FaultyConnections instance
    """
    return FaultyConnections(*args, **kwargs)


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
      numSegmentDeleted += self.connections.destroyCell(cellIdx)

    print "Total number of segments removed=", numSegmentDeleted


  def activatePredictedColumn(self, *args, **kwargs):
    # Return only non-destroyed cells
    return [idx
            for idx
            in TemporalMemory.activatePredictedColumn(*args, **kwargs)
            if not self.connections.cells[idx].destroyed]


  def activateDendrites(self, learn=True):
    """ Originally copied from temporal_memory.py and augmented to suppress
    segments from dead cells.  See TemporalMemory.activateDendrites() for
    original implementation.
    """
    (numActiveConnected,
     numActivePotential) = self.connections.computeActivity(
      self.activeCells,
      self.connectedPermanence)

    # Suppress segments from dead cells
    activeSegments = (
      self.connections.segmentForFlatIdx(i)
      for i in xrange(len(numActiveConnected))
      if numActiveConnected[i] >= self.activationThreshold and
         not self.connections.cells[
           self.connections.segmentForFlatIdx(i).cell].destroyed
    )

    # Suppress segments from dead cells
    matchingSegments = (
      self.connections.segmentForFlatIdx(i)
      for i in xrange(len(numActivePotential))
      if numActivePotential[i] >= self.minThreshold and
         not self.connections.cells[self.connections.segmentForFlatIdx(i).cell].destroyed
    )

    maxSegmentsPerCell = self.connections.maxSegmentsPerCell
    segmentKey = lambda segment: (segment.cell * maxSegmentsPerCell
                                  + segment.idx)
    self.activeSegments = sorted(activeSegments, key=segmentKey)
    self.matchingSegments = sorted(matchingSegments, key=segmentKey)
    self.numActiveConnectedSynapsesForSegment = numActiveConnected
    self.numActivePotentialSynapsesForSegment = numActivePotential

    if learn:
      for segment in self.activeSegments:
        self.connections.recordSegmentActivity(segment)
      self.connections.startNewIteration()


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
             if not self.connections.cells[cellIdx].destroyed]

    if columnMatchingSegments is not None:
      numActive = lambda s: numActivePotentialSynapsesForSegment[s.flatIdx]

      bestMatchingSegment = None
      for seg in columnMatchingSegments:
        if bestMatchingSegment is None:
          bestMatchingSegment = seg
        elif seg > bestMatchingSegment and not connections.cells[seg.cell].destroyed:
          bestMatchingSegment = seg

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


  @classmethod
  def growSynapses(cls, connections, random, segment, nDesiredNewSynapes,
                   prevWinnerCells, initialPermanence):
    # Do not allow synapses to be grown on destroyed cells
    if connections.cells[segment.cell].destroyed:
      return

    # Ignore destroyed prevWinnerCells
    prevWinnerCells = (prevWinnerCell
                       for prevWinnerCell
                       in prevWinnerCells
                       if not connections.cells[prevWinnerCell].destroyed)

    TemporalMemory.growSynapses(connections, random, segment,
                                nDesiredNewSynapes, prevWinnerCells,
                                initialPermanence)


  @classmethod
  def leastUsedCell(cls, random, cells, connections):
    # Strip out destroyed cells for consideration
    cells = [cellIdx
             for cellIdx
             in cells
             if not connections.cells[cellIdx].destroyed]

    return TemporalMemory.leastUsedCell(random, cells, connections)


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