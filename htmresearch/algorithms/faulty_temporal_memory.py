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
from nupic.bindings.algorithms import TemporalMemory



class FaultyTemporalMemory(TemporalMemory):
  """
  Class implementing a fallible Temporal Memory class. This class allows the
  user to kill a certain number of cells. The dead cells cannot become active,
  will no longer participate in predictions, and cannot become winning cells.
  This feature enables us to test the robustness of the Temporal Memory
  algorithm under such situations.
  """

  def __init__(self, **kwargs):
    super(FaultyTemporalMemory, self).__init__(**kwargs)
    self.deadCells = set()
    self.zombiePermutation = None # Contains the order in which cells
                                  # will be killed
    self.numDead = 0


  def killCells(self, percent=0.05):
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

    print "Total number of dead cells=", len(self.deadCells)

    numSegmentDeleted = 0
    for cell in self.deadCells:
      segmentsPerCell = list(self.connections.segmentsForCell(cell))
      for segment in segmentsPerCell:
        self.connections.destroySegment(segment)
        numSegmentDeleted += 1

    print "Total number of segments removed=", numSegmentDeleted


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


