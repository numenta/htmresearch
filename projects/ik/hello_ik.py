# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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


print """
This program shows how to access the Temporal Memory directly by demonstrating
how to create a TM instance, train it with vectors, get predictions, and
inspect the state.

The code here runs a very simple version of sequence learning, with one
cell per column. The TP is trained with the simple sequence A->B->C->D->E

HOMEWORK: once you have understood exactly what is going on here, try changing
cellsPerColumn to 4. What is the difference between once cell per column and 4
cells per column?

PLEASE READ THROUGH THE CODE COMMENTS - THEY EXPLAIN THE OUTPUT IN DETAIL

"""

# Can't live without numpy
import numpy

# izip for maximum efficiency
from itertools import izip as zip, count

# Python implementation of Temporal Memory

from htmresearch.algorithms.extended_temporal_memory import ExtendedTemporalMemory as TM

# Utility routine for printing the input vector
def formatRow(x):
  s = ''
  for c in range(len(x)):
    if c > 0 and c % 10 == 0:
      s += ' '
    s += str(x[c])
  s += ' '
  return s

def numSegments(tm):
  return tm.basalConnections.numSegments()

def printSegmentForCell(tm, cell):
  for seg in tm.basalConnections._cells[cell]._segments:
    synapses = seg._synapses
    print "Segment for cell",cell,":",
    for s in synapses:
      print s.presynapticCell,
    print


# Step 1: create Temporal Pooler instance with appropriate parameters

tm = TM(columnDimensions = (50,),
        basalInputDimensions = (30,),
        cellsPerColumn=1,
        initialPermanence=0.5,
        connectedPermanence=0.5,
        minThreshold=10,
        maxNewSynapseCount=20,
        permanenceIncrement=0.1,
        permanenceDecrement=0.0,
        activationThreshold=15,
        )


# Mappings we want to learn:
#
# (E0, X0) -> X1
# (E0, X1) -> X0
# (E0, X2) -> X0
# (E1, X0) -> X2
# (E1, X1) -> X2
# (E1, X2) -> X1

# Indices corresponding to above mapping
mappingIndices = [
  (0,0,1), (0,1,0), (0,2,0), (1,0,2), (1,1,2), (1,2,1)
]

# Step 2: create input vectors to feed to the TM. Each input vector
# must be numberOfCols wide.
x = numpy.zeros((3, tm.numberOfColumns()), dtype="uint32")
x[0, 0:10] = 1    # Input SDR representing "X1", corresponding to columns 0-9
x[1, 10:20] = 1   # Input SDR representing "X2", corresponding to columns 10-19
x[2, 20:30] = 1   # Input SDR representing "X3", corresponding to columns 20-29


# Step 2.1: create external input vectors to feed to the TM.
ex = numpy.zeros((3, 30), dtype="uint32")
ex[0, 0:10] = 1    # Input SDR representing "E1"
ex[1, 10:20] = 1   # Input SDR representing "E2"
ex[2, 20:30] = 1   # Input SDR representing "E3"


# Step 3: send this simple sequence to the temporal memory for learning
# We repeat the sequence 10 times
for i in range(1):

  # First step
  print "------------------------"
  tm.depolarizeCells(ex[0].nonzero()[0])
  tm.activateCells(
    x[1].nonzero()[0],
    reinforceCandidatesExternalBasal=ex[0].nonzero()[0],
    growthCandidatesExternalBasal=ex[0].nonzero()[0])
  print("active cells " + str(tm.getActiveCells()))
  print("predictive cells " + str(tm.getPredictiveCells()))
  # printSegmentForCell(tm,0)
  # printSegmentForCell(tm,10)

  print "------------------------"
  tm.depolarizeCells(ex[0].nonzero()[0])
  tm.activateCells(
    x[0].nonzero()[0],
    reinforceCandidatesExternalBasal=ex[0].nonzero()[0],
    growthCandidatesExternalBasal=ex[0].nonzero()[0])
  print("active cells " + str(tm.getActiveCells()))
  # printSegmentForCell(tm,0)
  # printSegmentForCell(tm,10)

  print "------------------------"
  tm.depolarizeCells(ex[0].nonzero()[0])
  tm.activateCells(
    x[1].nonzero()[0],
    reinforceCandidatesExternalBasal=ex[0].nonzero()[0],
    growthCandidatesExternalBasal=ex[0].nonzero()[0])

  print("active cells " + str(tm.getActiveCells()))
  # printSegmentForCell(tm,0)
  # printSegmentForCell(tm,10)

  print "------------------------"
  tm.depolarizeCells(ex[0].nonzero()[0])
  tm.activateCells(
    x[0].nonzero()[0],
    reinforceCandidatesExternalBasal=ex[0].nonzero()[0],
    growthCandidatesExternalBasal=ex[0].nonzero()[0])
  print("active cells " + str(tm.getActiveCells()))
  print("predictive cells " + str(tm.getPredictiveCells()))
  # printSegmentForCell(tm,0)
  # printSegmentForCell(tm,10)

  print "------------------------"
  tm.depolarizeCells(ex[0].nonzero()[0])
  tm.activateCells(
    x[1].nonzero()[0],
    reinforceCandidatesExternalBasal=ex[0].nonzero()[0],
    growthCandidatesExternalBasal=ex[0].nonzero()[0])
  print("active cells " + str(tm.getActiveCells()))
  print("predictive cells " + str(tm.getPredictiveCells()))
  # printSegmentForCell(tm,0)
  # printSegmentForCell(tm,10)

  print "------------------------"
  tm.depolarizeCells(ex[0].nonzero()[0])
  tm.activateCells(
    x[0].nonzero()[0],
    reinforceCandidatesExternalBasal=ex[0].nonzero()[0],
    growthCandidatesExternalBasal=ex[0].nonzero()[0])
  print("active cells " + str(tm.getActiveCells()))
  print("predictive cells " + str(tm.getPredictiveCells()))
  # printSegmentForCell(tm,0)
  # printSegmentForCell(tm,10)


  # The reset command tells the TP that a sequence just ended and essentially
  # zeros out all the states. It is not strictly necessary but it's a bit
  # messier without resets, and the TP learns quicker with resets.
  tm.reset()


# #######################################################################
# #
# # Step 3: send the same sequence of vectors and look at predictions made by
# # temporal memory
# for j in range(3):
#   print "\n\n--------","ABCDE"[j],"-----------"
#   print "Raw input vector : " + formatRow(x[j])
#
#   # Send each vector to the TM, with learning turned off
#   tm.compute(x[j].nonzero()[0], learn = False)
#
#   # The following print statements prints out the active cells, predictive
#   # cells, active segments and winner cells.
#   #
#   # What you should notice is that the columns where active state is 1
#   # represent the SDR for the current input pattern and the columns where
#   # predicted state is 1 represent the SDR for the next expected pattern
#   print "\nAll the active and predicted cells:"
#
#   print("active cells " + str(tm.getActiveCells()))
#   print("predictive cells " + str(tm.getPredictiveCells()))
#   print("winner cells " + str(tm.getWinnerCells()))
#   print("# of active segments " + str(tm.basalConnections.numSegments()))
#
#   activeColumnsIndeces = [tm.columnForCell(i) for i in tm.getActiveCells()]
#   predictedColumnIndeces = [tm.columnForCell(i) for i in tm.getPredictiveCells()]
#
#
#   # Reconstructing the active and inactive columns with 1 as active and 0 as
#   # inactive representation.
#
#   actColState = ['1' if i in activeColumnsIndeces else '0' for i in range(tm.numberOfColumns())]
#   actColStr = ("".join(actColState))
#   predColState = ['1' if i in predictedColumnIndeces else '0' for i in range(tm.numberOfColumns())]
#   predColStr = ("".join(predColState))
#
#   # For convenience the cells are grouped
#   # 10 at a time. When there are multiple cells per column the printout
#   # is arranged so the cells in a column are stacked together
#   print "Active columns:    " + formatRow(actColStr)
#   print "Predicted columns: " + formatRow(predColStr)
#
#   # predictedCells[c][i] represents the state of the i'th cell in the c'th
#   # column. To see if a column is predicted, we can simply take the OR
#   # across all the cells in that column. In numpy we can do this by taking
#   # the max along axis 1.
