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

import sys

import numpy

from htmresearch.algorithms.extended_temporal_memory import ExtendedTemporalMemory as TM


def numSegments(tm):
  return tm.basalConnections.numSegments()


def printSegmentForCell(tm, cell):
  """Print segment information for this cell"""
  print "Segments for cell", cell, ":"
  for seg in tm.basalConnections._cells[cell]._segments:
    print "    ",
    synapses = seg._synapses
    for s in synapses:
      print "%d:%g" %(s.presynapticCell,s.permanence),
    print


class NIK(object):
  """Class implementing NIK"""

  def __init__(self):
    self.tm = TM(columnDimensions = (2048,),
            basalInputDimensions = (30,),
            cellsPerColumn=1,
            initialPermanence=0.4,
            connectedPermanence=0.5,
            minThreshold=10,
            maxNewSynapseCount=20,
            permanenceIncrement=0.1,
            permanenceDecrement=0.0,
            activationThreshold=15,
            predictedSegmentDecrement=0.01
            )


  def encodeDeltas(self, dx,dy):
    """Return the SDR for dx,dy"""
    pass

  def encodeThetas(self, theta1, theta2):
    """Return the SDR for theta1 and theta2"""
    pass

  def trainTM(self, dx, dy, xt, yt, theta1, theta2):
    pass

  def compute(self, dx, dy, xt, yt, theta1, theta2, learn):
    print >>sys.stderr, dx, dy, xt, yt, theta1, theta2, learn


if __name__ == "__main__":
  print """
  This program shows how to do HTM mapping. It reads in 7 numbers from stdin:
    dx dy x_t y_t theta1_t theta2_t learn

  learn is either True or False
  dx should be x_t - x_(t-1)
  dy should be y_t - y_(t-1)

  Example:
    -1.2 2.1 10 20 0.3 0.4 True

  """

  nik = NIK()

  x = ""
  while x != "quit":
    xs = raw_input()
    x = xs.split()
    nik.compute(dx=float(x[0]), dy=float(x[1]),
                xt=float(x[2]), yt=float(x[3]),
                theta1=float(x[4]), theta2=float(x[5]),
                learn=bool(x[6]))
