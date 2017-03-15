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
import cPickle

import numpy
import random

from nupic.encoders.scalar import ScalarEncoder

from htmresearch.algorithms.extended_temporal_memory import ExtendedTemporalMemory as TM

#
# import matplotlib.pyplot as plt
#     ...:
#
# In [55]: griddelta, xedges, yedges = numpy.histogram2d(nik.dxValues[:, 0], nik.dxValues[:, 1],
#     ...:                                  bins=[griddx, griddy])
#
# In [56]:
#
# In [56]: plt.figure()
# Out[56]: <matplotlib.figure.Figure at 0x10fa1a5d0>
#
# In [57]: plt.plot(nik.dxValues[:, 0], nik.dxValues[:, 1], 'ro')
# Out[57]: [<matplotlib.lines.Line2D at 0x10fa2d0d0>]
#
# In [58]: plt.grid(True)
#
# In [59]: plt.figure()
#     ...:     myextent  =[xedges[0],xedges[-1],yedges[0],yedges[-1]]
#   File "<ipython-input-59-2af354dfe89d>", line 2
#     myextent  =[xedges[0],xedges[-1],yedges[0],yedges[-1]]
#     ^
# IndentationError: unexpected indent
#
#
# In [60]: plt.figure()
#     ...:
#     ...:
#     ...: myextent  =[xedges[0],xedges[-1],yedges[0],yedges[-1]]
#     ...:
#
# In [61]: plt.imshow(griddelta.T,origin='low',extent=myextent,interpolation='nearest',aspect='auto')
# Out[61]: <matplotlib.image.AxesImage at 0x111afe290>
#
# In [62]: plt.plot(nik.dxValues[:, 0], nik.dxValues[:, 1], 'ro')
# Out[62]: [<matplotlib.lines.Line2D at 0x111a974d0>]
#
# In [63]: plt.colorbar()
# Out[63]: <matplotlib.colorbar.Colorbar instance at 0x1128337e8>
#
# In [64]: plt.show()
# /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/matplotlib/collections.py:548: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
#   if self._edgecolors == 'face':


def numSegments(tm):
  return tm.basalConnections.numSegments()

def str2bool(v):
  return v[0].lower() in ("yes", "true", "t", "1")

def printSegmentForCell(tm, cell):
  """Print segment information for this cell"""
  print "Segments for cell", cell, ":"
  for seg in tm.basalConnections._cells[cell]._segments:
    print "    ",
    synapses = seg._synapses
    for s in synapses:
      print "%d:%g" %(s.presynapticCell,s.permanence),
    print


class NIKAnalysis(object):
  """Class implementing NIK"""

  def __init__(self,
               minDx=-2.0, maxDx=2.0,
               minDy=-2.0, maxDy=2.0,
               minTheta1=0.0, maxTheta1=85.0,
               minTheta2=0.0, maxTheta2=360.0,
               ):

    self.dxEncoder = ScalarEncoder(5, minDx, maxDx, n=75, forced=True)
    self.dyEncoder = ScalarEncoder(5, minDy, maxDy, n=75, forced=True)
    self.externalSize = self.dxEncoder.getWidth()**2
    self.externalOnBits = self.dxEncoder.w**2

    self.theta1Encoder = ScalarEncoder(5, minTheta1, maxTheta1, n=75, forced=True)
    self.theta2Encoder = ScalarEncoder(5, minTheta2, maxTheta2, n=75, forced=True)
    self.bottomUpInputSize = self.theta1Encoder.getWidth()*self.theta2Encoder.getWidth()
    self.bottomUpOnBits = self.theta1Encoder.w*self.theta2Encoder.w

    self.minDx = 100.0
    self.maxDx = -100.0
    self.minTheta1 = minTheta1
    self.minTheta2 = minTheta2
    self.maxTheta1 = maxTheta1
    self.maxTheta2 = maxTheta2

    self.numPoints = 0

    self.maxPredictionError = 0
    self.totalPredictionError = 0
    self.numMissedPredictions = 0

    self.dxValues = numpy.zeros((45000,2))
    self.thetaValues = numpy.zeros((45000,2))



  def compute(self, xt1, yt1, xt, yt, theta1t1, theta2t1, theta1, theta2, learn):
    """
    The main function to call.
    If learn is False, it will print a prediction: (theta1, theta2)
    """
    dx = xt - xt1
    dy = yt - yt1

    self.minDx = min(self.minDx, dx)
    self.maxDx = max(self.maxDx, dx)

    self.dxValues[self.numPoints,0] = dx
    self.dxValues[self.numPoints,1] = dy

    self.thetaValues[self.numPoints,0] = theta1
    self.thetaValues[self.numPoints,1] = theta2

    self.numPoints += 1

    print >>sys.stderr, "Xt's: ", xt1, yt1, xt, yt, "Delta's: ", dx, dy
    print >>sys.stderr, "Theta t-1: ", theta1t1, theta2t1, "t:",theta1, theta2


  def reset(self):
    pass


  def printStats(self):
    print >> sys.stderr, "min/max dx=",self.minDx, self.maxDx
    print >> sys.stderr, "min/max dx=",self.dxValues[:,0].min(), self.dxValues[:,0].max()
    print >> sys.stderr, "min/max dy=",self.dxValues[:,1].min(), self.dxValues[:,1].max()
    print >> sys.stderr, "mean dx=",self.dxValues[:,0].mean(), self.dxValues[:,1].mean()
    print >> sys.stderr, "mean theta=",self.thetaValues[:,0].mean(), self.thetaValues[:,1].mean()
    print "Number of points=",self.numPoints


  def computeHistograms(self):
    gridt1 = numpy.linspace(0, 85, 20)
    gridt2 = numpy.linspace(0, 360, 20)
    grid, _, _ = numpy.histogram2d(self.thetaValues[:, 0], self.thetaValues[:, 1],
                                 bins=[gridt1, gridt2])

    griddx = numpy.linspace(-0.5, 0.5, 20)
    griddy = numpy.linspace(-0.5, 0.5, 20)
    griddelta, _, _ = numpy.histogram2d(self.dxValues[:, 0], self.dxValues[:, 1],
                                 bins=[griddx, griddy])

    print grid


  def save(self, filename="temp.pkl"):
    """
    Save TM in the filename specified above
    """
    pass


  def load(self, filename="temp.pkl"):
    """
    Save TM in the filename specified above
    """
    pass


if __name__ == "__main__":
  usage = """
  This program shows how to do HTM mapping. It reads in 9 inputs from stdin:

  x(t-1), y(t-1), x(t), y(t), theta1(t-1), theta2(t-1), theta1(t), theta2(t), training

  learn is either True or False

  Example:
    -3.998477,0.1047044,-3.996574,0.1570263,86,85,86,85,true

  """

  nik = NIKAnalysis()
  line = 0

  while True:
    try:
      xs = raw_input()
      xs = xs.strip()
      line += 1
      x = xs.split(",")
      if x[0] == "load":
        filename = x[1]
        print >> sys.stderr, "Loading from file:",filename
        nik.load(filename)
        print >> sys.stderr, "Done loading file:",filename

      elif x[0] == "save":
        filename = x[1]
        print >> sys.stderr, "Saving to filename", filename
        nik.save(filename)

      elif len(x) != 9 or xs[0] == "x":
        print >> sys.stderr, "Resetting at line",line
        print >> sys.stderr

      else:
        nik.compute(xt1=float(x[0]), yt1=float(x[1]),
                    xt=float(x[2]), yt=float(x[3]),
                    theta1t1=float(x[4]), theta2t1=float(x[5]),
                    theta1=float(x[6]), theta2=float(x[7]),
                    learn=str2bool(x[8]))
      sys.stdout.flush()
    except EOFError:
      print >>sys.stderr, "Quitting!!!"
      break
    except Exception as e:
      print >>sys.stderr, "Error in line",line,"!!!!"
      print >>sys.stderr, xs
      print >>sys.stderr, str(e)
      break

  nik.printStats()

  # Save nik analysis
  output = open("nik_analysis.pkl", 'wb')
  cPickle.dump(nik, output, protocol=cPickle.HIGHEST_PROTOCOL)
  output.close()

  # Load nik analysis
  inputFile = open("nik_analysis.pkl", 'rb')
  nik = cPickle.load(inputFile)
