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
import matplotlib.pyplot as plt

from nupic.encoders.scalar import ScalarEncoder

class NIKAnalysis(object):
  """Class implementing NIKAnalysis"""

  def __init__(self):

    self.maxPoints = 45000
    self.numPoints = 0
    self.dxValues = numpy.zeros((self.maxPoints,2))
    self.thetaValues = numpy.zeros((self.maxPoints,2))



  def compute(self, xt1, yt1, xt, yt, theta1t1, theta2t1, theta1, theta2):
    """
    Accumulate the various inputs.
    """
    dx = xt - xt1
    dy = yt - yt1

    if self.numPoints < self.maxPoints:
      self.dxValues[self.numPoints,0] = dx
      self.dxValues[self.numPoints,1] = dy

      self.thetaValues[self.numPoints,0] = theta1
      self.thetaValues[self.numPoints,1] = theta2

      self.numPoints += 1

      # print >>sys.stderr, "Xt's: ", xt1, yt1, xt, yt, "Delta's: ", dx, dy
      # print >>sys.stderr, "Theta t-1: ", theta1t1, theta2t1, "t:",theta1, theta2

    elif self.numPoints == self.maxPoints:
      print >> sys.stderr,"Max points exceeded, analyzing ",self.maxPoints,"points only"
      self.numPoints += 1


  def printStats(self):
    print "Number of points=",self.numPoints
    print >> sys.stderr, "min/max dx=",self.dxValues[:,0].min(), self.dxValues[:,0].max()
    print >> sys.stderr, "min/max dy=",self.dxValues[:,1].min(), self.dxValues[:,1].max()
    print >> sys.stderr, "mean dx=",self.dxValues[:,0].mean(), self.dxValues[:,1].mean()
    print >> sys.stderr, "mean theta=",self.thetaValues[:,0].mean(), self.thetaValues[:,1].mean()


  def computeHistograms(self):

    # Plot the theta values and their histogram
    gridt1 = numpy.linspace(0, 85, 20)
    gridt2 = numpy.linspace(0, 360, 20)
    grid, xedges, yedges = numpy.histogram2d(self.thetaValues[:, 0], self.thetaValues[:, 1],
                                 bins=[gridt1, gridt2])

    plt.figure()
    myextent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(grid.T, origin='low', extent=myextent,
               interpolation='nearest', aspect='auto')
    plt.plot(self.thetaValues[:, 0], self.thetaValues[:, 1], 'ro')
    plt.colorbar()
    plt.savefig("theta_histogram.png")

    # Plot the delta values and their histogram
    griddx = numpy.linspace(-0.6, 0.6, 20)
    griddy = numpy.linspace(-0.6, 0.6, 20)
    griddelta, xedges, yedges = numpy.histogram2d(self.dxValues[:, 0], self.dxValues[:, 1],
                                 bins=[griddx, griddy])

    plt.figure()
    myextent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(griddelta.T, origin='low', extent=myextent,
               interpolation='nearest', aspect='auto')
    plt.plot(self.dxValues[:, 0], self.dxValues[:, 1], 'ro')
    plt.colorbar()
    plt.savefig("dxhistograms.png")


if __name__ == "__main__":
  usage = """
  This program analyzes the training file and computes the distribution of
  delta_x values.

  Example invocation:
    %> cat train50k_seed37_2.csv | python nik_analysis.py

  """

  nik = NIKAnalysis()
  line = 0
  xs = ""

  while True:
    try:
      xs = raw_input()
      xs = xs.strip()
      line += 1
      x = xs.split(",")
      if not (x[0] in ["load", "save"] or len(x) != 9 or xs[0] == "x"):
        nik.compute(xt1=float(x[0]), yt1=float(x[1]),
                    xt=float(x[2]), yt=float(x[3]),
                    theta1t1=float(x[4]), theta2t1=float(x[5]),
                    theta1=float(x[6]), theta2=float(x[7]))
      sys.stdout.flush()
    except EOFError:
      break
    except Exception as e:
      print >>sys.stderr, "Error in line",line,"!!!!"
      print >>sys.stderr, xs
      print >>sys.stderr, str(e)
      break

  nik.printStats()
  nik.computeHistograms()
