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

from nupic.encoders.scalar import ScalarEncoder

from htmresearch.algorithms.extended_temporal_memory import ExtendedTemporalMemory as TM


def numSegments(tm):
  return tm.basalConnections.numSegments()

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

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

  def __init__(self,
               minDx=-1.0, maxDx=1.0,
               minDy=-1.0, maxDy=1.0,
               minTheta1=0.0, maxTheta1=70.0,
               minTheta2=0.0, maxTheta2=350.0,
               ):

    self.dxEncoder = ScalarEncoder(5, minDx, maxDx, n=100, forced=True)
    self.dyEncoder = ScalarEncoder(5, minDy, maxDy, n=100, forced=True)
    self.externalSize = self.dxEncoder.getWidth()**2
    self.externalOnBits = self.dxEncoder.w**2

    self.theta1Encoder = ScalarEncoder(5, minTheta1, maxTheta1, n=100, forced=True)
    self.theta2Encoder = ScalarEncoder(5, minTheta2, maxTheta2, n=100, forced=True)
    self.bottomUpInputSize = self.theta1Encoder.getWidth()**2
    self.bottomUpOnBits = self.theta1Encoder.w**2

    self.minDx = 100.0
    self.maxDx = -100.0

    self.tm = TM(columnDimensions = (self.bottomUpInputSize,),
            basalInputDimensions = (self.externalSize,),
            cellsPerColumn=1,
            initialPermanence=0.4,
            connectedPermanence=0.5,
            minThreshold= self.externalOnBits,
            maxNewSynapseCount=40,
            permanenceIncrement=0.1,
            permanenceDecrement=0.00,
            activationThreshold=int(0.75*(self.externalOnBits+self.bottomUpOnBits)),
            predictedSegmentDecrement=0.00,
            checkInputs=False
            )

    print >>sys.stderr, "TM parameters:"
    print >>sys.stderr, "  num columns=",self.tm.getColumnDimensions()
    print >>sys.stderr, "  activation threshold=",self.tm.getActivationThreshold()
    print >>sys.stderr, "  min threshold=",self.tm.getMinThreshold()
    print >>sys.stderr, "  basal input dimensions=",self.tm.getBasalInputDimensions()
    print >>sys.stderr
    print >>sys.stderr



  def compute(self, xt1, yt1, xt, yt, theta1t1, theta2t1, theta1, theta2, learn):
    """
    The main function to call.
    Return a prediction for theta1 and theta2
    """
    dx = xt - xt1
    dy = yt - yt1

    self.minDx = min(self.minDx, dx)
    self.maxDx = max(self.maxDx, dx)

    print >>sys.stderr, "Learn: ", learn
    print >>sys.stderr, "Xt's: ", xt1, yt1, xt, yt, "Delta's: ", dx, dy
    print >>sys.stderr, "Theta t-1: ", theta1t1, theta2t1, "t:",theta1, theta2

    bottomUpSDR = self.encodeThetas(theta1, theta2)
    self.decodeThetas(bottomUpSDR)

    # Encode the inputs appropriately and train the HTM
    externalSDR = self.encodeDeltas(dx,dy)
    if learn:
      # During learning we provide the current pose angle as bottom up input
      bottomUpSDR = self.encodeThetas(theta1, theta2)
      self.trainTM(bottomUpSDR, externalSDR)
    else:
      # During inference we provide the previous pose angle as bottom up input
      bottomUpSDR = self.encodeThetas(theta1t1, theta2t1)
      predictedCells = self.inferTM(bottomUpSDR, externalSDR)
      print self.decodeThetas(predictedCells)

    print >> sys.stderr


  def reset(self):
    self.tm.reset()


  def encodeDeltas(self, dx,dy):
    """Return the SDR for dx,dy"""
    dxe = self.dxEncoder.encode(dx)
    dye = self.dyEncoder.encode(dy)
    ex = numpy.outer(dxe,dye)
    return ex.flatten().nonzero()[0]


  def encodeThetas(self, theta1, theta2):
    """Return the SDR for theta1 and theta2"""
    # print >> sys.stderr, "encoded theta1 value = ", theta1
    # print >> sys.stderr, "encoded theta2 value = ", theta2
    t1e = self.theta1Encoder.encode(theta1)
    t2e = self.theta2Encoder.encode(theta2)
    # print >> sys.stderr, "encoded theta1 = ", t1e.nonzero()[0]
    # print >> sys.stderr, "encoded theta2 = ", t2e.nonzero()[0]
    ex = numpy.outer(t2e,t1e)
    return ex.flatten().nonzero()[0]


  def decodeThetas(self, predictedCells):
    """
    Given the set of predicted cells, return the predicted theta1 and theta2
    """
    a = numpy.zeros(self.bottomUpInputSize)
    a[predictedCells] = 1
    a = a.reshape((self.theta1Encoder.getWidth(), self.theta1Encoder.getWidth()))
    theta1PredictedBits = a.mean(axis=0).nonzero()[0]
    theta2PredictedBits = a.mean(axis=1).nonzero()[0]

    # To decode it we need to create a flattened array again and pass it
    # to encoder.
    # TODO: We use encoder's topDownCompute method - not sure if that is best.
    t1 = numpy.zeros(self.theta1Encoder.getWidth())
    t1[theta1PredictedBits] = 1
    t1Prediction = self.theta1Encoder.topDownCompute(t1)[0].value

    t2 = numpy.zeros(self.theta2Encoder.getWidth())
    t2[theta2PredictedBits] = 1
    t2Prediction = self.theta2Encoder.topDownCompute(t2)[0].value

    # print >> sys.stderr, "decoded theta1 bits = ", theta1PredictedBits
    # print >> sys.stderr, "decoded theta2 bits = ", theta2PredictedBits
    # print >> sys.stderr, "decoded theta1 value = ", t1Prediction
    # print >> sys.stderr, "decoded theta2 value = ", t2Prediction

    return t1Prediction, t2Prediction


  def printStats(self):
    print >> sys.stderr, "min/max dx=",self.minDx, self.maxDx
    print >> sys.stderr, "Total number of segments=", numSegments(self.tm  )


  def trainTM(self, bottomUp, externalInput):
    print >> sys.stderr, "Bottom up: ", bottomUp
    print >> sys.stderr, "ExternalInput: ",externalInput
    self.tm.depolarizeCells(externalInput, learn=True)
    self.tm.activateCells(bottomUp,
           reinforceCandidatesExternalBasal=externalInput,
           growthCandidatesExternalBasal=externalInput,
           learn=True)
    print >> sys.stderr, ("new active cells " + str(self.tm.getActiveCells()))
    print >> sys.stderr, "Total number of segments=", numSegments(self.tm  )


  def inferTM(self, bottomUp, externalInput):
    """
    Run inference and return the set of predicted cells
    """
    self.reset()
    print >> sys.stderr, "Bottom up: ", bottomUp
    print >> sys.stderr, "ExternalInput: ",externalInput
    self.tm.compute(bottomUp,
            activeCellsExternalBasal=externalInput,
            learn=False)
    print >> sys.stderr, ("new active cells " + str(self.tm.getActiveCells()))
    print >> sys.stderr, ("new predictive cells " + str(self.tm.getPredictiveCells()))
    return self.tm.getPredictiveCells()


if __name__ == "__main__":
  usage = """
  This program shows how to do HTM mapping. It reads in 9 inputs from stdin:

  x(t-1), y(t-1), x(t), y(t), theta1(t-1), theta2(t-1), theta1(t), theta2(t), training

  learn is either True or False

  Example:
    -3.998477,0.1047044,-3.996574,0.1570263,86,85,86,85,true

  """

  nik = NIK()
  line = 0

  while True:
    try:
      xs = raw_input()
      line += 1
      x = xs.split(",")
      if len(x) != 9 or xs[0] == "x":
        print >> sys.stderr, "Resetting at line",line
        print >> sys.stderr
        nik.reset()
      else:
        nik.compute(xt1=float(x[0]), yt1=float(x[1]),
                    xt=float(x[2]), yt=float(x[3]),
                    theta1t1=float(x[4]), theta2t1=float(x[5]),
                    theta1=float(x[6]), theta2=float(x[7]),
                    learn=str2bool(x[8]))
    except EOFError:
      print >>sys.stderr, "Quitting!!!"
      break
    except Exception as e:
      print >>sys.stderr, "Error in line",line,"!!!!"
      print >>sys.stderr, xs
      print >>sys.stderr, str(e)
      break

  nik.printStats()

