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


class NIK(object):
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

    self.trainingIterations = 0
    self.testIterations = 0
    self.maxPredictionError = 0
    self.totalPredictionError = 0
    self.numMissedPredictions = 0

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
    If learn is False, it will print a prediction: (theta1, theta2)
    """
    dx = xt - xt1
    dy = yt - yt1

    self.minDx = min(self.minDx, dx)
    self.maxDx = max(self.maxDx, dx)

    print >>sys.stderr, "Learn: ", learn
    print >>sys.stderr, "Training iterations: ", self.trainingIterations
    print >>sys.stderr, "Test iterations: ", self.testIterations
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
      self.trainingIterations += 1
    else:
      # During inference we provide the previous pose angle as bottom up input
      # If we don't get a prediction, we keep trying random shifts until we get
      # something.
      predictedCells = []
      newt1 = theta1t1
      newt2 = theta2t1
      newdx = dx
      newdy = dy
      angleRange = 10
      numAttempts = 1
      while len(predictedCells) == 0 and numAttempts < 3:
        print >> sys.stderr, "Attempt:", numAttempts,
        print >> sys.stderr, "Trying to predict using thetas:", newt1, newt2,
        print >> sys.stderr, "and deltas:", newdx, newdy

        externalSDR = self.encodeDeltas(newdx, newdy)
        bottomUpSDR = self.encodeThetas(newt1, newt2)
        predictedCells = self.inferTM(bottomUpSDR, externalSDR)
        predictedValues = self.decodeThetas(predictedCells)

        print >> sys.stderr, "Predicted values",predictedValues

        newt1 = theta1t1 + random.randrange(-angleRange,angleRange)
        newt2 = theta2t1 + random.randrange(-angleRange,angleRange)
        newdx = dx + (random.random()/2.0 - 0.25)
        newdy = dy + (random.random()/2.0 - 0.25)

        # Ensure we are in bounds otherwise we get an exception
        newt1 = min(self.maxTheta1, max(self.minTheta1, newt1))
        newt2 = min(self.maxTheta2, max(self.minTheta2, newt2))
        newdx = min(2.0, max(-2.0, newdx))
        newdy = min(2.0, max(-2.0, newdy))

        numAttempts += 1
        if numAttempts % 10 == 0: angleRange += 2

      print predictedValues
      # Accumulate errors for our metrics
      if len(predictedCells) == 0:
        self.numMissedPredictions += 1
      self.testIterations += 1
      error = abs(predictedValues[0] - theta1) + abs(predictedValues[1] - theta2)
      self.totalPredictionError += error
      if self.maxPredictionError < error:
        self.maxPredictionError = error
        print >> sys.stderr, "Error: ", error


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

    # print >> sys.stderr, "predicted cells = ", predictedCells
    # print >> sys.stderr, "decoded theta1 bits = ", theta1PredictedBits
    # print >> sys.stderr, "decoded theta2 bits = ", theta2PredictedBits
    # print >> sys.stderr, "decoded theta1 value = ", t1Prediction
    # print >> sys.stderr, "decoded theta2 value = ", t2Prediction

    return t1Prediction, t2Prediction


  def printStats(self):
    print >> sys.stderr, "min/max dx=",self.minDx, self.maxDx
    print >> sys.stderr, "Total number of segments=", numSegments(self.tm  )
    if self.testIterations > 0:
      print >> sys.stderr, "Maximum prediction error: ", self.maxPredictionError
      print >> sys.stderr, "Mean prediction error: ", self.totalPredictionError / self.testIterations
      print >> sys.stderr, "Num missed predictions: ", self.numMissedPredictions

  def trainTM(self, bottomUp, externalInput):
    # print >> sys.stderr, "Bottom up: ", bottomUp
    # print >> sys.stderr, "ExternalInput: ",externalInput
    self.tm.depolarizeCells(externalInput, learn=True)
    self.tm.activateCells(bottomUp,
           reinforceCandidatesExternalBasal=externalInput,
           growthCandidatesExternalBasal=externalInput,
           learn=True)
    # print >> sys.stderr, ("new active cells " + str(self.tm.getActiveCells()))
    print >> sys.stderr, "Total number of segments=", numSegments(self.tm  )


  def inferTM(self, bottomUp, externalInput):
    """
    Run inference and return the set of predicted cells
    """
    self.reset()
    # print >> sys.stderr, "Bottom up: ", bottomUp
    # print >> sys.stderr, "ExternalInput: ",externalInput
    self.tm.compute(bottomUp,
            activeCellsExternalBasal=externalInput,
            learn=False)
    # print >> sys.stderr, ("new active cells " + str(self.tm.getActiveCells()))
    # print >> sys.stderr, ("new predictive cells " + str(self.tm.getPredictiveCells()))
    return self.tm.getPredictiveCells()


  def save(self, filename="temp.pkl"):
    """
    Save TM in the filename specified above
    """
    output = open(filename, 'wb')
    cPickle.dump(self.tm, output, protocol=cPickle.HIGHEST_PROTOCOL)


  def load(self, filename="temp.pkl"):
    """
    Save TM in the filename specified above
    """
    inputFile = open(filename, 'rb')
    self.tm = cPickle.load(inputFile)


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
        nik.reset()

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
