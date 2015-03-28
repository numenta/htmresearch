# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import pprint
import numpy

from sensorimotor.orphan_temporal_memory import OrphanTemporalMemory
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin)

class MonitoredTemporalMemory(TemporalMemoryMonitorMixin, OrphanTemporalMemory): pass


def lv(letter, w=40):
  """
  Return an input vector corresponding to a letter. Each vector has w contiguous
  bits ON and represented as a sequence of non-zero indices.
  """
  i = ord(letter) - ord('A')
  return set(range(i*w,(i+1)*w))

def getRandomVector(w=40, n=2048):
  "Return a list of w random indices out of a vector of n elements"
  return numpy.random.permutation(n)[0:w]


def getNextSequenceChunk(it, w=40, n=2048):
  """
  Given an iteration index, returns a list of vectors to be appended to the
  input stream.
  """
  if it%10==3:
    vecs = [lv('A',w), lv('B',w), lv('C',w), lv('D',w), lv('E',w), lv('F',w)]
  else:
    vecs= [set(getRandomVector(w, n))]

  return vecs



def computePredictionAccuracy(pac, pic):
  """
  Given a temporal memory instance return the prediction accuracy. The accuracy
  is computed as 1 - (#correctly predicted cols / # predicted cols). The
  accuracy is 0 if there were no predicted columns.
  """
  pcols = float(pac + pic)
  if pcols == 0:
    return 0.0
  else:
    return (pac / pcols)

def testEverything():
  """
  Temporary - for debugging stuff.
  """
  print "Running"
  numpy.random.seed(42)
  tm = MonitoredTemporalMemory(minThreshold=30, activationThreshold=30,
                              maxNewSynapseCount=40, cellsPerColumn=5,
                              learnOnOneCell = False,
                              permanenceOrphanDecrement = 0.005)

  inputVecs = []
  for i in range(4):
    v = set(getRandomVector())
    inputVecs.append(v)
    print "input i=",v

  print "=================="

  i=0
  while i < 1000:
    if i%100==0:
      print "i=",i
    if i%8 <= 3:
      inputVec = inputVecs[i%4]
    else:
      inputVec = set(getRandomVector())

    tm.compute(inputVec, learn=True)
    if i >= 1:
      print i,len(tm.predictiveCells)
    i += 1

  # doesn't work?
  #tm.mmGetCellActivityPlot()

  print tm.mmPrettyPrintSequenceCellRepresentations()

  pac = tm.mmGetTracePredictedActiveColumns()
  pic = tm.mmGetTracePredictedInactiveColumns()
  upac = tm.mmGetTraceUnpredictedActiveColumns()

  print len(pac.data),len(pic.data)
  print "i pac pic upac err"
  for i,j in enumerate(pac.data):
    print i,len(j),len(pic.data[i]),len(upac.data[i]),
    print computePredictionAccuracy(len(j), len(pic.data[i]))


def runExperiment1():
  print "Running"
  numpy.random.seed(42)

  tm = MonitoredTemporalMemory(minThreshold=30, activationThreshold=30,
                              maxNewSynapseCount=40, cellsPerColumn=5,
                              learnOnOneCell = False,
                              permanenceOrphanDecrement = 0.005)

  i=0
  while i < 100:
    if i%100==0:
      print "i=",i

    vecs = getNextSequenceChunk(i)
    for vec in vecs:
      tm.compute(vec, learn=True)

      if i >= 1:
        print i,len(tm.predictiveCells)
      i += 1

  # Print out trace of predictions and accuracy
  pac = tm.mmGetTracePredictedActiveColumns()
  pic = tm.mmGetTracePredictedInactiveColumns()
  upac = tm.mmGetTraceUnpredictedActiveColumns()

  print len(pac.data),len(pic.data)
  print "i pac pic upac err"
  for i,j in enumerate(pac.data):
    print i,len(j),len(pic.data[i]),len(upac.data[i]),
    print computePredictionAccuracy(len(j), len(pic.data[i]))

if __name__ == '__main__':
  #testEverything()
  runExperiment1()
