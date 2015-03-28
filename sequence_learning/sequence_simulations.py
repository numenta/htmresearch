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

from nupic.research.temporal_memory import TemporalMemory
from sensorimotor.orphan_temporal_memory import OrphanTemporalMemory

def returnLetterVectors(w=40):
  """
  Return a dictionary of 26 unique vectors indexed by a letter. Each vector
  has w contiguous bits ON and represented as a sequence of non-zero indices.
  """
  letters = {}
  for i,s in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    letters[s] = range(i*w,(i+1)*w)
  return letters


def getRandomVector(w=40, n=2048):
  "Return a list of w random indices out of a vector of n elements"
  return numpy.random.permutation(n)[0:w]


def computeError(tm):
  """
  Given a temporal memory instance compute the error metric.
  :param tm:
  :return:
  """


if __name__ == '__main__':
  print "Running"
  numpy.random.seed(42)

  returnLetterVectors()

  tm = OrphanTemporalMemory(minThreshold=30, activationThreshold=30,
                      maxNewSynapseCount=40, cellsPerColumn=5,
                      learnOnOneCell = False,
                      permanenceOrphanDecrement = 0.005)

  inputVecs = []
  for i in range(4):
    v = set(getRandomVector())
    inputVecs.append(v)
    print "input i=",v
    if 1977 in v: print "1977 in there"

  print "=================="

  col1977 = 0
  i=0
  while i < 2000:
    if i%100==0:
      print "i=",i
    if i%8 <= 3:
      inputVec = inputVecs[i%4]
    else:
      inputVec = set(getRandomVector())

    # if 1977 in inputVec:
    #   col1977 += 1
    #   print "Yes:",col1977
    #
    tm.compute(inputVec, learn=True)
    if i >= 1:
      print i,len(tm.predictiveCells)
      # print "input=",inputVec
      # print "predicted cells=",
      # pprint.pprint(tm.predictiveCells)
      # print "columns=",tm.mapCellsToColumns(tm.predictiveCells)
      # print
    i += 1
