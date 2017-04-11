#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
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

"""TODO"""

import random

import numpy

from sensorimotor import stp



if __name__ == "__main__":
  r = random.Random(42)
  w = 20
  nInputs = 500
  A, B, C, D, E = [numpy.zeros([nInputs], dtype=numpy.bool) for _ in xrange(5)]
  A[:w] = 1
  B[w:2*w] = 1
  C[2*w:3*w] = 1
  D[3*w:4*w] = 1
  E[4*w:5*w] = 1

  coincThreshold = 8
  nCols = 100
  nActive = 5
  activationSteps = 10  # doesn't matter
  pctConnected = 0.05
  # New cell inertia is `oldInertia * decay + innervation`
  decay = 0.9
  tp = stp.TP(nInputs, nCols, nActive, pctConnected, activationSteps, r=r,
              nActiveInputs=w, coincThreshold=coincThreshold, decay=decay)
  tp.learn = True

  allIndices = list(xrange(nInputs))

  seqLen = 80

  # Generate a sequence
  seq = [set(r.sample(allIndices, w)) for _ in xrange(seqLen)]
  sumSets = set([])
  for p in seq:
    sumSets = sumSets.union(p)

  initTop = stp.topColumns(tp, seq, 80)
  print "Initial cols", initTop

  # Run through pattern a few times
  nPasses = 3
  for _ in xrange(nPasses):
    for p in seq:
      tp.runOne(p)

  interTop = stp.topColumns(tp, seq, 80)
  print "Intermediate cols", interTop
  print "Overlap: ", len(interTop.intersection(initTop))

  for _ in xrange(nPasses):
    for p in seq:
      tp.runOne(p)

  finalTop = stp.topColumns(tp, seq, 80)
  print "Final cols", finalTop
  print "Overlap: ", len(finalTop.intersection(interTop))
