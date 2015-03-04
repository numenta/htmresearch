#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
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

"""TODO"""

import random

from sensorimotor import stp



if __name__ == "__main__":
  r = random.Random(42)
  nInputs = 8000
  w = int(float(nInputs) * 0.02)
  coincThreshold = int(float(w) / 4.0)
  nCols = 2000
  nActive = 40
  activationSteps = 10
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
