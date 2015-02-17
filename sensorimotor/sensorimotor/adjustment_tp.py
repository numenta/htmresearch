#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have purchased from
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

import collections
import operator
import random


class Cell(object):

  def __init__(self, nInputs, pctConnected, activationSteps, connectedThreshold=0.5, inc=0.3, dec=0.2, r=None):
    self.r = r or random.Random(42)

    nConnected = int(float(nInputs) * pctConnected)
    self.weights = ([1.0 for _ in xrange(nConnected)] +
                    [0.0 for _ in xrange(nInputs - nConnected)])
    self.r.shuffle(self.weights)

    self.connectedThreshold = connectedThreshold

    self.inc = inc
    self.dec = dec

    self.recentInnervations = collections.deque(maxlen=activationSteps)

  def computeInnervation(self, inputIndices, oneStep=False):
    innervation = sum(int(self.weights[i] > self.connectedThreshold) for i in inputIndices)
    if oneStep:
      return innervation

    self.recentInnervations.append(innervation)
    return sum(self.recentInnervations)

  def adjustWeights(self, inputIndices):
    inputIndices = set(inputIndices)
    for i in xrange(len(self.weights)):
      if i in inputIndices:
        self.weights[i] = min(1.0, self.weights[i] + self.inc)
      else:
        self.weights[i] = max(0.0, self.weights[i] - self.dec)

  def reset(self):
    self.recentInnervations.clear()


class TP(object):

  def __init__(self, nInputs, nCols, nActive, pctConnected, activationSteps, connectedThreshold=0.5, inc=0.3, dec=0.2, r=None):
    self.r = r or random.Random(42)
    self.nActive = nActive
    self.cells = [Cell(nInputs, pctConnected, activationSteps, connectedThreshold, inc, dec, r=r) for _ in xrange(nCols)]

    self.learn = True

  def runOne(self, p):
    innervations = [(c.computeInnervation(p), i) for i, c in enumerate(self.cells)]
    topIndices = [pair[1] for pair in sorted(innervations, reverse=True)[:self.nActive]]

    for i in topIndices:
      if self.learn:
        self.cells[i].adjustWeights(p)

  def reset(self):
    for c in self.cells:
      c.reset()


if __name__ == "__main__":
  r = random.Random(42)
  nInputs = 8000
  w = int(float(nInputs) * 0.02)
  nCols = 2000
  nActive = 40
  activationSteps = 10
  pctConnected = 0.05
  tp = TP(nInputs, nCols, nActive, pctConnected, activationSteps, r=r)

  allIndices = list(xrange(nInputs))

  seqLen = 80

  # Generate a sequence
  seq = [set(r.sample(allIndices, w)) for _ in xrange(seqLen)]
  sumSets = set([])
  for p in seq:
    sumSets = sumSets.union(p)

  # Determine best initial fits
  initialAverageFits = []
  for c in tp.cells:
    initialAverageFits.append(
        float(sum(c.computeInnervation(p, oneStep=True) for p in seq)) / float(seqLen))

  print "Top initial average fits:"
  for pair in sorted(enumerate(initialAverageFits), key=operator.itemgetter(1), reverse=True)[:40]:
    print "%i: %i" % pair

  initialTopFits = []
  for c in tp.cells:
    overlaps = [c.computeInnervation(p, oneStep=True) for p in seq]
    overThreshold = sum([int(v > int(float(w) * pctConnected * 1.5)) for v in overlaps])
    initialTopFits.append(overThreshold)

  print "Top initial best fits:"
  for pair in sorted(enumerate(initialTopFits), key=operator.itemgetter(1), reverse=True)[:40]:
    print "%i: %i" % pair

  # Run through pattern a few times
  nPasses = 3
  for _ in xrange(nPasses):
    for p in seq:
      tp.runOne(p)

  # Determine best fits and compare to original fits
  finalAverageFits = []
  for c in tp.cells:
    finalAverageFits.append(
        float(sum(c.computeInnervation(p, oneStep=True) for p in seq)) / float(seqLen))

  print "Top final average fits:"
  for pair in sorted(enumerate(finalAverageFits), key=operator.itemgetter(1), reverse=True)[:40]:
    print "%i: %i" % pair

  finalTopFits = []
  for c in tp.cells:
    overlaps = [c.computeInnervation(p, oneStep=True) for p in seq]
    overThreshold = sum([int(v > int(float(w) * pctConnected * 1.5)) for v in overlaps])
    finalTopFits.append(overThreshold)

  print "Top final best fits:"
  for pair in sorted(enumerate(finalTopFits), key=operator.itemgetter(1), reverse=True)[:40]:
    print "%i: %i" % pair
