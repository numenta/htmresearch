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

import collections
import math
import operator
import random


def computeSigmoid(x, infl=0.5, sharpness=1.0, _scaleY=True):
  scaledX = (x - infl) / min(1.0 - infl, infl)
  logistic = 1 / (1 + math.exp(-2 * sharpness * scaledX))
  if _scaleY:
    minVal = computeSigmoid(0.0, infl, sharpness, _scaleY=False)
    delta = computeSigmoid(1.0, infl, sharpness, _scaleY=False) - minVal
    return (logistic - minVal) / delta
  return logistic


def cellSum(x, max, coincThreshold):
  assert x >= 0 and x <= max
  # x is now a value between 0.0 and 1.0
  x = float(x) / float(max)
  infl = float(coincThreshold) / float(max)
  return computeSigmoid(x, infl, 1.0)


class Cell(object):

  def __init__(self, nInputs, pctConnected, activationSteps,
               connectedThreshold=0.5, inc=0.3, dec=0.002, r=None,
               nActiveInputs=80, coincThreshold=20):
    self.r = r or random.Random()

    nConnected = int(float(nInputs) * pctConnected)
    self.weights = ([1.0 for _ in xrange(nConnected)] +
                    [0.0 for _ in xrange(nInputs - nConnected)])
    self.r.shuffle(self.weights)

    self.connectedThreshold = connectedThreshold

    self.inc = inc
    self.dec = dec

    self.nActiveInputs = nActiveInputs
    self.coincThreshold = coincThreshold

    #self.recentInnervations = collections.deque(maxlen=activationSteps)
    self.inertia = 0.0

  def computeInnervation(self, inputIndices, decay):
    if decay:
      self.inertia *= decay
    connectedActive = sum(int(self.weights[i] > self.connectedThreshold) for i in inputIndices)
    innervation = cellSum(connectedActive, self.nActiveInputs, self.coincThreshold)

    # TODO: Fix hard coded max inertia
    self.inertia = min(self.inertia + innervation, 5.0)
    #self.recentInnervations.append(innervation)
    return self.inertia

  def adjustWeights(self, inputIndices):
    inputIndices = set(inputIndices)
    for i in xrange(len(self.weights)):
      if i in inputIndices:
        self.weights[i] = min(1.0, self.weights[i] + self.inc)
      else:
        self.weights[i] = max(0.0, self.weights[i] - self.dec)

  def reset(self):
    self.inertia = 0.0


class TP(object):

  def __init__(self, nInputs, nCols, nActive, pctConnected, activationSteps,
               connectedThreshold=0.5, inc=0.3, dec=0.1, r=None,
               nActiveInputs=80, coincThreshold=20, decay=0.9):
    """

    :param nInputs: number of input bits
    :param nCols: number of columns in TP
    :param nActive: number of TP columns active at a time
    :param pctConnected: fraction of inputs bits each column should initially
        be connected to
    :param activationSteps: NOT USED - REMOVE
    :param connectedThreshold: float value determining how large permanence
        has to be to count input bit as connected to the column
    :param inc: how much to increment column-bit permanence when column and
        bit are active
    :param dec: how much to decrement column-bit permanence when column is
        active but input bit is not
    :param r: random instance
    :param nActiveInputs: number of input bits that are active at each step.
        can we get rid of this and calculate in the compute method each step?
    :param coincThreshold: the target column inputs active for coincidence
        detection. This is used to determine the sigmoid function inflection
        point when computing the activity weight for a column
    :param decay: fraction of new inertia that comes from previous value. The
        closer to 1, the more old inputs have an effect on which columns become
        active. If 0.5, then half all previous inputs have a cumulative impact
        equal to the current input on which columns are active.
    """
    self.r = r or random.Random(42)
    self.nActive = nActive
    self.cells = [Cell(nInputs, pctConnected, activationSteps,
                       connectedThreshold, inc, dec, r=r,
                       nActiveInputs=nActiveInputs, coincThreshold=20)
                  for _ in xrange(nCols)]

    self.learn = True
    self.decay = decay

  def runOne(self, p):
    innervations = [(c.computeInnervation(p, decay=self.decay), i) for i, c in enumerate(self.cells)]
    topIndices = [pair[1] for pair in sorted(innervations, reverse=True)[:self.nActive]]

    for i in topIndices:
      if self.learn:
        self.cells[i].adjustWeights(p)

    return topIndices

  def reset(self):
    for c in self.cells:
      c.reset()


def topColumns(tp, seq, topN):
  initLearn = tp.learn
  tp.learn = False
  tp.reset()

  activeCols = collections.defaultdict(int)
  for pattern in seq:
    indices = tp.runOne(pattern)
    for i in indices:
      activeCols[i] += 1

  tp.learn = initLearn
  tp.reset()

  return set(sorted([v[0] for v in sorted(activeCols.iteritems(), key=lambda v: v[1], reverse=True)[:topN]]))
