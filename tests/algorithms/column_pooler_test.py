# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

import unittest
import numpy as np

import scipy.sparse as sparse

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin
)
from htmresearch.algorithms.column_pooler import ColumnPooler, realDType



class MonitoredColumnPooler(TemporalMemoryMonitorMixin, ColumnPooler):
  pass


class ExtensiveColumnPoolerTest(unittest.TestCase):
  """Algorithmic tests for the ColumnPooler region."""

  inputWidth = 2048 * 8
  numInputActiveBits = 0.02 * inputWidth
  outputWidth = 2048
  numOutputActiveBits = 40
  seed = 42


  def testNewInputs(self):
    self.init()

    # feed the first input, a random SDR should be generated
    firstPattern = self.getObject(1)
    self.learn(firstPattern, numRepetitions=1, newObject=True)
    currentRespresentation = set(self.pooler.getActiveCells())
    self.assertEqual(len(currentRespresentation), self.numOutputActiveBits)

    # feed a new input for the same object, the previous SDR should persist
    secondPattern = self.getObject(1)
    self.learn(secondPattern, numRepetitions=1, newObject=False)
    newRepresentation = set(self.pooler.getActiveCells())
    self.assertNotEqual(firstPattern, secondPattern)
    self.assertEqual(newRepresentation, currentRespresentation)

    # without sensory input, the SDR should persist as well
    self.learn([None], numRepetitions=1, newObject=False)
    newRepresentation = set(self.pooler.getActiveCells())
    self.assertEqual(newRepresentation, currentRespresentation)


  def setUp(self):
    """
    Sets up the test.
    """
    self.pooler = None
    self.proximalPatternMachine = PatternMachine(
      self.inputWidth,
      self.numOutputActiveBits,
      self.seed
    )
    self.patternId = 0
    np.random.seed(self.seed)


  def getPattern(self):
    """
    Returns a random proximal input pattern.
    """
    pattern = self.proximalPatternMachine.get(self.patternId)
    self.patternId += 1
    return pattern


  def learn(self,
            feedforwardPatterns,
            lateralPatterns=None,
            numRepetitions=1,
            randomOrder=True,
            newObject=True):
    """
    Learns a single object, with the provided patterns.

    @feedforwardPatterns   (list(set)) List of proximal input patterns
    @lateralPatterns       (list(set)) List of distal input patterns, or None
                                       if no lateral input is used. This is
                                       expected to have the same length as
                                       feedforwardPatterns
    @numRepetitions        (int)       Number of times the patterns will be fed
    @randomOrder           (bool)      If true, the order of patterns will be
                                       shuffled at each repetition
    """
    if newObject:
      self.pooler.mmClearHistory()
      self.pooler.reset()

    # set-up
    indices = range(len(feedforwardPatterns))
    if lateralPatterns is None:
      lateralPatterns = [None] * len(feedforwardPatterns)

    for _ in xrange(numRepetitions):
      if randomOrder:
        np.random.shuffle(indices)

      for idx in indices:
        self.pooler.compute(feedforwardPatterns[idx],
                            activeExternalCells=lateralPatterns[idx],
                            learn=True)


  def infer(self, feedforwardPattern, lateralPattern=None, printMetrics=False):
    """
    Feeds a single pattern to the column pooler (as well as an eventual lateral
    pattern).

    @param feedforwardPattern       (set) Input proximal pattern to the pooler
    @param lateralPattern           (set) Input dislal pattern to the pooler
    @param printMetrics             (bool) If true, will print cell metrics
    """
    self.pooler.compute(feedforwardPattern,
                        activeExternalCells=lateralPattern,
                        learn=False)

    if printMetrics:
      print self.pooler.mmPrettyPrintMetrics(self.pooler.mmGetDefaultMetrics())


  def getObject(self, numPatterns):
    """
    Creates a list of patterns, for a given object.
    """
    return [self.getPattern() for _ in xrange(numPatterns)]


  def init(self, overrides=None):
    """
    Creates the column pooler with specified parameter overrides.
    """
    params = self._computeParams(overrides)
    self.pooler = MonitoredColumnPooler(**params)


  def getDefaultPoolerParams(self):
    """
    Default params to be used for the column pooler, if no override is
    specified.
    """
    return {
      "inputWidth": self.inputWidth,
      "numActivecolumnsPerInhArea": self.numOutputActiveBits,
      "synPermProximalInc": 0.1,
      "synPermProximalDec": 0.001,
      "maxSynapsesPerSegment": self.inputWidth,
      "columnDimensions": (self.outputWidth,),
      "initialPermanence": 0.5,
      "connectedPermanence": 0.6,
      "minThreshold": 20,
      "maxNewSynapseCount": 30,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.02,
      "predictedSegmentDecrement": 0.08,
      "activationThreshold": 20,
      "seed": self.seed,
      "learnOnOneCell": False,
    }


  def _computeParams(self, overrides):
    """
    Overrides the default parameters with provided values and returns the
    parameters to use in the constructor.
    """
    if overrides is None:
      overrides = {}

    params = self.getDefaultPoolerParams()
    params.update(overrides)
    return params



if __name__ == "__main__":
  unittest.main()
