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

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin
)
from htmresearch.algorithms.column_pooler import ColumnPooler



class MonitoredColumnPooler(TemporalMemoryMonitorMixin, ColumnPooler):
  pass



class ExtensiveColumnPoolerTest(unittest.TestCase):
  """
  Algorithmic tests for the ColumnPooler region.

  Each test actually tests multiple aspects of the algorithm. For more
  atomic tests refer to column_pooler_unit_test.

  In these tests, the proximally-fed SDR's are simulated as unique (location,
  feature) pairs regardless of actual locations and features, unless stated
  otherwise.
  """
  # TODO: add robustness to spatial noice

  inputWidth = 2048 * 8
  numInputActiveBits = int(0.02 * inputWidth)
  outputWidth = 2048
  numOutputActiveBits = 40
  seed = 42


  def testNewInputs(self):
    """
    Checks that the behavior is correct when facing unseed inputs.
    """
    self.init()

    # feed the first input, a random SDR should be generated
    initialPattern = self.generateObject(1)
    self.learn(initialPattern, numRepetitions=1, newObject=True)
    representation = self._getActiveRepresentation()
    self.assertEqual(
      len(representation),
      self.numOutputActiveBits,
      "The generated representation is incorrect"
    )

    # feed a new input for the same object, the previous SDR should persist
    newPattern = self.generateObject(1)
    self.learn(newPattern, numRepetitions=1, newObject=False)
    newRepresentation = self._getActiveRepresentation()
    self.assertNotEqual(initialPattern, newPattern)
    self.assertEqual(
      newRepresentation,
      representation,
      "The SDR did not persist when learning the same object"
    )

    # without sensory input, the SDR should persist as well
    emptyPattern = [set()]
    self.learn(emptyPattern, numRepetitions=1, newObject=False)
    newRepresentation = self._getActiveRepresentation()
    self.assertEqual(
      newRepresentation,
      representation,
      "The SDR did not persist after an empty input."
    )


  def testLearnSinglePattern(self):
    """
    A single pattern is learnt for a single object.
    """
    self.init()

    object = self.generateObject(1)
    self.learn(object, numRepetitions=3, newObject=True)
    # check that the active representation is sparse
    representation = self._getActiveRepresentation()
    self.assertEqual(
      len(representation),
      self.numOutputActiveBits,
      "The generated representation is incorrect"
    )

    # check that the pattern was correctly learnt
    self.infer(feedforwardPattern=object[0])
    self.assertEqual(
      self._getActiveRepresentation(),
      representation,
      "The pooled representation is not stable"
    )

    # present new pattern, it should be mapped to the same representation
    object = self.generateObject(1)
    self.learn(object, numRepetitions=3, newObject=False)
    # check that the active representation is sparse
    newRepresentation = self._getActiveRepresentation()
    self.assertEqual(
      newRepresentation,
      representation,
      "The new pattern did not map to the same object representation"
    )

    # check that the pattern was correctly learnt and is stable
    self.infer(feedforwardPattern=object[0])
    self.assertEqual(
      self._getActiveRepresentation(),
      representation,
      "The pooled representation is not stable"
    )


  def testLearnSingleObject(self):
    """
    Many patterns are learnt for a single object.
    """
    self.init()

    object = self.generateObject(numPatterns=5)
    self.learn(object, numRepetitions=3, randomOrder=True, newObject=True)
    representation = self._getActiveRepresentation()

    # check that all patterns map to the same object
    for pattern in object:
      self.infer(feedforwardPattern=pattern)
      self.assertEqual(
        self._getActiveRepresentation(),
        representation,
        "The pooled representation is not stable"
      )

    # if activity stops, check that the representation persists
    self.infer(feedforwardPattern=set())
    self.assertEqual(
      self._getActiveRepresentation(),
      representation,
      "The pooled representation did not persist"
    )


  def testLearnTwoObjectNoCommonPattern(self):
    """
    Same test as before, using two objects, without common pattern.
    """
    self.init()

    objectA = self.generateObject(numPatterns=5)
    self.learn(objectA, numRepetitions=3, randomOrder=True, newObject=True)
    representationA = self._getActiveRepresentation()

    objectB = self.generateObject(numPatterns=5)
    self.learn(objectB, numRepetitions=3, randomOrder=True, newObject=True)
    representationB = self._getActiveRepresentation()

    self.assertNotEqual(representationA, representationB)

    # check that all patterns map to the same object
    for pattern in objectA:
      self.infer(feedforwardPattern=pattern)
      self.assertEqual(
        self._getActiveRepresentation(),
        representationA,
        "The pooled representation for the first object is not stable"
      )

    # check that all patterns map to the same object
    for pattern in objectB:
      self.infer(feedforwardPattern=pattern)
      self.assertEqual(
        self._getActiveRepresentation(),
        representationB,
        "The pooled representation for the second object is not stable"
      )

    # feed union of patterns in object A
    pattern = objectA[0] | objectA[1]
    self.infer(feedforwardPattern=pattern)
    self.assertEqual(
      self._getActiveRepresentation(),
      representationA,
      "The active representation is incorrect"
    )

    # feed unions of patterns in objects A and B
    pattern = objectA[0] | objectB[0]
    self.infer(feedforwardPattern=pattern)
    self.assertEqual(
      self._getActiveRepresentation(),
      representationA | representationB,
      "The active representation is incorrect"
    )



  def testLearnTwoObjectsOneCommonPattern(self):
    """
    Same test as before, except the two objects share a pattern
    """
    self.init()

    objectA = self.generateObject(numPatterns=5)
    self.learn(objectA, numRepetitions=3, randomOrder=True, newObject=True)
    representationA = self._getActiveRepresentation()

    objectB = self.generateObject(numPatterns=5)
    objectB[0] = objectA[0]
    self.learn(objectB, numRepetitions=3, randomOrder=True, newObject=True)
    representationB = self._getActiveRepresentation()

    self.assertNotEqual(representationA, representationB)

    # check that all patterns except the common one map to the same object
    for pattern in objectA[1:]:
      self.infer(feedforwardPattern=pattern)
      self.assertEqual(
        self._getActiveRepresentation(),
        representationA,
        "The pooled representation for the first object is not stable"
      )

    # check that all patterns except the common one map to the same object
    for pattern in objectB[1:]:
      self.infer(feedforwardPattern=pattern)
      self.assertEqual(
        self._getActiveRepresentation(),
        representationB,
        "The pooled representation for the second object is not stable"
      )

    # feed shared pattern
    pattern = objectA[0]
    self.infer(feedforwardPattern=pattern)
    self.assertEqual(
      self._getActiveRepresentation(),
      representationA | representationB,
      "The active representation is incorrect"
    )

    # feed union of patterns in object A
    pattern = objectA[1] | objectA[2]
    self.infer(feedforwardPattern=pattern)
    self.assertEqual(
      self._getActiveRepresentation(),
      representationA,
      "The active representation is incorrect"
    )

    # feed unions of patterns in objects A and B
    pattern = objectA[1] | objectB[1]
    self.infer(feedforwardPattern=pattern)
    self.assertEqual(
      self._getActiveRepresentation(),
      representationA | representationB,
      "The active representation is incorrect"
    )

  def testLearnThreeObjectsOneCommonPattern(self):
    """
    Same test as before, except the two objects share a pattern
    """
    self.init()

    objectA = self.generateObject(numPatterns=5)
    self.learn(objectA, numRepetitions=3, randomOrder=True, newObject=True)
    representationA = self._getActiveRepresentation()

    objectB = self.generateObject(numPatterns=5)
    objectB[0] = objectA[0]
    self.learn(objectB, numRepetitions=3, randomOrder=True, newObject=True)
    representationB = self._getActiveRepresentation()

    objectC = self.generateObject(numPatterns=5)
    objectC[0] = objectB[1]
    self.learn(objectC, numRepetitions=3, randomOrder=True, newObject=True)
    representationC = self._getActiveRepresentation()

    self.assertNotEquals(representationA, representationB, representationC)

    # check that all patterns except the common one map to the same object
    for pattern in objectA[1:]:
      self.infer(feedforwardPattern=pattern)
      self.assertEqual(
        self._getActiveRepresentation(),
        representationA,
        "The pooled representation for the first object is not stable"
      )

    # check that all patterns except the common one map to the same object
    for pattern in objectB[2:]:
      self.infer(feedforwardPattern=pattern)
      self.assertEqual(
        self._getActiveRepresentation(),
        representationB,
        "The pooled representation for the second object is not stable"
      )

    # check that all patterns except the common one map to the same object
    for pattern in objectC[1:]:
      self.infer(feedforwardPattern=pattern)
      self.assertEqual(
        self._getActiveRepresentation(),
        representationC,
        "The pooled representation for the third object is not stable"
      )

    # feed shared pattern between A and B
    pattern = objectA[0]
    self.infer(feedforwardPattern=pattern)
    self.assertEqual(
      self._getActiveRepresentation(),
      representationA | representationB,
      "The active representation is incorrect"
    )

    # feed shared pattern between B and C
    pattern = objectB[1]
    self.infer(feedforwardPattern=pattern)
    self.assertEqual(
      self._getActiveRepresentation(),
      representationB | representationC,
      "The active representation is incorrect"
    )

    # feed union of patterns in object A
    pattern = objectA[1] | objectA[2]
    self.infer(feedforwardPattern=pattern)
    self.assertEqual(
      self._getActiveRepresentation(),
      representationA,
      "The active representation is incorrect"
    )

    # feed unions of patterns to activate all objects
    pattern = objectA[1] | objectB[1]
    self.infer(feedforwardPattern=pattern)
    self.assertEqual(
      self._getActiveRepresentation(),
      representationA | representationB | representationC,
      "The active representation is incorrect"
    )


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


  # Wrappers around ColumnPooler API

  def learn(self,
            feedforwardPatterns,
            lateralPatterns=None,
            numRepetitions=1,
            randomOrder=True,
            newObject=True):
    """
    Parameters:
    ----------------------------
    Learns a single object, with the provided patterns.

    @param   feedforwardPatterns   (list(set))
             List of proximal input patterns

    @param   lateralPatterns       (list(list(set)))
             List of distal input patterns, or None. If no lateral input is
             used. The outer list is expected to have the same length as
             feedforwardPatterns, whereas each inner list's length is the
             number of cortical columns which are distally connected to the
             pooler.

    @param   numRepetitions        (int)
             Number of times the patterns will be fed

    @param   randomOrder           (bool)
             If true, the order of patterns will be shuffled at each
             repetition

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


  def infer(self,
            feedforwardPattern,
            lateralPatterns=None,
            printMetrics=False):
    """
    Feeds a single pattern to the column pooler (as well as an eventual lateral
    pattern).

    Parameters:
    ----------------------------
    @param feedforwardPattern       (set)
           Input proximal pattern to the pooler

    @param lateralPatterns          (list(set))
           Input dislal patterns to the pooler (one for each neighboring CC's)

    @param printMetrics             (bool)
           If true, will print cell metrics

    """
    self.pooler.compute(feedforwardPattern,
                        activeExternalCells=lateralPatterns,
                        learn=False)

    if printMetrics:
      print self.pooler.mmPrettyPrintMetrics(
        self.pooler.mmGetDefaultMetrics()
      )


  # Helper functions

  def generatePattern(self):
    """
    Returns a random proximal input pattern.
    """
    pattern = self.proximalPatternMachine.get(self.patternId)
    self.patternId += 1
    return pattern


  def generateObject(self, numPatterns):
    """
    Creates a list of patterns, for a given object.
    """
    return [self.generatePattern() for _ in xrange(numPatterns)]


  def init(self, overrides=None):
    """
    Creates the column pooler with specified parameter overrides.
    """
    params = self._computeParams(overrides)
    self.pooler = MonitoredColumnPooler(**params)


  def _getActiveRepresentation(self):
    """
    Retrieves the current active representation in the pooler.
    """
    if self.pooler is None:
      raise ValueError("No pooler has been instantiated")

    return set(self.pooler.getActiveCells())


  def _getDefaultPoolerParams(self):
    """
    Default params to be used for the column pooler, if no override is
    specified.
    """
    return {
      "inputWidth": self.inputWidth,
      "numActivecolumnsPerInhArea": self.numOutputActiveBits,
      "synPermProximalInc": 0.1,
      "synPermProximalDec": 0.001,
      "initialProximalPermanence": 0.51,
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

    params = self._getDefaultPoolerParams()
    params.update(overrides)
    return params



if __name__ == "__main__":
  unittest.main()
