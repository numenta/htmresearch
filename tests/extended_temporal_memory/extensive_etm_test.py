#!/usr/bin/env python
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

from nupic.data.generators.pattern_machine import PatternMachine
from nupic.support.unittesthelpers.abstract_temporal_memory_test import AbstractTemporalMemoryTest

from htmresearch.algorithms.extended_temporal_memory import ExtendedTemporalMemory


class ExtensiveExtendedTemporalMemoryTest(AbstractTemporalMemoryTest, unittest.TestCase):
  """
  Tests the specific aspects of extended temporal memory (external and apical input, learning on
  one cell, etc.

  Note: these tests use typical default parameters for ETM, thus learning takes multiple pass and
  can be slow.

  ==============================================================================
                  Learning with external input
  ==============================================================================


  ==============================================================================
                  Learning with apical input
  ==============================================================================


  ==============================================================================
                  Other tests
  ==============================================================================

  """

  n = 2048
  w = range(38, 43)


  def testE1(self):
    """Joint proximal and external first order learning, correct inputs at test time."""
    self.init({"cellsPerColumn": 1})

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)

    self._testTM(proximalInputA, activeExternalCellsSequence=externalInputA)
    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


  def testE2(self):
    """Joint proximal and external first order learning, no external input at test time."""
    self.init({"cellsPerColumn": 1})

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)

    self._testTM(proximalInputA, activeExternalCellsSequence=None)
    self.assertAllActiveWereUnpredicted()


  def testE3(self):
    """Joint proximal and external first order learning, incorrect external input at test time."""
    self.init({"cellsPerColumn": 1})

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)

    self._testTM(proximalInputA, activeExternalCellsSequence=externalInputB)
    self.assertAlmostAllActiveWereUnpredicted()


  def testE4(self):
    """Same as E3, but the threshold is set to infer on proximal OR external input.

    To do so, set the thresholds to be half the max new synapse count.
    """
    self.init({"cellsPerColumn": 1,
               "activationThreshold": 15,
               "minThreshold": 15})

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)

    self._testTM(proximalInputA, activeExternalCellsSequence=externalInputB)
    self.assertAllActiveWerePredicted()


  def testE5(self):
    """Like E1, with slower learning."""
    self.init({"cellsPerColumn": 4,
               "initialPermanence": 0.2,
               "connectedPermanence": 0.7,
               "permanenceIncrement": 0.2})

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(4):
      # feed sequence multiple times to compensate for slower learning rate
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)

    self._testTM(proximalInputA, activeExternalCellsSequence=externalInputA)
    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


  def testE6(self):
    """Joint proximal and external first order learning, incorrect proximal input at test time."""
    self.init({"cellsPerColumn": 1})

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)

    self._testTM(proximalInputB, activeExternalCellsSequence=externalInputA)
    self.assertAlmostAllActiveWereUnpredicted()


  def testE7(self):
    """Joint proximal and external higher order learning learning."""
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)

    self._testTM(proximalInputA, activeExternalCellsSequence=externalInputA)
    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


  def testE8(self):
    """Joint proximal and external higher order learning learning, possible ambiguity on proximal."""
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)
      self.feedTM(proximalInputB, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)

    self._testTM(proximalInputA, activeExternalCellsSequence=externalInputA)
    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


  def testE9(self):
    """Joint proximal and external higher order learning learning, possible ambiguity on external."""
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    externalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputB,
                  formInternalConnections=True)

    self._testTM(proximalInputA, activeExternalCellsSequence=externalInputA)
    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


  def testE10(self):
    """Same as E8, but last pattern is incorrect given the high order the sequence."""
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    # A B C D E
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[0] = max(numbers) + 1
    # X B C D E
    proximalInputC = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[-2] = max(numbers) + 1
    # X B C D F
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      # train on A B C D E and X B C D F
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)
      self.feedTM(proximalInputB, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)

    # test on X B C D E
    self._testTM(proximalInputC, activeExternalCellsSequence=externalInputA)
    self.assertAllActiveWereUnpredicted()


  def testE11(self):
    """Repeated motor command copy as external input.

    External input is of the form PQPQPQPQPQPQ...
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = [100, 200] * 50 + [None]
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)


    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)

    self._testTM(proximalInputA, activeExternalCellsSequence=externalInputA)
    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


  def testE12(self):
    """Simple learnOnOneCell test.

    Train on ABCADC, check that C has the same representation in ADC and ABC.
    """
    self.init({"learnOnOneCell": True})
    self.assertTrue(self.tm.learnOnOneCell)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    numbers[0] = 50
    numbers[-2] = 75
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 10)
    numbers[0] = 50
    numbers[-2] = 75
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, formInternalConnections=True)
      self.feedTM(proximalInputB, formInternalConnections=True)

    self._testTM(proximalInputA)
    predictedActiveA = self.tm.mmGetTracePredictedActiveCells().data[-1]
    self._testTM(proximalInputB)
    predictedActiveB = self.tm.mmGetTracePredictedActiveCells().data[-1]
    self.assertEqual(predictedActiveA, predictedActiveB)


  def testE13(self):
    """learnOnOneCell with motor command should not impair behavior.

    Using learnOnOneCell, does the same basic test as E1.
    """
    self.init({"learnOnOneCell": True})

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)

    self._testTM(proximalInputA, activeExternalCellsSequence=externalInputA)
    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


  def testA1(self):
    """Basic feedback disambiguation.

    Train on ABCDE with F1, XBCDE with F2.
    Test with BCDE. Without feedback, two patterns are predicted.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    # A B C D E
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    # B C D E
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[1:])
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    # X B C D F
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeApicalCellsSequence=feedbackA, formInternalConnections=True)
      self.feedTM(proximalInputB, activeApicalCellsSequence=feedbackB, formInternalConnections=True)

    self._testTM(testProximalInput, activeApicalCellsSequence=None)
    self.assertAllActiveWerePredicted()
    # many extra predictions
    predictedInactiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTracePredictedInactiveColumns())
    self.assertTrue(predictedInactiveColumnsMetric > min(self.w))


  def testA2(self):
    """Basic feedback disambiguation.

    Train on ABCDE with F1, XBCDE with F2.
    Test with BCDE. With feedback, one pattern is expected.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    # A B C D E
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    # B C D E
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[1:])
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    # X B C D F
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeApicalCellsSequence=feedbackA, formInternalConnections=True)
      self.feedTM(proximalInputB, activeApicalCellsSequence=feedbackB, formInternalConnections=True)

    self._testTM(testProximalInput, activeApicalCellsSequence=feedbackA)
    self.assertAllActiveWerePredicted()
    # many extra predictions
    predictedInactiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTracePredictedInactiveColumns())
    self.assertTrue(predictedInactiveColumnsMetric > min(self.w))


  def testA3(self):
    """Basic feedback disambiguation.

    Train on ABCDE with F1, XBCDE with F2.
    Test with BCDE. With feedback, one pattern is expected.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    # A B C D E
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    # B C D E
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[1:])
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    # X B C D F
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeApicalCellsSequence=feedbackA, formInternalConnections=True)
      self.feedTM(proximalInputB, activeApicalCellsSequence=feedbackB, formInternalConnections=True)

    self._testTM(testProximalInput, activeApicalCellsSequence=feedbackB)
    self.assertAllActiveWereUnpredicted()


  def testA4(self):
    """Robustness to temporal noise with feedback.

    Train on ABCDE with F1, XBCDE with F2.
    Test with ACDF (one step is missing). Without feedback, both E and F are predicted."""
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    print numbers
    # A B C D E
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    print numbers
    # X B C D F
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)
    # A C D F
    print [numbers[0]] + numbers[2:]
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[:6] + numbers[7:])

    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeApicalCellsSequence=None, formInternalConnections=True)
      self.feedTM(proximalInputB, activeApicalCellsSequence=None, formInternalConnections=True)

    self._testTM(testProximalInput, activeApicalCellsSequence=None)
    # TODO: add asserts


  def testA5(self):
    """Robustness to temporal noise with feedback.

    Train on ABCDE with F1, XBCDE with F2.
    Test with ACDF (one step is missing). With feedback F1, burst."""
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    # A B C D E
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    # X B C D F
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)
    # A C D F
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[:5] + numbers[6:])

    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeApicalCellsSequence=feedbackA, formInternalConnections=True)
      self.feedTM(proximalInputB, activeApicalCellsSequence=feedbackB, formInternalConnections=True)

    self._testTM(testProximalInput, activeApicalCellsSequence=feedbackA)
    print self.tm.mmGetTraceUnpredictedActiveColumns().data
    # TODO: add asserts


  def testA6(self):
    """Robustness to temporal noise with feedback.

    Train on ABCDE with F1, XBCDE with F2.
    Test with AZCDF (one step is corrupted). Without feedback, both E and F are predicted."""
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    # A B C D E
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    # X B C D F
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)
    # A C D F
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[:5] + numbers[6:])

    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(10): # does not work else???
      self.feedTM(proximalInputA, activeApicalCellsSequence=feedbackA, formInternalConnections=True)
      self.feedTM(proximalInputB, activeApicalCellsSequence=feedbackB, formInternalConnections=True)

    self._testTM(testProximalInput, activeApicalCellsSequence=feedbackA)
    print [len(x) for x in self.tm.mmGetTraceUnpredictedActiveColumns().data]
    # TODO: add asserts


  def testA7(self):
    """Robustness to temporal noise with feedback.

    Train on ABCDE with F1, XBCDE with F2.
    Test with AZCDF (one step is corrupted). With feedback F1, burst."""
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    # A B C D E
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    # X B C D F
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)
    # A C D F
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[:5] + numbers[6:])

    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2): # does not work else???
      self.feedTM(proximalInputA, activeApicalCellsSequence=feedbackA, formInternalConnections=True)
      self.feedTM(proximalInputB, activeApicalCellsSequence=feedbackB, formInternalConnections=True)

    self._testTM(testProximalInput, activeApicalCellsSequence=feedbackA)
    print [len(x) for x in self.tm.mmGetTraceUnpredictedActiveColumns().data]
      # TODO: add asserts


  def testA8(self):
    """Robustness to temporal noise with feedback.

    Train on ABCDE with F1, XBCDE with F2.
    Test with AZCDF (one step is corrupted). With feedback F2, no bursting, almost no extra."""
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    # A B C D E
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    # X B C D F
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)
    # A C D F
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[:5] + numbers[6:])

    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2): # does not work else???
      self.feedTM(proximalInputA, activeApicalCellsSequence=feedbackA, formInternalConnections=True)
      self.feedTM(proximalInputB, activeApicalCellsSequence=feedbackB, formInternalConnections=True)

    self._testTM(testProximalInput, activeApicalCellsSequence=feedbackA)
    print [len(x) for x in self.tm.mmGetTraceUnpredictedActiveColumns().data]


  def testA9(self):
    """Without lateral input, feedback is ineffective.

    Train on ABCDE with F1, disabling internal connections.
    Test with ABCDE and feedback F1. Should burst constantly."""
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeApicalCellsSequence=feedbackA, formInternalConnections=False)

    self._testTM(proximalInputA, activeApicalCellsSequence=feedbackA)
    self.assertAllActiveWereUnpredicted()


  def testO1(self):
    """Joint external / apical inputs.

    Train on ABCDE / PQRST with F1, XBCDEF / PQRST with F2.
    Test with BCDE / QRST. Without feedback, both E and F are expected."""
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[1:])
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  activeApicalCellsSequence=None, formInternalConnections=True)
      self.feedTM(proximalInputB, activeExternalCellsSequence=externalInputA,
                  activeApicalCellsSequence=None, formInternalConnections=True)

    self._testTM(testProximalInput, activeExternalCellsSequence=externalInputA[1:],
                 activeApicalCellsSequence=None)
    self.assertAllActiveWerePredicted()
      # TODO: add 'extra' assert


  def testO2(self):
    """Joint external / apical inputs.

    Train on ABCDE / PQRST with F1, XBCDEF / PQRST with F2.
    Test with BCDE / QRST. With feedback F2, burst."""
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[1:])
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  activeApicalCellsSequence=feedbackA, formInternalConnections=True)
      self.feedTM(proximalInputB, activeExternalCellsSequence=externalInputA,
                  activeApicalCellsSequence=feedbackB, formInternalConnections=True)

    self._testTM(testProximalInput, activeExternalCellsSequence=externalInputA[1:],
                 activeApicalCellsSequence=feedbackB)
    self.assertAllActiveWereUnpredicted()


  def testO3(self):
    """Joint external / apical inputs.

    Train on ABCDE / PQRST with F1, XBCDEF / PQRST with F2.
    Test with BCDE / QRXT. With feedback F1, burst."""
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[1:])
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[-3] = max(numbers) + 1
    testExternalInput = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  activeApicalCellsSequence=feedbackA, formInternalConnections=True)
      self.feedTM(proximalInputB, activeExternalCellsSequence=externalInputA,
                  activeApicalCellsSequence=feedbackB, formInternalConnections=True)

    self._testTM(testProximalInput, activeExternalCellsSequence=testExternalInput[1:],
                 activeApicalCellsSequence=feedbackA)
    self.assertAllActiveWereUnpredicted()
      # TODO: change to assert LAST


  def testO4(self):
    """Joint external / apical inputs.

    Train on ABCDE / PQRST with F1, XBCDEF / PQRST with F2.
    Test with BCDE / QRST. With feedback F1, no burst nor extra (feedback disambiguation)"""
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[1:])
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)[:10] + [None]
    feedbackB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  activeApicalCellsSequence=feedbackA, formInternalConnections=True)
      self.feedTM(proximalInputB, activeExternalCellsSequence=externalInputA,
                  activeApicalCellsSequence=feedbackB, formInternalConnections=True)

    self._testTM(testProximalInput, activeExternalCellsSequence=externalInputA[1:],
                 activeApicalCellsSequence=feedbackA)
    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


    def testO5(self):
      pass


  # ==============================
  # Overrides
  # ==============================

  def getTMClass(self):
    return ExtendedTemporalMemory


  def getPatternMachine(self):
    return PatternMachine(self.n, self.w, num=300)


  def getDefaultTMParams(self):
    """We use typical default parameters for the test suite."""
    return {
      "columnDimensions": (self.n,),
      "cellsPerColumn": 32,
      "initialPermanence": 0.5,
      "connectedPermanence": 0.6,
      "minThreshold": 20,
      "maxNewSynapseCount": 30,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.,
      "activationThreshold": 25,
      "seed": 42,
      "learnOnOneCell": False,
    }


  def setUp(self):
    super(ExtensiveExtendedTemporalMemoryTest, self).setUp()

    print ("\n"
           "======================================================\n"
           "Test: {0} \n"
           "{1}\n"
           "======================================================\n"
    ).format(self.id(), self.shortDescription())


  def feedTM(self,
             sequence,
             activeExternalCellsSequence=None,
             activeApicalCellsSequence=None,
             formInternalConnections=True,
             learn=True,
             num=1):
    """
    Needs to be implemented to take advantage of ETM's compute's specific signature.
    :param sequence:                    (list)     Sequence of SDR's to feed the ETM
    :param activeExternalCellsSequence: (list)     Sequence of external inputs
    :param activeApicalCellsSequence:   (list)     Sequence of apical inputs
    :param formInternalConnections:     (boolean)  Flag to determine whether to form connections with
                                                   internal cells within this temporal memory
    :param num:                         (boolean)  Optional parameter to repeat the input sequences

    Note: sequence, activeExternalCells, and activeApicalCells should have the same length.
    As opposed to in the sequence of input pattern, where None resets the TM, having None in the
    external and apical input sequences simply means to such input is received at this instant.
    """

    if activeApicalCellsSequence is None and activeExternalCellsSequence is None:
      return super(ExtensiveExtendedTemporalMemoryTest, self).feedTM(sequence, learn=learn, num=num)

    repeatedSequence = sequence * num
    if activeExternalCellsSequence is not None:
      repeatedExternal = activeExternalCellsSequence * num
    else:
      # no external input
      repeatedExternal = [None] * len(repeatedSequence)
    if activeApicalCellsSequence is not None:
      repeatedApical = activeApicalCellsSequence * num
    else:
      # no apical input
      repeatedApical = [None] * len(repeatedSequence)

    self.tm.mmClearHistory()

    for pattern, externalInput, apicalInput in zip(repeatedSequence,
                                                   repeatedExternal,
                                                   repeatedApical):
      if pattern is None:
        self.tm.reset()
      else:
        self.tm.compute(pattern,
                        activeExternalCells=externalInput,
                        activeApicalCells=apicalInput,
                        formInternalConnections=formInternalConnections,
                        learn=learn)

    if self.VERBOSITY >= 2:
      print self.tm.mmPrettyPrintTraces(self.tm.mmGetDefaultTraces(verbosity=self.VERBOSITY - 1))
      print

    if learn and self.VERBOSITY >= 3:
      print self.tm.mmPrettyPrintConnections()


  def _testTM(self,
              sequence,
              activeExternalCellsSequence=None,
              activeApicalCellsSequence=None,
              formInternalConnections=True):
    self.feedTM(sequence,
                activeExternalCellsSequence=activeExternalCellsSequence,
                activeApicalCellsSequence=activeApicalCellsSequence,
                learn=False)

    print self.tm.mmPrettyPrintMetrics(self.tm.mmGetDefaultMetrics())


  # ==============================
  # Helper functions
  # ==============================

  def assertAllActiveWerePredicted(self):
    unpredictedActiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTraceUnpredictedActiveColumns())
    predictedActiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTracePredictedActiveColumns())

    self.assertEqual(unpredictedActiveColumnsMetric.sum, 0)

    self.assertEqual(predictedActiveColumnsMetric.min, min(self.w))
    self.assertEqual(predictedActiveColumnsMetric.max, max(self.w))


  def assertAllInactiveWereUnpredicted(self):
    predictedInactiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTracePredictedInactiveColumns())

    self.assertEqual(predictedInactiveColumnsMetric.sum, 0)


  def assertAllActiveWereUnpredicted(self):
    unpredictedActiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTraceUnpredictedActiveColumns())
    predictedActiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTracePredictedActiveColumns())

    self.assertEqual(predictedActiveColumnsMetric.sum, 0)

    self.assertEqual(unpredictedActiveColumnsMetric.min, min(self.w))
    self.assertEqual(unpredictedActiveColumnsMetric.max, max(self.w))


  def assertAlmostAllActiveWereUnpredicted(self, meanPredActiveThr=1):
    unpredictedActiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTraceUnpredictedActiveColumns())
    predictedActiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTracePredictedActiveColumns())

    self.assertTrue(predictedActiveColumnsMetric.mean < meanPredActiveThr)

    self.assertTrue(unpredictedActiveColumnsMetric.mean > min(self.w))
    self.assertTrue(unpredictedActiveColumnsMetric.mean < max(self.w))
