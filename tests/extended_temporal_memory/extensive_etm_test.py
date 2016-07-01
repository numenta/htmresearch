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

  The parameters used in those tests are the typical default parameters for temporal memory, unless
  stated otherwise in the experiment (when self.init() is called).

      columnDimensions: (2048,)
      cellsPerColumn: 32
      initialPermanence: 0.5
      connectedPermanence: 0.6
      minThreshold: 25
      maxNewSynapseCount: 30
      permanenceIncrement: 0.1
      permanenceDecrement: 0
      activationThreshold: 25
      seed: 42
      learnOnOneCell: False

  It is interesting to note that under those parameters, an pattern needs activation from both
  lateral inputs (internal and externals) to be predicted.

  ==================================================================================================
                                  Learning with external input
  ==================================================================================================

  These tests (labeled "E") simulate sequences from proximal input as well as sequences fed through
  distal dendrites par external neurons (e.g. motor efference copy).

  E1-E6: First-order learning through proximal and external input.

  E7-E10: Higher-order learning through proximal and external input.

  E11: Repeated motor command copy.

  E12-E13: Testing the "learnOnOneCell" feature, fixing the winning cell for each columns between
  subsequent resets.

  ==================================================================================================
                                  Learning with apical input
  ==================================================================================================

  A1-A3: Basic disambiguation through feedback from higher regions.

  A4-A8: Robustness to temporal noise through feedback from higher regions.

  A10: Ineffectiveness of feedback without correct proximal input.

  ==================================================================================================
                                          Other tests
  ==================================================================================================

  O1-O4: Joint learning from proximal/external input with feedback.

  O5-O7: Tests drawn from the previous categories, adding slight spatial noise to input patterns
  at test time.

  """

  n = 2048
  w = range(38, 43)


  def testE1(self):
    """Joint proximal and external first order learning, correct inputs at test time.

    Train on ABCDE/PQRST, test on same sequences should yield perfect predictions.
    """
    self.init({"cellsPerColumn": 1})

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=None,
                  formInternalConnections=True)

    self._testTM(proximalInputA, activeExternalCellsSequence=None)

    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


  def testE2(self):
    """Joint proximal and external first order learning, no external input at test time.

    Train on ABCDE/PQRST, test on ABCDE/None leads to bursting.
    """
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
    """Joint proximal and external first order learning, incorrect external input at test time.

    Train on ABCDE/PQRST, test on ABCDE/VWXYZ leads to bursting.
    """
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
    """Joint proximal and external first order learning, incorrect proximal input at test time.

    Train on ABCDE/PQRST, test on FGHIJ/PQRST.
    """
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
    """Joint proximal and external higher order learning learning.

    Same as E1, with 32 cells per column.
    """
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
    """Joint proximal and external higher order learning learning, possible ambiguity on proximal.

    Train on ABCDE/PQRST and ABCDF/PQRST, test on ABCDE/PQRST does not lead to extra predictions.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 20)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[-2] = max(numbers) + 1
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 20)
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
    """Joint proximal and external higher order learning learning, possible ambiguity on proximal.

    Train on ABCDE/PQRST and XBCDF/PQRST, test on ABCDE/PQRST does not lead to extra predictions.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 20)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 20)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)
      self.feedTM(proximalInputB, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)

    self._testTM(proximalInputA, activeExternalCellsSequence=externalInputA)

    print self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


  def testE10(self):
    """Joint proximal and external higher order learning learning, possible ambiguity on external.

    Train on ABCDE/PQRST and ABCDE/XQRSZ, test on ABCDE/PQRST does not lead to extra predictions.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 20)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 20)
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


  def testE11(self):
    """Same as E8, but the last pattern is incorrect given the high order the sequence."""
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 50)
    # A B C D E
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[0] = max(numbers) + 1
    # X B C D E
    proximalInputC = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[-2] = max(numbers) + 1
    # X B C D F
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 50)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      # train on A B C D E and X B C D F
      self.feedTM(proximalInputA, activeExternalCellsSequence=None,
                  formInternalConnections=True)
      self.feedTM(proximalInputB, activeExternalCellsSequence=None,
                  formInternalConnections=True)

    # test on X B C D E
    self._testTM(proximalInputC, activeExternalCellsSequence=None)
    self.assertAllActiveWereUnpredicted()


  def testE12(self):
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


  def testE13(self):
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


  def testE14(self):
    """learnOnOneCell with motor command should not impair behavior.

    Using learnOnOneCell, does the same basic test as E1.
    """
    self.init({"learnOnOneCell": True})
    self.assertTrue(self.tm.learnOnOneCell)

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
    """Basic feedback disambiguation, test without feedback.

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
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertGreaterEqual(len(predictedInactiveColumns[-1]), min(self.w))


  def testA2(self):
    """Basic feedback disambiguation, test with correct feedback.

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
    # few extra predictions
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertLessEqual(len(predictedInactiveColumns[-1]), 3)


  def testA3(self):
    """Basic feedback disambiguation, test with incorrect feedback.

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
    # burst at last step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertGreaterEqual(len(unpredictedActiveColumns[-1]), min(self.w))

  def testA4(self):
    """Robustness to temporal noise with feedback, test without feedback.

    Train on ABCDE with F1, XBCDE with F2.
    Test with ACDF (one step is missing). Without feedback, both E and F are predicted.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    # A B C D E
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[-2] = max(numbers) + 1
    # A C D F
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[:4] + numbers[5:])
    numbers[0] = max(numbers) + 1
    # X B C D F
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeApicalCellsSequence=None, formInternalConnections=True)
      self.feedTM(proximalInputB, activeApicalCellsSequence=None, formInternalConnections=True)

    self._testTM(testProximalInput, activeApicalCellsSequence=None)
    # no bursting at last step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertEqual(len(unpredictedActiveColumns[-1]), 0)
    # many extra predictions
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertGreaterEqual(len(predictedInactiveColumns[-1]), min(self.w))


  def testA5(self):
    """Robustness to temporal noise with feedback, test with incorrect feedback.

    Train on ABCDE with F1, XBCDE with F2.
    Test with ACDF (one step is missing). With feedback F1, burst.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    # A B C D E
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[-2] = max(numbers) + 1
    # A C D F
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[:4] + numbers[5:])
    numbers[0] = max(numbers) + 1
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
    # burst at last step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertGreaterEqual(len(unpredictedActiveColumns[-1]), min(self.w))


  def testA6(self):
    """Robustness to temporal noise with feedback, test without feedback.

    Train on ABCDE with F1, XBCDE with F2.
    Test with AZCDF (one step is corrupted). Without feedback, both E and F are predicted.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    # A B C D E
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[-2] = max(numbers) + 1
    # A Z C D F
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[:4] + [50] + numbers[5:])
    numbers[0] = max(numbers) + 1
    # X B C D F
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeApicalCellsSequence=None, formInternalConnections=True)
      self.feedTM(proximalInputB, activeApicalCellsSequence=None, formInternalConnections=True)

    self._testTM(testProximalInput, activeApicalCellsSequence=None)
    # no bursting at last step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertEqual(len(unpredictedActiveColumns[-1]), 0)
    # many extra predictions
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertGreaterEqual(len(predictedInactiveColumns[-1]), min(self.w))


  def testA7(self):
    """Robustness to temporal noise with feedback, test with incorrect feedback.

    Train on ABCDE with F1, XBCDE with F2.
    Test with AZCDF (one step is corrupted). With feedback F1, burst
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    # A B C D E
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[-2] = max(numbers) + 1
    # A Z C D F
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[:4] + [50] + numbers[5:])
    numbers[0] = max(numbers) + 1
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
    # burst at last step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertGreaterEqual(len(unpredictedActiveColumns[-1]), min(self.w))


  def testA8(self):
    """Robustness to temporal noise with feedback, test with correct feedback.

    Train on ABCDE with F1, XBCDE with F2.
    Test with AZCDF (one step is corrupted). With feedback F2, no bursting, almost no extra.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    # A B C D E
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[-2] = max(numbers) + 1
    # A Z C D F
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[:4] + [50] + numbers[5:])
    numbers[0] = max(numbers) + 1
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
    # no bursting at last step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertEqual(len(unpredictedActiveColumns[-1]), 0)
    # few extra predictions
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertLessEqual(len(predictedInactiveColumns[-1]), 3)


  def testA9(self):
    """Without lateral input, feedback is ineffective.

    Train on ABCDE with F1, disabling internal connections.
    Test with ABCDE and feedback F1. Should burst constantly.
    """
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
    """Joint external / apical inputs, without feedback.

    Train on ABCDE / PQRST,, XBCDEF / PQRST.
    Test with BCDE / QRST. Without feedback, both E and F are expected.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers[1:])
    numbers[0] = max(numbers) + 1
    numbers[-2] = max(numbers) + 1
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  activeApicalCellsSequence=None, formInternalConnections=True)
      self.feedTM(proximalInputB, activeExternalCellsSequence=externalInputA,
                  activeApicalCellsSequence=None, formInternalConnections=True)

    self._testTM(testProximalInput, activeExternalCellsSequence=externalInputA[1:],
                 activeApicalCellsSequence=None)
    # many extra predictions
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertGreaterEqual(len(predictedInactiveColumns[-1]), min(self.w))


  def testO2(self):
    """Joint external / apical inputs, with incorrect feedback.

    Train on ABCDE / PQRST with F1, XBCDEF / PQRST with F2.
    Test with BCDE / QRST. With feedback F2, burst.
    """
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
    # burst at last step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertGreaterEqual(len(unpredictedActiveColumns[-1]), min(self.w))


  def testO3(self):
    """Joint external / apical inputs, with incorrect external input.

    Train on ABCDE / PQRST with F1, XBCDEF / PQRST with F2.
    Test with BCDE / QRXT. With feedback F1, burst
    """
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
    # burst at last step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertGreaterEqual(len(unpredictedActiveColumns[-1]), min(self.w))


  def testO4(self):
    """Joint external / apical inputs, with correct feedback.

    Train on ABCDE / PQRST with F1, XBCDEF / PQRST with F2.
    Test with BCDE / QRST. With feedback F1, no burst nor extra (feedback disambiguation).
    """
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
    # few extra predictions
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertLessEqual(len(predictedInactiveColumns[-1]), 3)


  def testO5(self):
    """Same as E1, with slight spatial noise on proximal and external input."""
    self.init({"cellsPerColumn": 1})

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, activeExternalCellsSequence=externalInputA,
                  formInternalConnections=True)

    proximalInputA = self.sequenceMachine.addSpatialNoise(proximalInputA, 0.05)
    externalInputA = self.sequenceMachine.addSpatialNoise(externalInputA, 0.05)

    self._testTM(proximalInputA, activeExternalCellsSequence=externalInputA)
    self.assertAlmostAllActiveWerePredicted()
    self.assertAlmostAllInactiveWereUnpredicted()


  def testO6(self):
    """Same as E8, with slight spatial noise.

    Does not pass as E8 alone does not pass.
    """
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

    proximalInputA = self.sequenceMachine.addSpatialNoise(proximalInputA, 0.05)
    externalInputA = self.sequenceMachine.addSpatialNoise(externalInputA, 0.05)

    self._testTM(proximalInputA, activeExternalCellsSequence=externalInputA)
    self.assertAlmostAllActiveWerePredicted()
    self.assertAlmostAllInactiveWereUnpredicted()


  def testO7(self):
    """Same as A2, with slight spatial noise on proximal input and feedback.

    Does not pass as A2 alone does not pass.
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

    testProximalInput = self.sequenceMachine.addSpatialNoise(testProximalInput, 0.05)
    feedbackA = self.sequenceMachine.addSpatialNoise(feedbackA, 0.05)

    self._testTM(testProximalInput, activeApicalCellsSequence=feedbackA)
    self.assertAlmostAllActiveWerePredicted()
    self.assertAlmostAllInactiveWereUnpredicted()


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
      "minThreshold": 25,
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


  def assertAlmostAllActiveWerePredicted(self, meanThr=1):
    unpredictedActiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTraceUnpredictedActiveColumns())
    predictedActiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTracePredictedActiveColumns())

    self.assertLessEqual(unpredictedActiveColumnsMetric.mean, meanThr)

    self.assertGreaterEqual(predictedActiveColumnsMetric.mean, min(self.w))
    self.assertLessEqual(predictedActiveColumnsMetric.mean, max(self.w))


  def assertAllInactiveWereUnpredicted(self):
    predictedInactiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTracePredictedInactiveColumns())

    self.assertEqual(predictedInactiveColumnsMetric.sum, 0)


  def assertAlmostAllInactiveWereUnpredicted(self, meanThr=1):
    predictedInactiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTracePredictedInactiveColumns())

    self.assertLessEqual(predictedInactiveColumnsMetric.mean, meanThr)


  def assertAllActiveWereUnpredicted(self):
    unpredictedActiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTraceUnpredictedActiveColumns())
    predictedActiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTracePredictedActiveColumns())

    self.assertEqual(predictedActiveColumnsMetric.sum, 0)

    self.assertEqual(unpredictedActiveColumnsMetric.min, min(self.w))
    self.assertEqual(unpredictedActiveColumnsMetric.max, max(self.w))


  def assertAlmostAllActiveWereUnpredicted(self, meanThr=1):
    unpredictedActiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTraceUnpredictedActiveColumns())
    predictedActiveColumnsMetric = self.tm.mmGetMetricFromTrace(
      self.tm.mmGetTracePredictedActiveColumns())

    self.assertLess(predictedActiveColumnsMetric.mean, meanThr)

    self.assertGreaterEqual(unpredictedActiveColumnsMetric.mean, min(self.w))
    self.assertLessEqual(unpredictedActiveColumnsMetric.mean, max(self.w))
