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
import random

from htmresearch_core.experimental import ExtendedTemporalMemory
from nupic.data.generators.pattern_machine import PatternMachine
from nupic.data.generators.sequence_machine import SequenceMachine

from htmresearch.support.etm_monitor_mixin import (
  ExtendedTemporalMemoryMonitorMixin)


class MonitoredTemporalMemory(ExtendedTemporalMemoryMonitorMixin,
                              ExtendedTemporalMemory):
  pass


class ExtendedTemporalMemoryAlgorithmTest(unittest.TestCase):
  """
  Tests specific aspects of extended temporal memory:

  - External basal connections
  - Combining internal and external basal connections to handle motor
    transitions between features
  - learnOnOneCell
  - Combining internal and external basal connections, and also apical
    connections, *and* learnOnOneCell, all in one test.

  ============================================================================
                        Learning with external input
  ============================================================================

  These tests (labeled "E") simulate sequences from proximal input as well as
  sequences fed through distal dendrites par external neurons (e.g. motor
  command copy).

  E1-E5: First-order learning through proximal and external input.

  E6-E10: Higher-order learning through proximal and external input.

  E11: Repeated motor command copy.

  E12-E15: Testing the "learnOnOneCell" feature, fixing the winning cell for
  each columns between subsequent resets.

  E16-E17: "learnOnOneCell" with motor command should not impair behavior.

  E18-E19: Ambiguous sequences should predict union of representations.

  ============================================================================
                        Learning with apical input
  ============================================================================

  These tests (labeled "A", simulate sequences from proximal input as well as
  feedback from higher regions connected to apical dendrites. It tests some
  basic properties that feedback should achieve in a real setting (sequence
  disambiguation, robustness to temporal noise).

  A5-A9: Robustness to temporal noise through feedback from higher regions.

  ============================================================================
                             Other tests
  ============================================================================

  O1-O4: Joint learning from proximal/external input with feedback.

  O5-O7: Tests drawn from the previous categories, adding slight spatial noise
  to input patterns at test time.

  O8-O10: Ambiguous feedback should lead to ambiguous predictions.
  """
  VERBOSITY = 1
  n = 2048
  w = range(38, 43)
  feedback_size = 400


  def testE1(self):
    """Joint proximal and external first order learning, correct inputs at
    test time.

    Train on ABCDE/PQRST, test on same sequences should yield perfect
    predictions.
    """
    self.init({"cellsPerColumn": 1})

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)

    self._testTM(proximalInputA, externalBasalSequence=externalInputA)

    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


  def testE2(self):
    """Joint proximal and external first order learning, no external input at
    test time.

    Train on ABCDE/PQRST, test on ABCDE/None leads to bursting.
    """
    self.init({"cellsPerColumn": 1})

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)

    self._testTM(proximalInputA, externalBasalSequence=None)
    self.assertAllActiveWereUnpredicted()


  def testE3(self):
    """Joint proximal and external first order learning, incorrect external
    prediction at test time.

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
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)

    self._testTM(proximalInputA, externalBasalSequence=externalInputB)
    self.assertAlmostAllActiveWereUnpredicted()


  def testE4(self):
    """Like E1, with slower learning."""
    self.init({"cellsPerColumn": 1,
               "initialPermanence": 0.5,
               "connectedPermanence": 0.6,
               "permanenceIncrement": 0.05})

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(4):
      # feed sequence multiple times to compensate for slower learning rate
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)

    self._testTM(proximalInputA, externalBasalSequence=externalInputA)
    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


  def testE5(self):
    """Joint proximal and external first order learning, incorrect proximal
    input at test time.

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
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)

    self._testTM(proximalInputB, externalBasalSequence=externalInputA)
    self.assertAlmostAllActiveWereUnpredicted()


  def testE6(self):
    """Joint proximal and external variable order learning.

    Same as E1, with 32 cells per column.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)

    self._testTM(proximalInputA, externalBasalSequence=externalInputA)
    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


  def testE7(self):
    """Joint proximal and external variable order learning, possible ambiguity
    on proximal.

    Train on ABCDE/PQRST and ABCDF/PQRST, test on ABCDE/PQRST leads to extra
    predictions.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 20)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[-2] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 20)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(10):
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)
      self.feedTM(proximalInputB, externalBasalSequence=externalInputA)

    self._testTM(proximalInputA, externalBasalSequence=externalInputA)
    self.assertAllActiveWerePredicted()
    # many extra predictions
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertGreaterEqual(len(predictedInactiveColumns[-1]), min(self.w))


  def testE8(self):
    """Joint proximal and external variable order learning, possible ambiguity
    on proximal.

    Train on ABCDE/PQRST and XBCDF/PQRST, test on ABCDE/PQRST does not lead to
    extra predictions.

    Using assertAlmostAllActiveWerePredicted as minimum # of active bits is
    not min(self.w) with this seed. In practice, this will be constant.
    Using a short sequence, as it needs many iterations to learn the shared
    subsequence.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[0] = max(numbers) + 10
    numbers[-2] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(20):
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)
      self.feedTM(proximalInputB, externalBasalSequence=externalInputA)

    self._testTM(proximalInputA, externalBasalSequence=externalInputA)

    self.assertAlmostAllActiveWerePredicted(meanThr=0)
    self.assertAlmostAllInactiveWereUnpredicted(meanThr=0)


  def testE9(self):
    """Joint proximal and external variable order learning, possible ambiguity
    on external.

    Train on ABCDE/PQRST and ABCDE/XQRZT, test on ABCDE/PQRST does not lead to
    extra predictions.

    Using a short sequence, as it needs many iterations to learn the shared
    subsequence.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[0] = max(numbers) + 1
    numbers[-3] = max(numbers) + 1
    externalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(10):
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)
      self.feedTM(proximalInputA, externalBasalSequence=externalInputB)

    self._testTM(proximalInputA, externalBasalSequence=externalInputA)
    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


  def testE10(self):
    """Same as E8, but the last pattern is incorrect given the high order the
    sequence.

    Using a short sequence, as it needs many iterations to learn the shared
    subsequence.
    """
    self.init()

    # A B C D E
    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    # X B C D E
    numbers[0] = max(numbers) + 1
    testProximalInput = self.sequenceMachine.generateFromNumbers(numbers)

    # X B C D F
    numbers[-2] = max(numbers) + 1
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(10):
      # train on A B C D E and X B C D F
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)
      self.feedTM(proximalInputB, externalBasalSequence=externalInputA)

    # test on X B C D E
    self._testTM(testProximalInput,
                 externalBasalSequence=externalInputA)

    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertGreaterEqual(unpredictedActiveColumns, min(self.w))


  def testE11(self):
    """Repeated motor command copy as external input.

    This tests is straightforward but simulates a realistic setting where the
    external input is the copy of motor command which is a repeated sequence
    of the form XYXYXYXY.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 100)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = [100, 200] * 50 + [None]
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)

    self._testTM(proximalInputA, externalBasalSequence=externalInputA)
    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


  def testE12(self):
    """Simple learnOnOneCell test.

    Train on ABCADC. Without learnOnOneCell, C should have different
    representations in ABC and ADC.
    """
    self.init({"learnOnOneCell": False})
    self.assertFalse(self.tm.getLearnOnOneCell())

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    numbers[0] = 50
    numbers[-2] = 75
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    numbers[0] = 50
    numbers[-2] = 75
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    trainInput = proximalInputA[:-1] + proximalInputB

    for _ in xrange(2):
      self.feedTM(trainInput)

    self._testTM(proximalInputA)
    predictedActiveA = self.tm.mmGetTracePredictedActiveCells().data[-1]
    self._testTM(proximalInputB)
    predictedActiveB = self.tm.mmGetTracePredictedActiveCells().data[-1]
    # check that the two representations are the same
    self.assertNotEquals(predictedActiveA, predictedActiveB)


  def testE13(self):
    """Simple learnOnOneCell test.

    Train on ABCADC, check that C has the same representation in ADC and ABC.
    """
    self.init({"learnOnOneCell": True})
    self.assertTrue(self.tm.getLearnOnOneCell())

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    numbers[0] = 50
    numbers[-2] = 75
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    numbers[0] = 50
    numbers[-2] = 75
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    trainInput = proximalInputA[:-1] + proximalInputB

    for _ in xrange(2):
      self.feedTM(trainInput)

    self._testTM(proximalInputA)
    predictedActiveA = self.tm.mmGetTracePredictedActiveCells().data[-1]
    self._testTM(proximalInputB)
    predictedActiveB = self.tm.mmGetTracePredictedActiveCells().data[-1]
    # check that the two representations are the same
    self.assertEqual(predictedActiveA, predictedActiveB)

  def testE14(self):
    """Simple learnOnOneCell test.

    Train on ABCADC / XYZFGH. Without learnOnOneCell, C should have different
    representations in ABC and ADC.
    """
    self.init({"learnOnOneCell": False})
    self.assertFalse(self.tm.getLearnOnOneCell())

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    numbers[0] = 50
    numbers[-2] = 75
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    numbers[0] = 50
    numbers[-2] = 75
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    trainInput = proximalInputA[:-1] + proximalInputB
    externalTrainInput = externalInputA[:-1] + externalInputB

    for _ in xrange(2):
      self.feedTM(trainInput, externalBasalSequence=externalTrainInput)

    self._testTM(proximalInputA, externalBasalSequence=externalInputA)
    predictedActiveA = self.tm.mmGetTracePredictedActiveCells().data[-1]
    self._testTM(proximalInputB, externalBasalSequence=externalInputB)
    predictedActiveB = self.tm.mmGetTracePredictedActiveCells().data[-1]
    # check that the two representations are the same
    self.assertNotEquals(predictedActiveA, predictedActiveB)


  def testE15(self):
    """Simple learnOnOneCell test.

    Train on ABCADC / XYZFGH, check that C has the same representation in
    ADC and ABC.
    """
    self.init({"learnOnOneCell": True})
    self.assertTrue(self.tm.getLearnOnOneCell())

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    numbers[0] = 50
    numbers[-2] = 75
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    numbers[0] = 50
    numbers[-2] = 75
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    trainInput = proximalInputA[:-1] + proximalInputB
    externalTrainInput = externalInputA[:-1] + externalInputB

    for _ in xrange(2):
      self.feedTM(trainInput, externalBasalSequence=externalTrainInput)

    self._testTM(proximalInputA, externalBasalSequence=externalInputA)
    predictedActiveA = self.tm.mmGetTracePredictedActiveCells().data[-1]
    self._testTM(proximalInputB, externalBasalSequence=externalInputB)
    predictedActiveB = self.tm.mmGetTracePredictedActiveCells().data[-1]
    # check that the two representations are the same
    self.assertEqual(predictedActiveA, predictedActiveB)


  def testE16(self):
    """learnOnOneCell with motor command should not impair behavior.

    Using learnOnOneCell, does the same test as E6, using the same parameters.
    This test is slightly less stringent as shared columns between patterns
    can cause some bursting.
    """
    self.init({"learnOnOneCell": True})
    self.assertTrue(self.tm.getLearnOnOneCell())

    numbers = self.sequenceMachine.generateNumbers(1, 50)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 50)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(2):
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)

    self._testTM(proximalInputA, externalBasalSequence=externalInputA)

    self.assertAlmostAllActiveWerePredicted()
    self.assertAlmostAllInactiveWereUnpredicted()


  def testE17(self):
    """learnOnOneCell with motor command should not impair behavior.

    Same as E16, we need (#cellsPerColumn + 1) passes to learn all collisions
    between patterns.
    """
    self.init({"learnOnOneCell": True, "cellsPerColumn": 8})
    self.assertTrue(self.tm.getLearnOnOneCell())

    numbers = self.sequenceMachine.generateNumbers(1, 50)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 50)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(self.tm.getCellsPerColumn() + 1):
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)

    self._testTM(proximalInputA, externalBasalSequence=externalInputA)

    self.assertAllActiveWerePredicted()
    self.assertAllInactiveWereUnpredicted()


  def testE18(self):
    """Ambiguous sequences should predict union of representations.

    Train on ABCDE / PQRST and XB'C'D'F / PQRST.
    Test with BCDE. prediction at third step should be the union of C and C'.
    """
    self.init()

    # A B C D E
    numbers = self.sequenceMachine.generateNumbers(1, 5)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    # X B C D F
    numbers[0] = max(numbers) + 10
    numbers[-2] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    # P Q R S T
    numbers = self.sequenceMachine.generateNumbers(1, 5)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(10):
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)
      self.feedTM(proximalInputB, externalBasalSequence=externalInputA)

    self.feedTM(proximalInputA, externalBasalSequence=externalInputA)
    predictedC = self.tm.mmGetTracePredictedActiveCells().data[2]
    self.feedTM(proximalInputB, externalBasalSequence=externalInputA)
    predictedCPrime = self.tm.mmGetTracePredictedActiveCells().data[2]

    self._testTM(proximalInputA[1:],
                 externalBasalSequence=externalInputA[1:])
    # looking at predictive cells from previous step
    predictedUnion = self.tm.mmGetTracePredictedCells().data[1]
    self.assertNotEquals(predictedC, predictedCPrime)
    self.assertEqual(predictedUnion, predictedC | predictedCPrime)


  def testE19(self):
    """Ambiguous sequences should predict union of representations.

    Train on ABCDE / PQRST and XB'C'D'F / PQRST.
    Test with BCDE. prediction at last step should be the union of E and F.
    """
    self.init()

    # A B C D E
    numbers = self.sequenceMachine.generateNumbers(1, 5)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    # X B C D F
    numbers[0] = max(numbers) + 10
    numbers[-2] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    # P Q R S T
    numbers = self.sequenceMachine.generateNumbers(1, 5)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(10):
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)
      self.feedTM(proximalInputB, externalBasalSequence=externalInputA)

    self.feedTM(proximalInputA, externalBasalSequence=externalInputA)
    predictedE = self.tm.mmGetTracePredictedActiveCells().data[-1]
    self.feedTM(proximalInputB, externalBasalSequence=externalInputA)
    predictedF = self.tm.mmGetTracePredictedActiveCells().data[-1]

    self._testTM(proximalInputA[1:],
                 externalBasalSequence=externalInputA[1:])
    # looking at predictive cells from previous step
    predictedUnion = self.tm.mmGetTracePredictedCells().data[-1]
    self.assertNotEquals(predictedE, predictedF)
    self.assertEqual(predictedUnion, predictedE | predictedF)


  def testA5(self):
    """Robustness to temporal noise with feedback, test without feedback.

    Train on ABCDE with F1, XBCDF with F2.
    Test with ACDF (one step is missing). Without feedback, both E and F are
    predicted.
    """
    self.init()

    # A B C D E
    numbers = self.sequenceMachine.generateNumbers(1, 5)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    # A C D F
    numbers[-2] = max(numbers) + 10
    noisyNumbers = numbers[:1] + numbers[2:]
    testProximalInput = self.sequenceMachine.generateFromNumbers(noisyNumbers)

    # X B C D F
    numbers[0] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(10):
      self.feedTM(proximalInputA, externalApicalSequence=None)
      self.feedTM(proximalInputB, externalApicalSequence=None)

    self._testTM(testProximalInput, externalApicalSequence=None)
    # no bursting at last step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertEqual(len(unpredictedActiveColumns[-1]), 0)
    # many extra predictions
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertGreaterEqual(len(predictedInactiveColumns[-1]), min(self.w))


  def testA6(self):
    """Robustness to temporal noise with feedback, test with incorrect
    feedback.

    Train on ABCDE with F1, XBCDF with F2.
    Test with ACDF (one step is missing). With feedback F1, burst at last
    timestep.
    """
    self.init()

    # A B C D E
    numbers = self.sequenceMachine.generateNumbers(1, 5)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    # A C D F
    numbers[-2] = max(numbers) + 10
    noisyNumbers = numbers[:1] + numbers[2:]
    testProximalInput = self.sequenceMachine.generateFromNumbers(noisyNumbers)

    # X B C D F
    numbers[0] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    feedbackA = self.generateFeedbackSequence(len(numbers))
    feedbackB = self.generateFeedbackSequence(len(numbers))

    # start training with random feedback
    self.feedTM(proximalInputA,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))
    self.feedTM(proximalInputB,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))

    for _ in xrange(10):
      self.feedTM(proximalInputA, externalApicalSequence=feedbackA)
      self.feedTM(proximalInputB, externalApicalSequence=feedbackB)

    print "Testing"
    self._testTM(testProximalInput, externalApicalSequence=feedbackA)
    # burst at last step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertGreaterEqual(len(unpredictedActiveColumns[-1]), min(self.w))


  def testA7(self):
    """Robustness to temporal noise with feedback, test without feedback.

    Train on ABCDE with F1, XBCDF with F2.
    Test with AZCDF (one step is corrupted). Without feedback, both E and F
    are predicted.
    """
    self.init()

    # A B C D E
    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    # A Z C D F
    numbers[-2] = max(numbers) + 10
    noisyNumbers = numbers[:1] + [50] + numbers[2:]
    testProximalInput = self.sequenceMachine.generateFromNumbers(noisyNumbers)

    # X B C D F
    numbers[0] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(10):
      self.feedTM(proximalInputA, externalApicalSequence=None)
      self.feedTM(proximalInputB, externalApicalSequence=None)

    self._testTM(testProximalInput, externalApicalSequence=None)
    # no bursting at last step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertEqual(len(unpredictedActiveColumns[-1]), 0)
    # many extra predictions
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertGreaterEqual(len(predictedInactiveColumns[-1]), min(self.w))


  def testA8(self):
    """Robustness to temporal noise with feedback, test with incorrect feedback.

    Train on ABCDE with F1, XBCDF with F2.
    Test with AZCDF (one step is corrupted). With feedback F1, burst at last
    timestep.
    """
    self.init()

    # A B C D E
    numbers = self.sequenceMachine.generateNumbers(1, 5)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    # A Z C D F
    numbers[-2] = max(numbers) + 10
    noisyNumbers = numbers[:1] + [50] + numbers[2:]
    testProximalInput = self.sequenceMachine.generateFromNumbers(noisyNumbers)

    # X B C D F
    numbers[0] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    feedbackA = self.generateFeedbackSequence(len(numbers))
    feedbackB = self.generateFeedbackSequence(len(numbers))

    # start training with random feedback
    self.feedTM(proximalInputA,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))
    self.feedTM(proximalInputB,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))

    for _ in xrange(10):
      self.feedTM(proximalInputA, externalApicalSequence=feedbackA)
      self.feedTM(proximalInputB, externalApicalSequence=feedbackB)

    self._testTM(testProximalInput, externalApicalSequence=feedbackA)
    # burst at last step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertGreaterEqual(len(unpredictedActiveColumns[-1]), min(self.w))

  @unittest.skip("This no longer passes now that we fixed a bug where we would \
                  learn on non-matching apical segments.")
  def testA9(self):
    """Robustness to temporal noise with feedback, test with correct feedback.

    Train on ABCDE with F1, XBCDF with F2.
    Test with XZCDF (one step is corrupted). With feedback F2, no bursting,
    almost no extra prediction.
    """
    self.init()

    # A B C D E
    numbers = self.sequenceMachine.generateNumbers(1, 5)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    # X Z C D F
    numbers[0] = max(numbers) + 10
    numbers[-2] = max(numbers) + 10
    noisyNumbers = numbers[:1] + [50] + numbers[2:]
    testProximalInput = self.sequenceMachine.generateFromNumbers(noisyNumbers)

    # X B C D F
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    feedbackA = self.generateFeedbackSequence(len(numbers))
    feedbackB = self.generateFeedbackSequence(len(numbers))

    # start training with random feedback
    self.feedTM(proximalInputA,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))
    self.feedTM(proximalInputB,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))

    for _ in xrange(10):
      self.feedTM(proximalInputA, externalApicalSequence=feedbackA)
      self.feedTM(proximalInputB, externalApicalSequence=feedbackB)

    self._testTM(testProximalInput, externalApicalSequence=feedbackB)
    # no bursting at last step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertEqual(len(unpredictedActiveColumns[-1]), 0)
    # few extra predictions
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertLessEqual(len(predictedInactiveColumns[-1]), 3)


  def testO1(self):
    """Joint external and apical inputs, without feedback.

    Train on ABCDE / PQRST, XBCDF / PQRST.
    Test with BCDE / QRST. Without feedback, both E and F are expected.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 5)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    numbers[0] = max(numbers) + 10
    numbers[-2] = max(numbers) + 10

    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    testProximalInput = proximalInputA[1:]

    numbers = self.sequenceMachine.generateNumbers(1, 5)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    testExternalInput = externalInputA[1:]

    for _ in xrange(20):
      self.feedTM(proximalInputA, externalBasalSequence=externalInputA)
      self.feedTM(proximalInputB, externalBasalSequence=externalInputA)

    self._testTM(testProximalInput,
                 externalBasalSequence=testExternalInput)
    # no bursting at least step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertEqual(len(unpredictedActiveColumns[-1]), 0)
    # many extra predictions
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertGreaterEqual(len(predictedInactiveColumns[-1]), min(self.w))


  def testO2(self):
    """Joint external and apical inputs, with incorrect feedback.

    Train on ABCDE / PQRST with F1, XBCDF / PQRST with F2.
    Test with BCDE / QRST. With feedback F2, should burst at last step.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 5)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    testProximalInput = proximalInputA[1:]

    numbers[0] = max(numbers) + 10
    numbers[-2] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 5)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    feedbackA = self.generateFeedbackSequence(len(numbers))
    feedbackB = self.generateFeedbackSequence(len(numbers))

    # start training with random feedback
    self.feedTM(proximalInputA,
                externalBasalSequence=externalInputA,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))
    self.feedTM(proximalInputB,
                externalBasalSequence=externalInputA,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))

    for _ in xrange(20):
      self.feedTM(proximalInputA,
                  externalBasalSequence=externalInputA,
                  externalApicalSequence=feedbackA)
      self.feedTM(proximalInputB,
                  externalBasalSequence=externalInputA,
                  externalApicalSequence=feedbackB)

    self._testTM(testProximalInput,
                 externalBasalSequence=externalInputA[1:],
                 externalApicalSequence=feedbackB)
    # no burst at last step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertGreaterEqual(len(unpredictedActiveColumns[-1]), min(self.w))
    # many extra predictions
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertGreaterEqual(len(predictedInactiveColumns[-1]), min(self.w))


  def testO3(self):
    """Joint external and apical inputs, with incorrect external prediction.

    Train on ABCDE / PQRST with F1, XBCDF / PQRST with F2.
    Test with BCDE / QRXT. With feedback F1, burst at last step, as prediction
    from external input is incorrect.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 5)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    testProximalInput = proximalInputA[1:]

    numbers[0] = max(numbers) + 10
    numbers[-2] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 5)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[-3] = max(numbers) + 10
    testExternalInput = self.sequenceMachine.generateFromNumbers(numbers)[1:]

    feedbackA = self.generateFeedbackSequence(len(numbers))
    feedbackB = self.generateFeedbackSequence(len(numbers))

    # start training with random feedback
    self.feedTM(proximalInputA,
                externalBasalSequence=externalInputA,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))
    self.feedTM(proximalInputB,
                externalBasalSequence=externalInputA,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))

    for _ in xrange(30):
      self.feedTM(proximalInputA,
                  externalBasalSequence=externalInputA,
                  externalApicalSequence=feedbackA)
      self.feedTM(proximalInputB,
                  externalBasalSequence=externalInputA,
                  externalApicalSequence=feedbackB)

    self._testTM(testProximalInput,
                 externalBasalSequence=testExternalInput,
                 externalApicalSequence=feedbackA)
    # burst at last step
    unpredictedActiveColumns = self.tm.mmGetTraceUnpredictedActiveColumns().data
    self.assertGreaterEqual(len(unpredictedActiveColumns[-1]), min(self.w))


  def testO4(self):
    """Joint external / apical inputs, with correct feedback.

    Train on ABCDE / PQRST with F1, XBCDEF / PQRST with F2.
    Test with BCDE / QRST. With feedback F1, no burst nor extra (feedback
    disambiguation).
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    testProximalInput = proximalInputA[1:]

    numbers[0] = max(numbers) + 10
    numbers[-2] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    feedbackA = self.generateFeedbackSequence(len(numbers))
    feedbackB = self.generateFeedbackSequence(len(numbers))

    # start training with random feedback
    self.feedTM(proximalInputA,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))
    self.feedTM(proximalInputB,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))

    for _ in xrange(20):
      self.feedTM(proximalInputA,
                  externalBasalSequence=externalInputA,
                  externalApicalSequence=feedbackA)
      self.feedTM(proximalInputB,
                  externalBasalSequence=externalInputA,
                  externalApicalSequence=feedbackB)

    self._testTM(testProximalInput,
                 externalBasalSequence=externalInputA[1:],
                 externalApicalSequence=feedbackA)
    self.assertAlmostAllActiveWerePredicted()
    # few extra predictions
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertLessEqual(len(predictedInactiveColumns[-1]), 3)


  def testO5(self):
    """Joint external / apical inputs, with correct feedback, varying external
    input length.

    Train on ABCDE / PQRST with F1, XBCDEF / PQRST with F2.
    Test with BCDE / QRST. With feedback F1, no burst nor extra (feedback
    disambiguation).
    The difference with the previous one is that external input has varying
    number of active neurons.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    testProximalInput = proximalInputA[1:]

    numbers[0] = max(numbers) + 10
    numbers[-2] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    feedbackA = self.generateFeedbackSequence(len(numbers))
    feedbackB = self.generateFeedbackSequence(len(numbers))

    # start training with random feedback
    self.feedTM(proximalInputA,
                externalApicalSequence=self.generateRandomFeedback(
                  len(numbers)))
    self.feedTM(proximalInputB,
                externalApicalSequence=self.generateRandomFeedback(
                  len(numbers)))

    for i in xrange(len(externalInputA)-1):
      numberNewInputs = random.randint(0, 50)
      externalInputA[i] = externalInputA[i] | \
            set([random.randint(0, self.n-1) for _ in xrange(numberNewInputs)])

    for _ in xrange(20):
      self.feedTM(proximalInputA,
                  externalBasalSequence=externalInputA,
                  externalApicalSequence=feedbackA)
      self.feedTM(proximalInputB,
                  externalBasalSequence=externalInputA,
                  externalApicalSequence=feedbackB)

    self._testTM(testProximalInput,
                 externalBasalSequence=externalInputA[1:],
                 externalApicalSequence=feedbackA)
    self.assertAlmostAllActiveWerePredicted()
    # few extra predictions
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertLessEqual(len(predictedInactiveColumns[-1]), 3)


  def testO6(self):
    """Same as E1, with slight spatial noise on proximal and external input.

    Tolerance is an average of one bursting column / timestep.
    """
    self.init({"cellsPerColumn": 1})

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(30):
      self.feedTM(proximalInputA,
                  externalBasalSequence=externalInputA)

    proximalInputA = self.sequenceMachine.addSpatialNoise(proximalInputA, 0.02)
    externalInputA = self.sequenceMachine.addSpatialNoise(externalInputA, 0.02)

    self._testTM(proximalInputA, externalBasalSequence=externalInputA)
    self.assertAlmostAllActiveWerePredicted()
    self.assertAlmostAllInactiveWereUnpredicted()


  def testO7(self):
    """Same as E8, with slight spatial noise.

    Tolerance is an average of one bursting column / timestep.
    """
    self.init()

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)
    numbers[0] = max(numbers) + 10
    numbers[-2] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    numbers = self.sequenceMachine.generateNumbers(1, 10)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    for _ in xrange(30):
      self.feedTM(proximalInputA,
                  externalBasalSequence=externalInputA)
      self.feedTM(proximalInputB,
                  externalBasalSequence=externalInputA)

    proximalInputA = self.sequenceMachine.addSpatialNoise(proximalInputA, 0.02)
    externalInputA = self.sequenceMachine.addSpatialNoise(externalInputA, 0.02)

    self._testTM(proximalInputA,
                 externalBasalSequence=externalInputA)

    self.assertAlmostAllActiveWerePredicted()
    self.assertAlmostAllInactiveWereUnpredicted()


  def testO8(self):
    """Same as A2, with slight spatial noise on proximal input and feedback.

    Tolerance is an average of 1 bursting column / timestep.
    """
    self.init()

    # A B C D E
    numbers = self.sequenceMachine.generateNumbers(1, 10)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    # B C D E
    testProximalInput = proximalInputA[1:]

    # X B C D F
    numbers[0] = max(numbers) + 10
    numbers[-2] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    feedbackA = self.generateFeedbackSequence(len(numbers))
    feedbackB = self.generateFeedbackSequence(len(numbers))

    # start training with random feedback
    self.feedTM(proximalInputA,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))
    self.feedTM(proximalInputB,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))

    for _ in xrange(30):
      self.feedTM(proximalInputA, externalApicalSequence=feedbackA)
      self.feedTM(proximalInputB, externalApicalSequence=feedbackB)

    testProximalInput = self.sequenceMachine.addSpatialNoise(testProximalInput,
                                                             0.02)
    feedbackA = self.sequenceMachine.addSpatialNoise(feedbackA,
                                                     0.02)

    self._testTM(testProximalInput, externalApicalSequence=feedbackA)
    self.assertAlmostAllActiveWerePredicted()
    # few extra predictions
    predictedInactiveColumns = self.tm.mmGetTracePredictedInactiveColumns().data
    self.assertLessEqual(len(predictedInactiveColumns[-1]), 3)


  def testO9(self):
    """Ambiguous feedback leads to ambiguous predictions, with external input.

    Same test as A13, with external input.
    """
    self.init()

    # A B C D E
    numbers = self.sequenceMachine.generateNumbers(1, 5)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    # X B C D F
    numbers[0] = max(numbers) + 10
    numbers[-2] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    # Z B C D G
    numbers[0] = max(numbers) + 10
    numbers[-2] = max(numbers) + 10
    proximalInputC = self.sequenceMachine.generateFromNumbers(numbers)

    # P Q R S T
    numbers = self.sequenceMachine.generateNumbers(1, 5)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    # B C D E
    testProximalInput = proximalInputA[1:]

    feedbackA = self.generateFeedbackSequence(len(numbers))
    feedbackB = self.generateFeedbackSequence(len(numbers))
    feedbackC = self.generateFeedbackSequence(len(numbers))
    unionFeedback = [feedbackA[0] | feedbackB[0]] * 2 + [feedbackA[0]] * 4

    # start training with random feedback
    self.feedTM(proximalInputA,
                externalBasalSequence=externalInputA,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))
    self.feedTM(proximalInputB,
                externalBasalSequence=externalInputA,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))
    self.feedTM(proximalInputC,
                externalBasalSequence=externalInputA,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))

    for _ in xrange(20):
      self.feedTM(proximalInputA,
                  externalBasalSequence=externalInputA,
                  externalApicalSequence=feedbackA)
      self.feedTM(proximalInputB,
                  externalBasalSequence=externalInputA,
                  externalApicalSequence=feedbackB)
      self.feedTM(proximalInputC,
                  externalBasalSequence=externalInputA,
                  externalApicalSequence=feedbackC)

    # retrieving patterns
    self.feedTM(proximalInputA,
                externalBasalSequence=externalInputA,
                externalApicalSequence=feedbackA)
    predictedC = self.tm.mmGetTracePredictedActiveCells().data[2]
    predictedE = self.tm.mmGetTracePredictedActiveCells().data[-1]
    self.feedTM(proximalInputB,
                externalBasalSequence=externalInputA,
                externalApicalSequence=feedbackB)
    predictedCPrime = self.tm.mmGetTracePredictedActiveCells().data[2]
    predictedF = self.tm.mmGetTracePredictedActiveCells().data[-1]
    self.feedTM(proximalInputC,
                externalBasalSequence=externalInputA,
                externalApicalSequence=feedbackC)
    predictedCSecond = self.tm.mmGetTracePredictedActiveCells().data[2]
    predictedG = self.tm.mmGetTracePredictedActiveCells().data[-1]

    self._testTM(testProximalInput,
                 externalBasalSequence=externalInputA[1:],
                 externalApicalSequence=unionFeedback[1:])
    predictedUnion1 = self.tm.mmGetTracePredictedCells().data[1]
    predictedUnion2 = self.tm.mmGetTracePredictedCells().data[-1]
    self.assertNotEquals(predictedC, predictedCPrime, predictedCSecond)
    self.assertEqual(predictedUnion1, predictedC | predictedCPrime)
    self.assertEqual(predictedUnion2, predictedE)


  def testO10(self):
    """Ambiguous feedback leads to ambiguous predictions, with external input.

    Same test as A13, with external input, and using learnOnOneCell.
    """
    self.init({"learnOnOneCell": True})
    self.assertTrue(self.tm.getLearnOnOneCell())

    # A B C D E
    numbers = self.sequenceMachine.generateNumbers(1, 5)
    proximalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    # X B C D F
    numbers[0] = max(numbers) + 10
    numbers[-2] = max(numbers) + 10
    proximalInputB = self.sequenceMachine.generateFromNumbers(numbers)

    # Z B C D G
    numbers[0] = max(numbers) + 10
    numbers[-2] = max(numbers) + 10
    proximalInputC = self.sequenceMachine.generateFromNumbers(numbers)

    # P Q R S T
    numbers = self.sequenceMachine.generateNumbers(1, 5)
    externalInputA = self.sequenceMachine.generateFromNumbers(numbers)

    # B C D E
    testProximalInput = proximalInputA[1:]

    feedbackA = self.generateFeedbackSequence(len(numbers))
    feedbackB = self.generateFeedbackSequence(len(numbers))
    feedbackC = self.generateFeedbackSequence(len(numbers))
    unionFeedback = [feedbackA[0] | feedbackB[0]] * 2 + [feedbackA[0]] * 4

    # start training with random feedback
    self.feedTM(proximalInputA,
                externalBasalSequence=externalInputA,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))
    self.feedTM(proximalInputB,
                externalBasalSequence=externalInputA,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))
    self.feedTM(proximalInputC,
                externalBasalSequence=externalInputA,
                externalApicalSequence=self.generateRandomFeedback(len(numbers)))

    for _ in xrange(20):
      self.feedTM(proximalInputA,
                  externalBasalSequence=externalInputA,
                  externalApicalSequence=feedbackA)
      self.feedTM(proximalInputB,
                  externalBasalSequence=externalInputA,
                  externalApicalSequence=feedbackB)
      self.feedTM(proximalInputC,
                  externalBasalSequence=externalInputA,
                  externalApicalSequence=feedbackC)

    # retrieving patterns
    self.feedTM(proximalInputA,
                externalBasalSequence=externalInputA,
                externalApicalSequence=feedbackA)
    predictedC = self.tm.mmGetTracePredictedActiveCells().data[2]
    predictedE = self.tm.mmGetTracePredictedActiveCells().data[-1]
    self.feedTM(proximalInputB,
                externalBasalSequence=externalInputA,
                externalApicalSequence=feedbackB)
    predictedCPrime = self.tm.mmGetTracePredictedActiveCells().data[2]
    predictedF = self.tm.mmGetTracePredictedActiveCells().data[-1]
    self.feedTM(proximalInputC,
                externalBasalSequence=externalInputA,
                externalApicalSequence=feedbackC)
    predictedCSecond = self.tm.mmGetTracePredictedActiveCells().data[2]
    predictedG = self.tm.mmGetTracePredictedActiveCells().data[-1]

    self._testTM(testProximalInput,
                 externalBasalSequence=externalInputA[1:],
                 externalApicalSequence=unionFeedback[1:])
    predictedUnion1 = self.tm.mmGetTracePredictedCells().data[1]
    predictedUnion2 = self.tm.mmGetTracePredictedCells().data[-1]
    self.assertNotEquals(predictedC, predictedCPrime, predictedCSecond)
    self.assertEqual(predictedUnion1, predictedC | predictedCPrime)
    self.assertEqual(predictedUnion2, predictedE)


  # ==============================
  # Overrides
  # ==============================


  def init(self, overrides=None):
    """
    Initialize Temporal Memory, and other member variables.

    :param overrides: overrides for default Temporal Memory parameters
    """
    params = self._computeTMParams(overrides)

    self.tm = MonitoredTemporalMemory(**params)


  def _computeTMParams(self, overrides):
    params = {
      "columnCount": self.n,
      "basalInputSize": (self.n * 32) + self.n,
      "apicalInputSize": self.n,
      "cellsPerColumn": 32,
      "initialPermanence": 0.5,
      "connectedPermanence": 0.6,
      "minThreshold": 25,
      "sampleSize": 30,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.02,
      "predictedSegmentDecrement": 0.08,
      "activationThreshold": 25,
      "seed": 42,
      "learnOnOneCell": False,
    }
    params.update(overrides or {})
    return params


  def generateFeedback(self):
    """Generates a random feedback pattern."""
    return set([random.randint(0, self.n-1) for _ in range(self.feedback_size)])


  def generateFeedbackSequence(self, size):
    """Generates a random feedback pattern sequence."""
    return [self.generateFeedback()] * size


  def generateRandomFeedback(self, size):
    """Generates a sequence of random feedback for initial training."""
    return [self.generateFeedback() for _ in xrange(size)]


  def setUp(self):
    self.tm = None
    self.patternMachine = PatternMachine(self.n, self.w, num=300)
    self.sequenceMachine = SequenceMachine(self.patternMachine)

    print ("\n"
           "======================================================\n"
           "Test: {0} \n"
           "{1}\n"
           "======================================================\n"
    ).format(self.id(), self.shortDescription())


  def feedTM(self,
             sequence,
             externalBasalSequence=None,
             externalApicalSequence=None,
             learn=True,
             num=1):
    """
    Needs to be implemented to take advantage of ETM's compute's specific
    signature.
    :param sequence:                 (list)     Sequence of SDR's to feed
                                                the ETM
    :param externalBasalSequence:    (list)     Sequence of external inputs
    :param externalApicalSequence:   (list)     Sequence of apical inputs
    :param num:                      (boolean)  Optional parameter to repeat
                                                the input sequences

    Note: sequence, activeExternalCells, and activeApicalCells should have the
    same length. As opposed to in the sequence of input pattern, where None
    resets the TM, having None in the external and apical input sequences
    simply means to such input is received at this instant.
    """
    self.tm.reset()

    # replicate sequences if necessary
    repeatedSequence = sequence * num
    if externalBasalSequence is not None:
      # Insert an empty input at the beginning. These tests expect basal/apical
      # inputs to be applied on a 1 timestep time-delay.
      repeatedBasal = [()] + externalBasalSequence * num
    else:
      # no external basal input
      repeatedBasal = [()] * len(repeatedSequence)

    if externalApicalSequence is not None:
      # Insert an empty input at the beginning. These tests expect basal/apical
      # inputs to be applied on a 1 timestep time-delay.
      repeatedApical = [()] + externalApicalSequence * num
    else:
      # no external apical input
      repeatedApical = [()] * len(repeatedSequence)

    self.tm.mmClearHistory()

    for pattern, basal, apical in zip(repeatedSequence,
                                      repeatedBasal,
                                      repeatedApical):
      print pattern
      if pattern is None:
        self.tm.reset()
      else:
        pattern = sorted(pattern)
        basalInput = (self.tm.getActiveCells().tolist() +
                      [cell + self.tm.numberOfCells()
                       for cell in sorted(basal)])
        basalGrowthCandidates = (self.tm.getWinnerCells().tolist() +
                                 [cell + self.tm.numberOfCells()
                                  for cell in sorted(basal)])
        apicalInput = sorted(apical)

        self.tm.compute(activeColumns=pattern,
                        basalInput=basalInput,
                        apicalInput=apicalInput,
                        basalGrowthCandidates=basalGrowthCandidates,
                        learn=learn)

    if self.VERBOSITY >= 2:
      print self.tm.mmPrettyPrintTraces(
        self.tm.mmGetDefaultTraces(verbosity=self.VERBOSITY - 1)
      )
      print

    if learn and self.VERBOSITY >= 3:
      print self.tm.mmPrettyPrintConnections()


  def _testTM(self,
              sequence,
              externalBasalSequence=None,
              externalApicalSequence=None):
    self.feedTM(sequence,
                externalBasalSequence=externalBasalSequence,
                externalApicalSequence=externalApicalSequence,
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



if __name__ == "__main__":
  unittest.main()
