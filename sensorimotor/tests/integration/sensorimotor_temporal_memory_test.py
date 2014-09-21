#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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

import unittest2 as unittest

from sensorimotor.one_d_world import OneDWorld
from sensorimotor.one_d_universe import OneDUniverse
from sensorimotor.random_one_d_agent import RandomOneDAgent

from sensorimotor.test.abstract_sensorimotor_test import (
  AbstractSensorimotorTest)



class SensorimotorTemporalMemoryTest(AbstractSensorimotorTest):

  VERBOSITY = 1
  DEFAULT_TM_PARAMS = {
    "columnDimensions": [100],
    "cellsPerColumn": 8,
    "initialPermanence": 0.5,
    "connectedPermanence": 0.6,
    "minThreshold": 15,
    "activationThreshold": 15,
    "maxNewSynapseCount": 50,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02
  }


  def testSingleWorldOneBitPerPattern(self):
    """
    Test Sensorimotor Temporal Memory learning in a single world.
    Patterns (sensor and motor) are represented with one active bit per pattern.
    """
    self._init({"columnDimensions": [4],
                "minThreshold": 2,
                "activationThreshold": 2})

    universe = OneDUniverse(debugSensor=True, debugMotor=True,
                            nSensor=4, wSensor=1,
                            nMotor=3, wMotor=1)
    world = OneDWorld(universe, [0, 1, 2, 3], 2)
    agent = RandomOneDAgent(world, possibleMotorValues=set(xrange(-1, 2)))

    sequence = self._generateSensorimotorSequences(100, [agent])

    self._feedTM(sequence)

    sequence = self._generateSensorimotorSequences(20, [agent])

    stats = self._testTM(sequence)
    self._assertAllActiveWerePredicted(stats, universe)
    self._assertAllInactiveWereUnpredicted(stats)
    self._assertSequencesOnePredictedActiveCellPerColumn(stats)


  def testSingleWorldBasic(self):
    """
    Test Sensorimotor Temporal Memory learning in a single world.
    Patterns are represented as complete SDRs. No patterns are repeated.
    Prediction should be perfect.
    """
    self._init()

    universe = OneDUniverse(debugSensor=True, debugMotor=True,
                            nSensor=100, wSensor=10,
                            nMotor=70, wMotor=10)
    world = OneDWorld(universe, [0, 1, 2, 3], 2)
    agent = RandomOneDAgent(world, possibleMotorValues=set(xrange(-3, 4)))

    sequence = self._generateSensorimotorSequences(100, [agent])

    self._feedTM(sequence)

    sequence = self._generateSensorimotorSequences(100, [agent])

    stats = self._testTM(sequence)
    self._assertAllActiveWerePredicted(stats, universe)
    self._assertAllInactiveWereUnpredicted(stats)
    self._assertSequencesOnePredictedActiveCellPerColumn(stats)


  def testMultipleWorldsBasic(self):
    """
    Test Sensorimotor Temporal Memory learning in multiple separate worlds.
    Patterns are represented as complete SDRs. No patterns are repeated.
    Prediction should be perfect.
    """
    self._init()

    universe = OneDUniverse(debugMotor=True,
                            nSensor=100, wSensor=10,
                            nMotor=70, wMotor=10)

    agents = []
    numWorlds = 5
    sequenceLength = 4

    for i in xrange(numWorlds):
      start = i * sequenceLength
      patterns = range(start, start + sequenceLength)
      world = OneDWorld(universe, patterns, 2)
      agent = RandomOneDAgent(world, possibleMotorValues=set(xrange(-3, 4)))
      agents.append(agent)

    sequence = self._generateSensorimotorSequences(150, agents)

    self._feedTM(sequence)

    sequence = self._generateSensorimotorSequences(100, agents)

    stats = self._testTM(sequence)
    self._assertAllActiveWerePredicted(stats, universe)
    self._assertAllInactiveWereUnpredicted(stats)
    self._assertSequencesOnePredictedActiveCellPerColumn(stats)


  def testMultipleWorldsSharedPatterns(self):
    """
    Test Sensorimotor Temporal Memory learning in multiple separate worlds.
    Patterns are represented as complete SDRs. Patterns are shared between
    worlds.
    All active columns should have been predicted.
    """
    self._init()

    universe = OneDUniverse(debugMotor=True,
                            nSensor=100, wSensor=10,
                            nMotor=70, wMotor=10)

    agents = []
    numWorlds = 5

    for i in xrange(numWorlds):
      patterns = range(4)
      self._random.shuffle(patterns)
      world = OneDWorld(universe, patterns, 2)
      agent = RandomOneDAgent(world, possibleMotorValues=set(xrange(-3, 4)))
      agents.append(agent)

    sequence = self._generateSensorimotorSequences(150, agents)

    self._feedTM(sequence)

    sequence = self._generateSensorimotorSequences(100, agents)

    stats = self._testTM(sequence)
    self._assertAllActiveWerePredicted(stats, universe)

    self.assertTrue(0 < stats.predictedInactiveColumns.average < 10)

    # TODO: Assert that patterns in different worlds have different cell
    # representations


  # ==============================
  # Helper functions
  # ==============================

  def _assertAllActiveWerePredicted(self, stats, universe):
    self.assertEqual(stats.unpredictedActiveColumns.sum, 0)

    self.assertEqual(stats.predictedActiveColumns.min, universe.wSensor)
    self.assertEqual(stats.predictedActiveColumns.max, universe.wSensor)


  def _assertAllInactiveWereUnpredicted(self, stats):
    self.assertEqual(stats.predictedInactiveColumns.sum, 0)


  def _assertAllActiveWereUnpredicted(self, stats, universe):
    self.assertEqual(stats.predictedActiveColumns.sum, 0)

    self.assertEqual(stats.unpredictedActiveColumns.min, universe.wSensor)
    self.assertEqual(stats.unpredictedActiveColumns.max, universe.wSensor)


  def _assertSequencesOnePredictedActiveCellPerColumn(self, stats):
    self.assertEqual(stats.sequencesPredictedActiveCellsPerColumn.min, 1)
    self.assertEqual(stats.sequencesPredictedActiveCellsPerColumn.max, 1)


if __name__ == "__main__":
  unittest.main()

