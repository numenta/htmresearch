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

from nupic.data.pattern_machine import ConsecutivePatternMachine

from sensorimotor.one_d_world import OneDWorld
from sensorimotor.one_d_universe import OneDUniverse
from sensorimotor.random_one_d_agent import RandomOneDAgent

from abstract_sensorimotor_test import AbstractSensorimotorTest



class SensorimotorTemporalMemoryTest(AbstractSensorimotorTest):

  DEFAULT_TM_PARAMS = {
    "columnDimensions": [140],
    "cellsPerColumn": 8,
    "initialPermanence": 0.5,
    "connectedPermanence": 0.6,
    "minThreshold": 10,
    "maxNewSynapseCount": 50,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "activationThreshold": 10
  }


  def testSingleWorld(self):
    """Test Sensorimotor Temporal Memory learning in a single world"""
    self._init()

    patternMachine = ConsecutivePatternMachine(35, 7)
    universe = OneDUniverse(3, patternMachine,
                            nSensor=175, wSensor=7,
                            nMotor=49, wMotor=7)
    world = OneDWorld(universe, [0, 1, 2, 3], 2)
    agent = RandomOneDAgent(possibleMotorValues=set(xrange(-3, 3)))

    (sensorSequence, motorSequence) = self._generateSensorimotorSequence(
      70, world, agent)

    self.feedTM(sensorSequence, motorSequence)

    (sensorSequence, motorSequence) = self._generateSensorimotorSequence(
      20, world, agent)

    self._testTM(sensorSequence, motorSequence)



if __name__ == "__main__":
  unittest.main()

