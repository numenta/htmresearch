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



class RandomOneDAgentTest(unittest.TestCase):


  def testChooseMotorValue(self):
    universe = OneDUniverse(nSensor=100, wSensor=5,
                            nMotor=105, wMotor=5)
    world = OneDWorld(universe, [2, 0, 5, 15, 10], 2)
    agent = RandomOneDAgent(world, possibleMotorValues=set(xrange(-10, 10)))

    for _ in range(100):
      motorValue = agent.chooseMotorValue()
      self.assertTrue(-2 <= motorValue <= 2)  # bounded by size of world

    world.move(-2)

    for _ in range(100):
      motorValue = agent.chooseMotorValue()
      self.assertTrue(0 <= motorValue <= 4)  # bounded by size of world


  def testGenerateSensorimotorSequence(self):
    universe = OneDUniverse(nSensor=100, wSensor=5,
                            nMotor=105, wMotor=5)
    world = OneDWorld(universe, [2, 0, 5, 15, 10], 2)
    agent = RandomOneDAgent(world, possibleMotorValues=set(xrange(-10, 10)))

    sensorSequence, motorSequence, sensorimotorSequence = (
      agent.generateSensorimotorSequence(20)
    )

    self.assertEqual(len(sensorSequence), 20)
    self.assertEqual(len(motorSequence), 20)
    self.assertEqual(len(sensorimotorSequence), 20)

    # Ensure each encoded pattern has the correct number of ON bits
    for i in range(20):
      self.assertEqual(len(sensorSequence[i]), 5)
      self.assertEqual(len(motorSequence[i]), 5)
      self.assertEqual(len(sensorimotorSequence[i]), 10)

if __name__ == "__main__":
  unittest.main()
