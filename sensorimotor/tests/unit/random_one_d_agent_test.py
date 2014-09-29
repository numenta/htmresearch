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


  def testMotion(self):
    universe = OneDUniverse(debugSensor=True, debugMotor=True,
                            nSensor=100, wSensor=5,
                            nMotor=100, wMotor=20)
    world = OneDWorld(universe, [2, 0, 5, 15, 10])
    agent = RandomOneDAgent(world, 2)

    self.assertEqual(set(xrange(25, 30)), agent.sense())

    self.assertEqual(agent.move(1), set(xrange(60, 80)))
    self.assertEqual(set(xrange(75, 80)), agent.sense())

    self.assertEqual(agent.move(-2), set(xrange(0, 20)))
    self.assertEqual(set(xrange(0, 5)), agent.sense())

    self.assertEqual(agent.move(0), set(xrange(40, 60)))
    self.assertEqual(set(xrange(0, 5)), agent.sense())


  def testDistanceToBoundaries(self):
    universe = OneDUniverse(debugSensor=True, debugMotor=True,
                            nSensor=100, wSensor=5,
                            nMotor=25, wMotor=5)
    world = OneDWorld(universe, [2, 0, 5, 15, 10])
    agent = RandomOneDAgent(world, 2)
    self.assertEqual(agent.distanceToBoundaries(), (2, 2))

    agent.move(-2)
    self.assertEqual(agent.distanceToBoundaries(), (0, 4))

    agent.move(2)
    agent.move(2)
    self.assertEqual(agent.distanceToBoundaries(), (4, 0))


  def testChooseMotorValue(self):
    universe = OneDUniverse(nSensor=100, wSensor=5,
                            nMotor=105, wMotor=5)
    world = OneDWorld(universe, [2, 0, 5, 15, 10])
    agent = RandomOneDAgent(world, 2, possibleMotorValues=set(xrange(-10, 10)))

    for _ in range(100):
      motorValue = agent.chooseMotorValue()
      self.assertTrue(-2 <= motorValue <= 2)  # bounded by size of world

    agent.move(-2)

    for _ in range(100):
      motorValue = agent.chooseMotorValue()
      self.assertTrue(0 <= motorValue <= 4)  # bounded by size of world


  def testGenerateSensorimotorSequence(self):
    universe = OneDUniverse(nSensor=100, wSensor=5,
                            nMotor=105, wMotor=5)
    world = OneDWorld(universe, [2, 0, 5, 15, 10])
    agent = RandomOneDAgent(world, 2, possibleMotorValues=set(xrange(-10, 10)))

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
