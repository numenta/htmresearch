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
from sensorimotor.exhaustive_one_d_agent import ExhaustiveOneDAgent



class ExhaustiveOneDAgentTest(unittest.TestCase):


  def testChooseMotorValue(self):
    numElements = 5
    universe = OneDUniverse(nSensor=100, wSensor=5,
                            nMotor=100, wMotor=5)
    world = OneDWorld(universe, range(numElements))
    agent = ExhaustiveOneDAgent(world, 0)

    motorValues = []

    for _ in range(numElements ** 2):
      motorValue = agent.chooseMotorValue()
      agent.move(motorValue)
      motorValues.append(motorValue)

    self.assertEqual(motorValues,
       [1, -1, 2, -2, 3, -3, 4, -4, 1, 1, -1, 2, -2, 3, -3, 1, 1, -1, 2, -2, 1, 1, -1, 1, -4])


  def testChooseMotorValueStartAt2(self):
    numElements = 5
    universe = OneDUniverse(nSensor=100, wSensor=5,
                            nMotor=100, wMotor=5)
    world = OneDWorld(universe, range(numElements))
    agent = ExhaustiveOneDAgent(world, 2)

    motorValues = []

    for _ in range(numElements ** 2):
      motorValue = agent.chooseMotorValue()
      agent.move(motorValue)
      motorValues.append(motorValue)

    self.assertEqual(motorValues,
       [-2, 2, -1, 1, 1, -1, 2, -2, -2, 1, -1, 3, -3, 4, -4, 1, 2, -2, 3, -3, 2, 1, -1, 1, -2])


if __name__ == "__main__":
  unittest.main()
