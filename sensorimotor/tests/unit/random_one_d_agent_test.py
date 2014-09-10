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



class RandomOneDAgentTest(unittest.TestCase):


  def testChooseMotorValue(self):
    patternMachine = ConsecutivePatternMachine(100, 5)
    universe = OneDUniverse(10, patternMachine,
                            nSensor=100, wSensor=5,
                            nMotor=100, wMotor=5)
    world = OneDWorld(universe, [2, 0, 5, 15, 10], 2)
    agent = RandomOneDAgent(possibleMotorValues=set(xrange(-10, 10)))

    for _ in range(100):
      motorValue = agent.chooseMotorValue(world)
      self.assertTrue(-2 <= motorValue <= 2)  # bounded by size of world



if __name__ == "__main__":
  unittest.main()
