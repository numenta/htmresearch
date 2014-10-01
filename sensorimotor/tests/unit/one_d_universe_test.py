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

from sensorimotor.one_d_universe import OneDUniverse



class OneDUniverseTest(unittest.TestCase):


  def testEncodeSensorValue(self):
    universe = OneDUniverse(debugSensor=True,
                            nSensor=105, wSensor=5, nMotor=105, wMotor=5)
    self.assertEqual(universe.encodeSensorValue(0), set(xrange(0, 5)))
    self.assertEqual(universe.encodeSensorValue(19), set(xrange(95, 100)))
    self.assertEqual(universe.encodeSensorValue(20), set(xrange(100, 105)))

    universe = OneDUniverse(debugSensor=False,
                            nSensor=105, wSensor=5, nMotor=105, wMotor=5)
    self.assertNotEqual(universe.encodeSensorValue(0), set(xrange(0, 5)))


  def testEncodeMotorValue(self):
    universe = OneDUniverse(debugMotor=True,
                            nSensor=105, wSensor=5, nMotor=48*21, wMotor=48)
    self.assertEqual(universe.encodeMotorValue(-10), set(xrange(0, 48)))
    self.assertEqual(universe.encodeMotorValue(0), set(xrange(480, 528)))
    self.assertEqual(universe.encodeMotorValue(10), set(xrange(960, 1008)))

    universe = OneDUniverse(debugMotor=False,
                            nSensor=105, wSensor=5, nMotor=48*21, wMotor=48)
    self.assertNotEqual(universe.encodeMotorValue(-10), set(xrange(0, 48)))


  def testDecodeSensorValue(self):
    universe = OneDUniverse()

    for i in range(universe.numDecodedElements):
      self.assertIsNotNone(universe.decodeSensorValue(i))

    self.assertRaises(IndexError,
                      universe.decodeSensorValue, universe.numDecodedElements)



if __name__ == "__main__":
  unittest.main()
