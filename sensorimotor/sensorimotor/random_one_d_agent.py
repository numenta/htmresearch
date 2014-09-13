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

import numpy

from sensorimotor.abstract_agent import AbstractAgent



class RandomOneDAgent(AbstractAgent):


  def __init__(self, world, possibleMotorValues=set([-1, 1]), seed=42):
    """
    @param world (AbstractWorld) The world this agent belongs in.
    @param possibleMotorValues (set) Set of motor values to choose from
    @param seed                (int) Seed for random number generator
    """
    self._world = world
    self._possibleMotorValues = possibleMotorValues
    self._random = numpy.random.RandomState(seed)


  def chooseMotorValue(self):
    """
    Return a plausible next motor value.

    @return (int) motor value
    """
    distanceToLeft, distanceToRight = self._world.distanceToBoundaries()
    minValue = -distanceToLeft
    maxValue = distanceToRight
    candidates = [x for x in self._possibleMotorValues if (x >= minValue and
                                                          x <= maxValue)]
    idx = self._random.randint(len(candidates))
    return candidates[idx]


  def generateSensorimotorSequence(self, length):
    """
    Generate a sensorimotor sequence of the given length through this agent's
    world.

    @param length (int)           Length of sequence to generate

    @return (tuple) (sensor sequence, motor sequence, sensorimotor sequence)
    """
    sensorSequence = []
    motorSequence = []
    sensorimotorSequence = []

    for _ in xrange(length):
      sensorPattern = self._world.sense()
      motorValue = self.chooseMotorValue()
      motorPattern = self._world.move(motorValue)
      sensorSequence.append(sensorPattern)
      motorSequence.append(motorPattern)

      sensorimotorPattern = (sensorPattern |
        set([x + self._world.universe.nSensor for x in motorPattern]))
      sensorimotorSequence.append(sensorimotorPattern)

    return (sensorSequence, motorSequence, sensorimotorSequence)
