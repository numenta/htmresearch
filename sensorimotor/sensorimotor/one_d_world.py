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

from sensorimotor.abstract_world import AbstractWorld



class OneDWorld(AbstractWorld):


  def __init__(self,
               universe,
               sensorSequence,
               startPosition):
    """
    Represents a simple 1D world.

    @param universe (AbstractUniverse) Universe that the world belongs to.
    @param sensorSequence (list) List of integers, each specifying a sensor
                                 value. Represents spatial configuration of
                                 patterns.
    @param startPosition  (int)  Initial position (index into sensor
                                 sequence)
    """
    super(OneDWorld, self).__init__(universe)

    self.sensorSequence = sensorSequence
    self.currentPosition = startPosition


  def __str__(self):
    """
    Human readable representation of the world
    """
    s = ""
    for p in self.sensorSequence:
      s += self.universe.decodeSensorValue(p)
    return s


  def sense(self):
    """
    Returns the encoding of the sensor pattern at the current position (as
    indices of active bits)

    @return (set) Sensor pattern
    """
    return self.universe.encodeSensorValue(self.getSensorValue())


  def getSensorValue(self):
    """
    Returns the sensor value at the current position

    @return (set) Sensor pattern
    """
    return self.sensorSequence[self.currentPosition]


  def move(self, motorValue):
    """
    Given a motor value, return an encoding of the motor command and move the
    agent based on that command.

    @param motorValue (int) Number of positions to move.
                            Positive => Right; Negative => Left; 0 => No-op

    @return (set) Motor pattern
    """
    self.currentPosition += motorValue
    return self.universe.encodeMotorValue(motorValue)


  def distanceToBoundaries(self):
    """
    @return (tuple) (distance to left, distance to right)
    """
    return (self.currentPosition,
            len(self.sensorSequence) - self.currentPosition - 1)