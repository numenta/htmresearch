# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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

import abc

from sensorimotor.abstract_agent import AbstractAgent



class AbstractOneDAgent(AbstractAgent):
  __metaclass__ = abc.ABCMeta


  def __init__(self, world, startPosition):
    """
    @param world         (AbstractWorld) The world this agent belongs in.
    @param startPosition (int)           The starting position for this agent.
    """
    super(AbstractOneDAgent, self).__init__(world)

    self.currentPosition = startPosition


  def getSensorValue(self):
    """
    Returns the sensor value at the current position

    @return (set) Sensor pattern
    """
    return self.world.sensorSequence[self.currentPosition]


  def move(self, motorValue):
    """
    Given a motor value, return an encoding of the motor command and move the
    agent based on that command.

    @param motorValue (int) Number of positions to move.
                            Positive => Right; Negative => Left; 0 => No-op

    @return (set) Motor pattern
    """
    self.currentPosition += motorValue
    return self.world.universe.encodeMotorValue(motorValue)


  def distanceToBoundaries(self):
    """
    @return (tuple) (distance to left, distance to right)
    """
    return (self.currentPosition,
            len(self.world.sensorSequence) - self.currentPosition - 1)
