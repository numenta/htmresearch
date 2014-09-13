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

import abc



class AbstractUniverse(object):
  """
  A Universe represents the space of possible sensory and motor patterns
  that can be produced in an experiment. This class holds all the sensory
  and motor encoder parameters, as well as the encoder instances.
  """
  __metaclass__ = abc.ABCMeta


  def __init__(self, nSensor=1024, wSensor=20, nMotor=1024, wMotor=20):
    """
    @param nSensor (int) Number of bits representing sensor pattern
    @param wSensor (int) Number of bits active in sensor pattern
    @param nMotor  (int) Number of bits representing motor pattern
    @param wMotor  (int) Number of bits active in motor pattern
    """
    self.nSensor = nSensor
    self.wSensor = wSensor
    self.nMotor = nMotor
    self.wMotor = wMotor


  @abc.abstractmethod
  def encodeSensorValue(self, sensorValue):
    """
    @param sensorValue (object) Sensor value

    @return (set) Sensor pattern
    """
    return


  @abc.abstractmethod
  def encodeMotorValue(self, motorValue):
    """
    @param motorValue (object) Motor value

    @return (set) Motor pattern
    """
    return
