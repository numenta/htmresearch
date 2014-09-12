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

from nupic.data.pattern_machine import ConsecutivePatternMachine

from sensorimotor.abstract_universe import AbstractUniverse



class OneDUniverse(AbstractUniverse):


  def __init__(self, motorRadius, patternMachine, **kwargs):
    """
    @param motorRadius    (int)            Radius of possible motor values
    @param patternMachine (PatternMachine) Pattern machine to use for
                                           encoding sensor values
    """
    super(OneDUniverse, self).__init__(**kwargs)

    self.motorRadius = motorRadius
    self.sensorPatternMachine = patternMachine

    numMotorPatterns = motorRadius * 2 + 1
    assert (self.nMotor / numMotorPatterns ==  self.wMotor), \
           "Number of motor patterns is too large given wMotor"

    self.motorPatternMachine = ConsecutivePatternMachine(
      self.nMotor, self.nMotor / numMotorPatterns)


  def encodeSensorValue(self, sensorValue):
    """
    @param sensorValue (object) Sensor value

    @return (set) Sensor pattern
    """
    return self.sensorPatternMachine.get(sensorValue)


  def encodeMotorValue(self, motorValue):
    """
    @param motorValue (object) Motor value

    @return (set) Motor pattern
    """
    return self.motorPatternMachine.get(motorValue + self.motorRadius)
