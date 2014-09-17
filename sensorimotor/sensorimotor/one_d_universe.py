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

from nupic.data.pattern_machine import PatternMachine, ConsecutivePatternMachine

from sensorimotor.abstract_universe import AbstractUniverse



class OneDUniverse(AbstractUniverse):


  def __init__(self, debugSensor=False, debugMotor=False, **kwargs):
    """
    @param debugSensor (bool) Controls whether sensor encodings are contiguous
    @param debugMotor  (bool) Controls whether motor encodings are contiguous
    """
    super(OneDUniverse, self).__init__(**kwargs)

    SensorPatternMachine = (ConsecutivePatternMachine if debugSensor
                            else PatternMachine)
    self.sensorPatternMachine = SensorPatternMachine(
      self.nSensor, self.wSensor)

    MotorPatternMachine = (ConsecutivePatternMachine if debugMotor
                            else PatternMachine)
    self.motorPatternMachine = MotorPatternMachine(
      self.nMotor, self.wMotor)

    # Purely for debugging convenience, this pool is a human friendly
    # representation of sensory values
    self.sensoryElements=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                               "abcdefghijklmnopqrstuvwxyz0123456789")


  def encodeSensorValue(self, sensorValue):
    """
    @param sensorValue (object) Sensor value

    @return (set) Sensor pattern
    """
    return self.sensorPatternMachine.get(sensorValue)


  def decodeSensorValue(self, sensorValue):
    """
    @param sensorValue (object) Sensor value

    @return (string) Human viewable representation of sensorValue
    """
    return self.sensoryElements[sensorValue]


  def encodeMotorValue(self, motorValue):
    """
    @param motorValue (object) Motor value

    @return (set) Motor pattern
    """
    numMotorValues = self.nMotor / self.wMotor
    motorRadius = (numMotorValues - 1) / 2
    return self.motorPatternMachine.get(motorValue + motorRadius)
