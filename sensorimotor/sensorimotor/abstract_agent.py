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
from prettytable import PrettyTable



class AbstractAgent(object):
  __metaclass__ = abc.ABCMeta


  def __init__(self, world):
    """
    @param world (AbstractWorld) The world this agent belongs in.
    """
    self.world = world


  def sense(self):
    """
    Returns the encoding of the sensor pattern at the current position (as
    indices of active bits)

    @return (set) Sensor pattern
    """
    return self.world.universe.encodeSensorValue(self.getSensorValue())


  @abc.abstractmethod
  def getSensorValue(self):
    """
    Returns the sensor value at the current position

    @return (set) Sensor pattern
    """
    return


  @abc.abstractmethod
  def chooseMotorValue(self):
    """
    Return a plausible next motor value.

    @return (object) motor value (type depends on world)
    """
    return


  @abc.abstractmethod
  def move(self, motorValue):
    """
    Given a motor value, return an encoding of the motor command and move the
    agent based on that command.

    @param motorValue (object) Action to perform in world. Type depends on
                               world.

    @return (set) Motor pattern
    """
    return


  def generateSensorimotorSequence(self, length, verbosity=0):
    """
    Generate a sensorimotor sequence of the given length through this agent's
    world.

    @param length (int)           Length of sequence to generate

    @param verbosity (int)        If > 0 print the sequence details

    @return (tuple) (sensor sequence, motor sequence, sensorimotor sequence)
    """
    sensorSequence = []
    motorSequence = []
    sensorimotorSequence = []
    sensorValues = []
    motorCommands = []

    for _ in xrange(length):
      sensorValues.append(self.getSensorValue())
      sensorPattern = self.sense()

      motorValue = self.chooseMotorValue()
      motorCommands.append(motorValue)
      motorPattern = self.move(motorValue)

      sensorSequence.append(sensorPattern)
      motorSequence.append(motorPattern)
      sensorimotorPattern = (sensorPattern |
        set([x + self.world.universe.nSensor for x in motorPattern]))
      sensorimotorSequence.append(sensorimotorPattern)

    if verbosity > 1:
      print "Generating sensorimotor sequence for world:", self.world.toString()
      self._printSequence(sensorSequence, motorSequence,
                          sensorValues, motorCommands)
      print

    return (sensorSequence, motorSequence, sensorimotorSequence)


  def _printSequence(self, sensorSequence, motorSequence,
                     sensorValues, motorCommands):
    """
    Nicely print the sequence to console for debugging purposes.
    """
    table = PrettyTable(["#",
                         "Sensor Pattern", "Motor Pattern",
                         "Sensor Value", "Motor Value"])
    for i in xrange(len(sensorSequence)):
      sensorPattern = sensorSequence[i]
      motorPattern = motorSequence[i]
      if sensorPattern is None:
        table.add_row([i, "<reset>", "<reset>"])
      else:
        table.add_row([i,
                       sorted(list(sensorPattern)),
                       sorted(list(motorPattern)),
                       self.world.universe.decodeSensorValue(sensorValues[i]),
                       motorCommands[i]])
    print table
