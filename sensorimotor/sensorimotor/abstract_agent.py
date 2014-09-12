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



class AbstractAgent(object):
  __metaclass__ = abc.ABCMeta


  @abc.abstractmethod
  def chooseMotorValue(self):
    """
    Return a plausible next motor value.

    @return (object) motor value (type depends on world)
    """
    return

  @abc.abstractmethod
  def generateSensorimotorSequence(self, length):
    """
    Generate a sensorimotor sequence through this agent's world.

    @param length (int)           Length of sequence to generate
    @param world  (AbstractWorld) World to act in
    @param agent  (AbstractAgent) Agent acting in world

    @return (tuple) (sensor sequence, motor sequence, sensorimotor sequence)
    """
    return