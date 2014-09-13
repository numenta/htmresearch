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



class AbstractWorld(object):
  """
  A World represents a particular (static) configuration of sensory parameters.
  Every instance of a World belongs to a single Universe, which in turn defines
  the space of possible sensory patterns and motor commands.

  The world keeps track of the current location. It is used to generate sensory
  SDR's given the  current position, and generate motor SDR's given the current
  location and a desired movement.
  """
  __metaclass__ = abc.ABCMeta


  def __init__(self, universe):
    """
    @param universe (AbstractUniverse) Universe that the world belongs to.
    """
    self.universe = universe


  @abc.abstractmethod
  def sense(self):
    """
    Returns the encoding of the sensor pattern at the current position (as
    indices of active bits)

    @return (set) Sensor pattern
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
