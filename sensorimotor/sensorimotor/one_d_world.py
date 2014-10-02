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
               sensorSequence):
    """
    Represents a simple 1D world.

    @param universe       (AbstractUniverse) Universe that the world belongs to.
    @param sensorSequence (list)             List of integers, each specifying
                                             a sensor value. Represents spatial
                                             configuration of patterns.
    """
    super(OneDWorld, self).__init__(universe)

    self.sensorSequence = sensorSequence


  def toString(self):
    """
    Human readable representation of the world
    """
    s = ""
    for p in self.sensorSequence:
      s += self.universe.decodeSensorValue(p)
    return s.encode("utf-8")
