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
  __metaclass__ = abc.ABCMeta


  def __init__(self, universe):
    """
    @param universe (AbstractUniverse) Universe that the world belongs to
    """
    self.universe = universe


  @abc.abstractmethod
  def sense(self):
    """
    Returns current sensor pattern being viewed (as indices of active bits)

    @return (set) Sensor pattern
    """
    return


  @abc.abstractmethod
  def move(self, motorValue):
    """
    @param motorValue (object) Action to perform in world. Type depends on
                               world.

    @return (set) Motor pattern
    """
    return
