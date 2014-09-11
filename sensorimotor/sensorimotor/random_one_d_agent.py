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

from sensorimotor.abstract_agent import AbstractAgent



class RandomOneDAgent(AbstractAgent):


  def __init__(self, possibleMotorValues=set([-1, 1]), seed=42):
    """
    @param possibleMotorValues (set) Set of motor values to choose from
    @param seed                (int) Seed for random number generator
    """
    self.possibleMotorValues = possibleMotorValues
    self._random = numpy.random.RandomState(seed)


  def chooseMotorValue(self, world):
    """
    @param world (OneDWorld) World to act in

    @return (int) motor value
    """
    distanceToLeft, distanceToRight = world.distanceToBoundaries()
    minValue = -distanceToLeft
    maxValue = distanceToRight
    candidates = [x for x in self.possibleMotorValues if (x >= minValue and
                                                          x <= maxValue)]
    idx = self._random.randint(len(candidates))
    return candidates[idx]
