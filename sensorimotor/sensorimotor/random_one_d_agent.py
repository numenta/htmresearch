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

import numpy

from sensorimotor.abstract_one_d_agent import AbstractOneDAgent



class RandomOneDAgent(AbstractOneDAgent):


  def __init__(self, world, startPosition,
               possibleMotorValues=(-1, 1), seed=42):
    """
    @param world         (AbstractWorld) The world this agent belongs in.
    @param startPosition (int)           The starting position for this agent.

    @param possibleMotorValues (set) Set of motor values to choose from
    @param seed                (int) Seed for random number generator
    """
    super(RandomOneDAgent, self).__init__(world, startPosition)

    self._possibleMotorValues = possibleMotorValues
    self._random = numpy.random.RandomState(seed)


  def chooseMotorValue(self):
    """
    Return a plausible next motor value.

    @return (int) motor value
    """
    distanceToLeft, distanceToRight = self.distanceToBoundaries()
    minValue = -distanceToLeft
    maxValue = distanceToRight
    candidates = [x for x in self._possibleMotorValues if (x >= minValue and
                                                           x <= maxValue)]
    idx = self._random.randint(len(candidates))
    return candidates[idx]
