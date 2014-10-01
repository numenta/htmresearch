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

from sensorimotor.abstract_one_d_agent import AbstractOneDAgent



class ExhaustiveOneDAgent(AbstractOneDAgent):


  def __init__(self, world, startPosition):
    """
    @param world         (AbstractWorld) The world this agent belongs in.
    @param startPosition (int)           The starting position for this agent.
    """
    super(ExhaustiveOneDAgent, self).__init__(world, startPosition)

    self.currentPosition = startPosition
    self._moves = []


  def chooseMotorValue(self):
    """
    Return a plausible next motor value.

    @return (int) motor value
    """
    if not len(self._moves):
      self._moves = self._generate(self.currentPosition)

    return self._moves.pop(0)


  def _generate(self, startPosition):
    moves = []
    currentPosition = startPosition
    homePositions = range(len(self.world.sensorSequence))
    homePositions.remove(currentPosition)
    awayPositions = list(homePositions)

    while len(homePositions):
      if not len(awayPositions):
        newHomePosition = homePositions.pop(0)
        moves.append(newHomePosition - currentPosition)
        currentPosition = newHomePosition
        awayPositions = list(homePositions)

      if len(awayPositions):
        awayPosition = awayPositions.pop(0)
        moves.append(awayPosition - currentPosition)
        moves.append(currentPosition - awayPosition)

    moves.append(startPosition - currentPosition)  # Finish at start position
    return moves
