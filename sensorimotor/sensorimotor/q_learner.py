# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

from collections import defaultdict
import random

import numpy

from sensorimotor.reinforcement_learner import ReinforcementLearner



class QLearner(ReinforcementLearner):

  def __init__(self, actions,
               alpha=0.2, gamma=0.8, elambda=0.3,
               n=2048):
    super(QLearner, self).__init__(actions,
                                   alpha=alpha, gamma=gamma, elambda=elambda)
    self.n = n

    self.weights = defaultdict(lambda: numpy.zeros(self.n))


  def qValue(self, state, action):
    qValue = 0

    for i in state.nonzero()[0]:
      qValue += self.weights[action][i] * state[i]

    return qValue


  def value(self, state):
    qValues = [self.qValue(state, action) for action in self.actions]
    return max(qValues) if len(qValues) else 0.0


  def bestAction(self, state):
    bestActions = []
    maxQValue = float("-inf")

    for action in self.actions:
      qValue = self.qValue(state, action)

      if qValue > maxQValue:
        bestActions = [action]
        maxQValue = qValue
      elif qValue == maxQValue:
        bestActions.append(action)

    return random.choice(bestActions) if len(bestActions) else None


  def update(self, state, action, nextState, nextAction, reward):
    targetValue = reward + self.gamma * self.value(nextState)
    qValue = self.qValue(state, action)
    correction = (targetValue - qValue) / sum(state)
    targetWeight = targetValue / sum(state)

    diffs = [abs(targetWeight - self.weights[action][i])
             for i in state.nonzero()[0]]
    maxDiff = max(diffs)

    if maxDiff != 0:
      for i in state.nonzero()[0]:
        diff = abs(targetWeight - self.weights[action][i])
        scale = diff / maxDiff
        self.weights[action][i] += self.alpha * correction * scale
