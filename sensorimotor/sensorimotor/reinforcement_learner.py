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

class ReinforcementLearner(object):

  def __init__(self, actions, alpha=0.2, gamma=0.8, elambda=0.3):
    self.actions = actions
    self.alpha = alpha
    self.gamma = gamma
    self.elambda = elambda


  def qValue(self, state, action):
    raise NotImplementedError()


  def value(self, state):
    raise NotImplementedError()


  def bestAction(self, state):
    raise NotImplementedError()


  def update(self, state, action, nextState, nextAction, reward):
    raise NotImplementedError()
