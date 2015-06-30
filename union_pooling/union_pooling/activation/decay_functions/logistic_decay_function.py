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

import numpy
import matplotlib.pyplot as plt
from decay_function_base import DecayFunctionBase



class LogisticDecayFunction(DecayFunctionBase):
  """
  Implementation of logistic decay.
  f(t) = maxValue / (1 + exp(-steepness * (tMidpoint - t) ) )
  tMidpoint is when activation decays to half of its initial level
  steepness controls the steepness of the decay function around tMidpoint
  """


  def __init__(self, tMidpoint=10, maxValue=20, steepness=1):
    """
    @param tMidpoint: Controls where function output is half of 'maxValue,'
                      i.e. f(xMidpoint) = maxValue / 2

    @param maxValue: Controls the maximum value of the function's range
    @param steepness: Controls the steepness of the "middle" part of the
                      curve where output values begin changing rapidly.
                      Must be a non-zero value.
    """
    assert steepness != 0

    self._xMidpoint = tMidpoint
    self._maxValue = maxValue
    self._steepness = steepness


  def decay(self, activationLevel, timeSinceActivation):
    """
    @param activationLevel: current activation level
    @param timeSinceActivation: time since the activation
    @return: activation level after decay
    """

    activationLevel = self._maxValue / (1 + numpy.exp(-self._steepness * (self._xMidpoint - timeSinceActivation)))

    return activationLevel

  def plot(self):
    initValue = 20
    nStep = 20
    x = numpy.arange(0, nStep, 1)
    y = numpy.zeros(x.shape)
    y[0] = initValue

    for i in range(0, nStep-1):
      y[i+1] = self.decay(y[i], i)

    plt.ion()
    plt.show()

    plt.plot(x, y)
    plt.xlabel('Time after activation (step)')
    plt.ylabel('Persistence')