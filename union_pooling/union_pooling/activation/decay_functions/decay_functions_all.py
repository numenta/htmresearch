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



class NoDecayFunction(DecayFunctionBase):
  """
  Implementation of no decay.
  """

  def decay(self, current, amount):
    return current

class ExponentialDecayFunction(DecayFunctionBase):
  """
  Implementation of exponential decay.
  f(t) = exp(- lambda * t)
  lambda is the decay constant. The time constant is 1 / lambda
  """


  def __init__(self, time_constant=10.0):
    """
    @param (float) lambda_constant: positive exponential decay time constant.
    """
    assert not time_constant < 0
    self._lambda_constant = 1/float(time_constant)


  def decay(self, initActivationLevel, timeSinceActivation):
    """
    @param initActivationLevel: initial activation level
    @param timeSinceActivation: time since the activation
    @return: activation level after decay
    """
    activationLevel = numpy.exp(-self._lambda_constant * timeSinceActivation) *  initActivationLevel
    return activationLevel

  def plot(self):
    initValue = 10
    nStep = 20
    x = numpy.arange(0, nStep, 1)
    y = numpy.zeros(x.shape)
    y[0] = initValue

    for i in range(0, nStep):
      y[i] = self.decay(y[0], i)

    plt.ion()
    plt.show()

    plt.plot(x, y)
    plt.title('Exponential Decay Function t=' + str(1.0/self._lambda_constant))
    plt.xlabel('Time after activation (step)')
    plt.ylabel('Persistence')

    plt.plot(x, y)
    plt.xlabel('Time after activation (step)')
    plt.ylabel('Persistence')

class LogisticDecayFunction(DecayFunctionBase):
  """
  Implementation of logistic decay.
  f(t) = maxValue / (1 + exp(-steepness * (tMidpoint - t) ) )
  tMidpoint is when activation decays to half of its initial level
  steepness controls the steepness of the decay function around tMidpoint
  """


  def __init__(self, tMidpoint=10, steepness=.5):
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
    self._steepness = steepness


  def decay(self, initActivationLevel, timeSinceActivation):
    """
    @param initActivationLevel: initial activation level
    @param timeSinceActivation: time since the activation
    @return: activation level after decay
    """

    activationLevel = initActivationLevel / (1 + numpy.exp(-self._steepness * (self._xMidpoint - timeSinceActivation)))

    return activationLevel

  def plot(self):
    initValue = 10
    nStep = 20
    x = numpy.arange(0, nStep, 1)
    y = numpy.zeros(x.shape)
    y[0] = initValue

    for i in range(0, nStep):
      y[i] = self.decay(y[0], i)

    plt.ion()
    plt.show()

    plt.plot(x, y)
    plt.title('Sigmoid Decay Function + Steepness: '+ str(self._steepness))
    plt.xlabel('Time after activation (step)')
    plt.ylabel('Persistence')    