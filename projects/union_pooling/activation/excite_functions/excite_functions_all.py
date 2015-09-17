# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
import matplotlib.pyplot as plt
from excite_function_base import ExciteFunctionBase



class LogisticExciteFunction(ExciteFunctionBase):
  """
  Implementation of a logistic activation function for activation updating.
  Specifically, the function has the following form:

  f(x) = (maxValue - minValue) / (1 + exp(-steepness * (x - xMidpoint) ) ) + minValue

  Note: The excitation rate is linear. The activation function is
  logistic.
  """


  def __init__(self, xMidpoint=5, minValue=10, maxValue=20, steepness=1):
    """
    @param xMidpoint: Controls where function output is half of 'maxValue,'
                      i.e. f(xMidpoint) = maxValue / 2
    @param minValue: Minimum value of the function
    @param maxValue: Controls the maximum value of the function's range
    @param steepness: Controls the steepness of the "middle" part of the
                      curve where output values begin changing rapidly.
                      Must be a non-zero value.
    """
    assert steepness != 0

    self._xMidpoint = xMidpoint
    self._maxValue = maxValue
    self._minValue = minValue
    self._steepness = steepness


  def excite(self, currentActivation, inputs):
    """
    Increases current activation by amount.
    @param currentActivation (numpy array) Current activation levels for each cell
    @param inputs            (numpy array) inputs for each cell
    """

    currentActivation = self._minValue + (self._maxValue - self._minValue) / \
                                          (1 + numpy.exp(-self._steepness * (inputs - self._xMidpoint)))

    return currentActivation

  def plot(self):
    """
    plot the activation function
    """
    plt.ion()
    plt.show()
    x = numpy.linspace(0, 15, 100)
    y = numpy.zeros(x.shape)
    y = self.excite(y, x)
    plt.plot(x, y)
    plt.xlabel('Input')
    plt.ylabel('Persistence')
    plt.title('Sigmoid Activation Function')


class FixedExciteFunction(ExciteFunctionBase):
  """
  Implementation of a simple fixed excite function
  The function reset the activation level to a fixed amount
  """


  def __init__(self, targetExcLevel=10.0):
    """
    """
    self._targetExcLevel = targetExcLevel

  def excite(self, currentActivation, inputs):
    """
    Increases current activation by a fixed amount.
    @param currentActivation (numpy array) Current activation levels for each cell
    @param inputs            (numpy array) inputs for each cell
    """

    currentActivation = self._targetExcLevel

    return currentActivation

  def plot(self):
    """
    plot the activation function
    """
    plt.ion()
    plt.show()
    x = numpy.linspace(0, 15, 100)
    y = numpy.zeros(x.shape)
    y = self.excite(y, x)
    plt.plot(x, y)
    plt.xlabel('Input')
    plt.ylabel('Persistence')    
