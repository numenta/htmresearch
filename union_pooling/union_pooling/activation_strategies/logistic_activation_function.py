import numpy

from union_pooling.activation_strategies.activation_function_base import (
  ActivationFunctionBase)

class LogisticActivationFunction(ActivationFunctionBase):


  def __init__(self, xMidpoint=0, maxValue=1, steepness=1):
    assert steepness != 0

    self._xMidpoint = xMidpoint
    self._maxValue = maxValue
    self._steepness = steepness


  def excite(self, current, amount):
    assert amount >= 0
    return self._updateActivation(current, amount)


  def decay(self, current, amount):
    assert amount >= 0
    return self._updateActivation(current, -amount)


  def _updateActivation(self, current, amount):
    # Ignore zero-valued elements since current is a divisor in equation
    nonzero = current.nonzero()

    # apply inverse logistic function
    converted = (numpy.log(self._maxValue / current[nonzero] - 1) /
                 -self._steepness + self._xMidpoint)
    converted += amount

    # apply logistic function to update
    current[nonzero] = self._maxValue / (1 + numpy.exp(-self._steepness *
                                        (converted - self._xMidpoint)))
    return current
