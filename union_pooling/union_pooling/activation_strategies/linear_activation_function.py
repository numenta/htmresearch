from union_pooling.activation_strategies.activation_function_base import (
  ActivationFunctionBase)

class LinearActivationFunction(ActivationFunctionBase):
  """
  Implementation of simple linear activation function for activation updating.
  Specifically, the function has the following form:
  f(x) = slope * x
  """
  

  def __init__(self, slope=1, lowerBound=0, upperBound=1):
    """
    :param slope: slope of linear function
    :param lowerBound: controls a lower bound on the minimum value of
    function output
    :param upperBound: controls a lower bound on the maximum value of
    function output
    """
    self._slope = slope
    self._lowerBound = lowerBound
    self._upperBound= upperBound


  def excite(self, current, amount):
    assert amount >= 0
    current += amount * self._slope
    current[current > self._upperBound] = self._upperBound
    return current


  def decay(self, current, amount):
    assert amount >= 0
    current -= amount * self._slope
    current[current < self._lowerBound] = self._lowerBound
    return current
