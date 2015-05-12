from union_pooling.strategies.activation_function_base import (
  ActivationFunctionBase)

class LinearActivationFunction(ActivationFunctionBase):
  

  def __init__(self, slope=1, lowerBound=0, upperBound=1):
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
