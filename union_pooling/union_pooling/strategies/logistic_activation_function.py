from union_pooling.strategies.activation_function_base import (
  ActivationFunctionBase)

class LogisticActivationFunction(ActivationFunctionBase):


  def __init__(self, xMidpoint=0, maxValue=1, steepness=1):
    self._xMidpoint = xMidpoint
    self._maxValue = maxValue
    self._steepness = steepness


  def excite(self, current, amount):
    assert amount >= 0
    pass

  def decay(self, current, amount):
    assert amount >= 0
    pass
