import sys

from excite_function_base import ExciteFunctionBase


class LinearExciteFunction(ExciteFunctionBase):
  """
  Implementation of simple linear activation function for activation excitation.
  Specifically, the function has the following form:
  f(x) = slope * x
  """


  def __init__(self, slope=1, lowerBound=0, upperBound=sys.maxint):
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
    current += amount * self._slope
    current[current > self._upperBound] = self._upperBound
    return current
