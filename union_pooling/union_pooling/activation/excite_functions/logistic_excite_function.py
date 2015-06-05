import numpy

from excite_function_base import ExciteFunctionBase



class LogisticExciteFunction(ExciteFunctionBase):
  """
  Implementation of a logistic activation function for activation updating.
  Specifically, the function has the following form:

  f(x) = maxValue / (1 + exp(-steepness * (x - xMidpoint) ) )

  Note: The excitation rate is linear. The activation function is
  logistic.
  """
  _SMALL_POSITIVE_CONSTANT = 0.000001


  def __init__(self, xMidpoint=0, maxValue=1, steepness=1):
    """
    :param xMidpoint: Controls where function output is half of 'maxValue,'
                      i.e. f(xMidpoint) = maxValue / 2
    :param maxValue: Controls the maximum value of the function's range
    :param steepness: Controls the steepness of the "middle" part of the
                      curve where output values begin changing rapidly.
                      Must be a non-zero value.
    """
    assert steepness != 0

    self._xMidpoint = xMidpoint
    self._maxValue = maxValue
    self._steepness = steepness


  def excite(self, current, amount):
    """
    Increases current activation by amount.
    :param current: Current activation value(s) to be excited
    :type current: ndarray
    :param amount: Amount of excitation. Must be a positive value.
    :type amount: float
    """
    assert amount >= 0
    return self._updateActivation(current, amount)


  def _updateActivation(self, current, amount):
    # Find zero-valued elements and bump values up slightly
    # since current is a divisor below
    zeroValued = numpy.where(current == 0)[0]

    # TODO: Code is not correct
    current[zeroValued] = self._SMALL_POSITIVE_CONSTANT

    # Apply inverse logistic function to current
    converted = (numpy.log(self._maxValue / current - 1) /
                 -self._steepness + self._xMidpoint)
    converted += amount

    # Apply logistic function to updated domain value
    current = self._maxValue / (1 + numpy.exp(-self._steepness *
                                (converted - self._xMidpoint)))
    return current
