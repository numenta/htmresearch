import numpy

from decay_function_base import DecayFunctionBase



class ExponentialDecayFunction(DecayFunctionBase):
  """
  Implementation of exponential decay.
  """


  def __init__(self, lambda_constant=1):
    """
    :param lambda_constant: positive exponential decay constant
    """
    assert not lambda_constant < 0
    self._lambda_constant = lambda_constant


  def decay(self, current, t):
    return current * numpy.exp(-self._lambda_constant * t)
