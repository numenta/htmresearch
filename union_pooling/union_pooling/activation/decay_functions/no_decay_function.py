from decay_function_base import DecayFunctionBase



class NoDecayFunction(DecayFunctionBase):
  """
  Implementation of no decay.
  """

  def decay(self, current, amount):
    return current
