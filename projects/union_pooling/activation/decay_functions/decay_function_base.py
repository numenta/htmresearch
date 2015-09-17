from abc import ABCMeta, abstractmethod

class DecayFunctionBase(object):

  __metaclass__ = ABCMeta


  def __init__(self):
    pass


  @abstractmethod
  def decay(self, current, amount):
    pass
