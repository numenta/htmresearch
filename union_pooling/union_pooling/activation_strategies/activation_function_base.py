from abc import ABCMeta, abstractmethod

class ActivationFunctionBase(object):

  __metaclass__ = ABCMeta


  def __init__(self):
    pass


  @abstractmethod
  def excite(self, current, amount):
    pass


  @abstractmethod
  def decay(self, current, amount):
    pass
