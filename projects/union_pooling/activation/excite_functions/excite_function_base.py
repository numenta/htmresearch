from abc import ABCMeta, abstractmethod

class ExciteFunctionBase(object):

  __metaclass__ = ABCMeta


  def __init__(self):
    pass


  @abstractmethod
  def excite(self, current, amount):
    pass
