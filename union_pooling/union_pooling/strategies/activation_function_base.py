from abc import ABCMeta, abstractmethod

class ActivationFunctionBase(metaclass=ABCMeta):


  @abstractmethod
  def excite(self, current, amount):
    pass


  @abstractmethod
  def decay(self, current, amount):
    pass
