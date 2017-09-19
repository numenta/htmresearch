# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
Factory for creating object machines.
"""

from htmresearch.frameworks.layers.simple_object_machine import (
  SimpleObjectMachine
)
from htmresearch.frameworks.layers.continuous_location_object_machine import (
  ContinuousLocationObjectMachine
)
from htmresearch.frameworks.layers.sequence_object_machine import (
  SequenceObjectMachine
)



class ObjectMachineTypes(object):
  """
  Enum class for implemented ObjectMachine types.
  """

  simple = SimpleObjectMachine
  continuous = ContinuousLocationObjectMachine
  sequence = SequenceObjectMachine

  @classmethod
  def getTypes(cls):
    """
    Get sequence of acceptable model types. Iterates through class
    attributes and separates the user-defined enumerations from the default
    attributes implicit to Python classes. i.e. this function returns the names
    of the attributes explicitly defined above.
    """
    for attrName in dir(cls):
      attrValue = getattr(cls, attrName)
      if (isinstance(attrValue, type)):
        yield attrName



def createObjectMachine(machineType, **kwargs):
  """
  Return an object machine of the appropriate type.

  @param machineType (str)  A supported ObjectMachine type

  @param kwargs      (dict) Constructor argument for the class that will be
                            instantiated. Keyword parameters specific to each
                            model type should be passed in here.
  """

  if machineType not in ObjectMachineTypes.getTypes():
    raise RuntimeError("Unknown model type: " + machineType)

  return getattr(ObjectMachineTypes, machineType)(**kwargs)
