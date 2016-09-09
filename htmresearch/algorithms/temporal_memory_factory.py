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

"""Module providing a factory for instantiating a temporal memory instance."""

import inspect

from nupic.research.temporal_memory import TemporalMemory
from nupic.bindings.algorithms import TemporalMemory as TemporalMemoryCPP
from htmresearch.algorithms.extended_temporal_memory import (
  ExtendedTemporalMemory)
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
  TemporalMemoryMonitorMixin)
from nupic.bindings.experimental import ExtendedTemporalMemory as FastETM


class MonitoredTemporalMemory(TemporalMemoryMonitorMixin, TemporalMemory):
  pass


class MonitoredExtendedTemporalMemory(TemporalMemoryMonitorMixin,
                                      ExtendedTemporalMemory):
  pass


class ReversedExtendedTemporalMemory(FastETM):
  """
  Modified version of ETM. Should be implemented (or at least allowed) when
  the "new" ETM is ported to Python.

  This class inherits from Python binding of nupic.core's extended temporal
  memory, to overwrite its compute() function. The goal is to reverse the two
  steps of inference: depolarize before activate, so that external and
  proximal input are used at the same time step.
  """

  def compute(self,
              activeColumns,
              activeExternalCells=None,
              activeApicalCells=None,
              formInternalConnections=False,
              learn=True):
    """
    Use bindings methods to reverse the calls in compute.
    """
    # sort indices for consistency with C++ version
    activeColumns = sorted(list(activeColumns))

    if activeExternalCells is not None:
      activeExternalCells = sorted(list(activeExternalCells))
    else:
      activeExternalCells = []

    if activeApicalCells is not None:
      activeApicalCells = sorted(list(activeApicalCells))
    else:
      activeApicalCells = []

    self.activateBasalDendrites(
      activeExternalCells,
      learn
    )
    self.activateApicalDendrites(
      activeApicalCells,
      learn
    )
    self.activateCells(
      activeColumns,
      activeExternalCells,
      activeApicalCells,
      learn
    )



class TemporalMemoryTypes(object):
  """ Enumeration of supported classification model types, mapping userland
  identifier to constructor.  See createModel() for actual factory method
  implementation.
  """
  extended = ExtendedTemporalMemory
  extendedCPP = FastETM
  reversedExtendedCPP = ReversedExtendedTemporalMemory
  extendedMixin = MonitoredExtendedTemporalMemory
  tm = TemporalMemory
  tmMixin = MonitoredTemporalMemory
  tmCPP = TemporalMemoryCPP


  @classmethod
  def getTypes(cls):
    """ Get sequence of acceptable model types.  Iterates through class
    attributes and separates the user-defined enumerations from the default
    attributes implicit to Python classes. i.e. this function returns the names
    of the attributes explicitly defined above.
    """

    for attrName in dir(cls):
      attrValue = getattr(cls, attrName)
      if (isinstance(attrValue, type)):
        yield attrName # attrName is an acceptable model name and


def createModel(modelName, **kwargs):
  """
  Return a classification model of the appropriate type. The model could be any
  supported subclass of ClassficationModel based on modelName.

  @param modelName (str)  A supported temporal memory type

  @param kwargs    (dict) Constructor argument for the class that will be
                          instantiated. Keyword parameters specific to each
                          model type should be passed in here.
  """

  if modelName not in TemporalMemoryTypes.getTypes():
    raise RuntimeError("Unknown model type: " + modelName)

  return getattr(TemporalMemoryTypes, modelName)(**kwargs)


def getConstructorArguments(modelName):
  """
  Return constructor arguments and associated default values for the
  given model type.

  @param modelName (str)  A supported temporal memory type

  @return argNames (list of str) a list of strings corresponding to constructor
                                 arguments for the given model type, excluding
                                 'self'.
  @return defaults (list)        a list of default values for each argument
  """

  if modelName not in TemporalMemoryTypes.getTypes():
    raise RuntimeError("Unknown model type: " + modelName)

  argspec = inspect.getargspec(
                            getattr(TemporalMemoryTypes, modelName).__init__)
  return (argspec.args[1:], argspec.defaults)

