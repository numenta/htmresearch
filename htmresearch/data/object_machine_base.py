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

from abc import ABCMeta, abstractmethod



class ObjectMachineBase(object):
  """
  An object machine is a helper class that allows the user to quickly create
  objects and sensations for inference experiments. It is designed to send
  SDR's in a canonical format to such experiments. It should be subclassed
  for various ways of defining and creating objects (discrete (location,
  feature) pairs, continuous location SDR's, etc.).

  This is the base class. It has a few generic methods and specifies
  required methods for any child class.
  """

  __metaclass__ = ABCMeta


  def __init__(self,
               numInputBits,
               sensorInputSize,
               externalInputSize,
               numCorticalColumns,
               seed):
    """
    Creates the Machine, specifying its input and output size.

    Parameters:
    ----------------------------
    @param   numInputBits (int)
             Number of ON bits in the input

    @param   sensorInputSize (int)
             Total number of bits in the sensory input

    @param   externalInputSize (int)
             Total number of bits the external (location) input

    @param   numCorticalColumns (int)
             Number of cortical columns used in the experiment

    @param   seed (int)
             Seed to be used in the machine

    """
    # input and output size
    self.numColumns = numCorticalColumns
    self.numInputBits = numInputBits
    self.sensorInputSize = sensorInputSize
    self.externalInputSize = externalInputSize

    # seed
    self.seed = seed

    # objects
    self.objects = {}


  @abstractmethod
  def provideObjectsToLearn(self, objectNames=None):
    """
    This method provides the specified objects in an acceptable form for
    experiments, for the learning part.

    The expected form is a a dictionary where the keys are object names, and
    values are lists of sensations, each sensation being a mapping from
    cortical column index to a pair of SDR's (one location and one feature).

    Parameters:
    ----------------------------
    @param   objectNames (list)
             List of object names to provide to the experiment

    """


  @abstractmethod
  def provideObjectToInfer(self, inferenceConfig):
    """
    This method provides the specified objects in an acceptable form for
    experiments, for the inference part.

    The expected form is a a lists of sensations, each sensation being a
    mapping from cortical column index to a pair of SDR's (one location and
    one feature).

    Its argument is an inferenceConfig dictionary, whose fields may vary
    with the type of ObjectMachine.

    Parameters:
    ----------------------------
    @param   inferenceConfig (dict)
             Inference spec for experiment (specific to each object machine
             type)

    """


  @abstractmethod
  def addObject(self, *args, **kwargs):
    """
    Adds an object to the Machine. The arguments are specific to each
    implementation.
    """


  def _checkObjectsToLearn(self, objects):
    """
    Checks that objects have the correct format before being sent to the
    experiment.
    """
    for name, object in objects.iteritems():
      if name not in self.objects:
        raise ValueError(
          "Invalid object name \"{}\" sent to experiment".format(name)
        )

      for sensations in object:
        if set(sensations.keys()) != set(range(self.numColumns)):
          raise ValueError(
            "Invalid number of cortical column sensations sent to experiment"
          )
        for pair in sensations.values():
          if not isinstance(pair, tuple) or len(pair) != 2 or \
                  not isinstance(pair[0], set) or \
                  not isinstance(pair[1], set):
            raise ValueError("Invalid SDR's sent to experiment")


  def _checkObjectToInfer(self, sensationList):
    """
    Checks that objects have the correct format before being sent to the
    experiment.
    """
    for sensations in sensationList:
      if set(sensations.keys()) != set(range(self.numColumns)):
        raise ValueError(
          "Invalid number of cortical column sensations sent to experiment"
        )
      for pair in sensations.values():
        if not isinstance(pair, tuple) or len(pair) != 2 or \
                not isinstance(pair[0], set) or \
                not isinstance(pair[1], set):
          raise ValueError("Invalid SDR's sent to experiment")


  def __len__(self):
    """
    Custom length method.
    """
    return len(self.objects)


  def __iter__(self):
    """
    Custom iteration method.
    """
    return self.objects.__iter__()


  def __getitem__(self, item):
    """
    Custom accessor, for backwards compatibility and convenience.
    """
    return self.objects[item]
