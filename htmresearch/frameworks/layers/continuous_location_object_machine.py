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

import math
import random

import numpy as np
import matplotlib.pyplot as plt

from nupic.encoders.coordinate import CoordinateEncoder
from htmresearch.frameworks.layers.object_machine_base import ObjectMachineBase



class ContinuousLocationObjectMachine(ObjectMachineBase):
  """
  This implementation of the object machine uses continuous locations instead
  of discrete random ones. They are created using a CoordinateEncoder.
  """

  def __init__(self,
               numInputBits=41,
               sensorInputSize=2048,
               externalInputSize=2048,
               numCorticalColumns=1,
               numFeatures=400,
               dimension=3,
               seed=42):
    """
    At creation, the SimpleObjectMachine creates a pool of locations and
    features SDR's.

    Parameters:
    ----------------------------
    @param   numInputBits (int)
             Number of ON bits in the input. Note: it should be uneven as the
             encoder only accepts uneven number of bits.

    @param   sensorInputSize (int)
             Total number of bits in the sensory input

    @param   externalInputSize (int)
             Total number of bits the external (location) input

    @param   numCorticalColumns (int)
             Number of cortical columns used in the experiment

    @param   dimension (int)
             Dimension of the locations. Will typically be 3.

    @param   numFeatures (int)
             Number of feature SDRs to generate per cortical column. There is
             typically no need to not use the default value, unless the user
             knows he will use more than 400 patterns.

    @param   seed (int)
             Seed to be used in the machine

    """
    super(ContinuousLocationObjectMachine, self).__init__(numInputBits,
                                                          sensorInputSize,
                                                          externalInputSize,
                                                          numCorticalColumns,
                                                          seed)

    # location and features pool
    self.numFeatures = numFeatures
    self._generateFeatures()

    self.dimension = dimension
    self.locationEncoder = CoordinateEncoder(
      w=numInputBits,
      n=externalInputSize,
      name="locationEncoder"
    )


  def provideObjectsToLearn(self, learningConfig, plot=False):
    """
    Returns the objects in a canonical format to be sent to an experiment.

    The returned format is a a dictionary where the keys are object names, and
    values are lists of sensations, each sensation being a mapping from
    cortical column index to a pair of SDR's (one location and one feature).

    The input, learningConfig, should have the following format. It is a
    mapping from object to a list of locations to sample from, those and
    the number of points to sample from each location. Note at these objects
    should be first added with .addObjects()

    learningConfig = {
      "cube": [("face", 5), ("edge", 5)]
      "cylinder": [(feature, 5) for feature in cylinder.getLocations()]
    }

    Note: instead of those tuples, an explicit location can be provided, e.g.:
      "cube": [(10, 22, 33), (12, 45, 31)]

    Parameters:
    ----------------------------
    @param   learningConfig (dict)
             Configuration for learning, as described above.

    """
    objects = {}

    for objectName, locationList in learningConfig.iteritems():

      sensationList = []
      physicalObject = self.objects[objectName]
      if plot:
        fig, ax = physicalObject.plot()

      for element in locationList:

        #  location name and number of points
        if len(element) == 2:
          featureName, numLocations = element
          for _ in xrange(numLocations):
            location = physicalObject.sampleLocationFromFeature(featureName)
            sensationList.append(
              self._getSDRPairs(
                [(location,
                  physicalObject.getFeatureID(location))] * self.numColumns
              )
            )
            if plot:
              x, y, z = tuple(location)
              ax.scatter(x, y, z, marker="v", s=100, c="r")

        # explicit location
        elif len(element) == 3:
          location = list(element)
          sensationList.append(
            self._getSDRPairs(
              [(location,
                physicalObject.getFeatureID(location))] * self.numColumns
            )
          )
          if plot:
            x, y, z = tuple(location)
            ax.scatter(x, y, z, marker="v", s=100, c="r")

        else:
          raise ValueError("Unsupported type for location spec")

      objects[objectName] = sensationList
      if plot:
        plt.title("Learning points for object {}".format(objectName))
        plt.savefig("learn_{}.png".format(objectName))
        plt.close()

    self._checkObjectsToLearn(objects)
    return objects


  def provideObjectToInfer(self, inferenceConfig, plot=False):
    """
    Returns the sensations in a canonical format to be sent to an experiment.

    The input inferenceConfig should be a dict with the following form:
    {
      "numSteps": 2  # number of sensations
      "noiseLevel": 0.05  # noise to add to sensations (optional)
      "objectName": 0  # optional
      "pairs": {
        0: ["random" 2), ("face", 2)]  # locations for cortical column 0
        1: [(12, 32, 34), (23, 23, 32)]  # locations for cortical column 1
      }
    }

    Again, the locations can be explicitly provided.

    The returned format is a a lists of sensations, each sensation being a
    mapping from cortical column index to a pair of SDR's (one location and
    one feature).

    Parameters:
    ----------------------------
    @param   inferenceConfig (dict)
             Inference spec for experiment (cf above for format)

    """
    if "numSteps" in inferenceConfig:
      numSteps = inferenceConfig["numSteps"]
    else:
      numSteps = len(inferenceConfig["pairs"][0])

    if "noiseLevel" in inferenceConfig:
      noise = inferenceConfig["noiseLevel"]
    else:
      noise = None

    # some checks
    if numSteps == 0:
      raise ValueError("No inference steps were provided")
    for col in xrange(self.numColumns):
      if len(inferenceConfig["pairs"][col]) != numSteps:
        raise ValueError("Incompatible numSteps and actual inference steps")

    if "objectName" in inferenceConfig:
      physicalObject = self.objects[inferenceConfig["objectName"]]
    else:
      physicalObject = None
    if plot:
      # don't use if object is not known
      fig, ax = physicalObject.plot()
      colors = plt.cm.rainbow(np.linspace(0, 1, numSteps))

    sensationSteps = []
    for step in xrange(numSteps):
      pairs = [
        inferenceConfig["pairs"][col][step] for col in xrange(self.numColumns)
      ]
      for i in xrange(len(pairs)):
        if isinstance(pairs[i], str):
          location = physicalObject.sampleLocationFromFeature(pairs[i])
          pairs[i] = (
            location,
            physicalObject.getFeatureID(location)
          )
        else:
          location = pairs[i]
          pairs[i] = (
            location,
            physicalObject.getFeatureID(location)
          )
        if plot:
          x, y, z = tuple(location)
          ax.scatter(x, y, z, marker="v", s=100, c=colors[step])

      sensationSteps.append(self._getSDRPairs(pairs, noise=noise))

    if plot:
      plt.title("Inference points for object {}".format(
        inferenceConfig["objectName"])
      )
      plt.savefig("infer_{}.png".format( inferenceConfig["objectName"]))
      plt.close()

    self._checkObjectToInfer(sensationSteps)
    return sensationSteps


  def addObject(self, object, name=None):
    """
    Adds an object to the Machine.

    Objects should be PhysicalObjects.
    """
    if name is None:
      name = len(self.objects)

    self.objects[name] = object


  def _getSDRPairs(self, pairs, noise=None):
    """
    This method takes a list of (location, feature) pairs (one pair per
    cortical column), and returns a sensation dict in the correct format,
    adding noise if necessary.

    In each pair, the location is an actual integer location to be encoded,
    and the feature is just an index.
    """
    sensations = {}
    for col in xrange(self.numColumns):
      location, featureID = pairs[col]
      location = [int(coord) for coord in location]

      location = self.locationEncoder.encode(
        (np.array(location, dtype="int32"), self._getRadius(location))
      )
      location = set(location.nonzero()[0])

      # generate empty feature if requested
      if featureID == -1:
        feature = set()
      # generate union of features if requested
      elif isinstance(featureID, tuple):
        feature = set()
        for idx in list(featureID):
          feature = feature | self.features[col][idx]
      else:
        feature = self.features[col][featureID]

      if noise is not None:
        location = self._addNoise(location, noise)
        feature = self._addNoise(feature, noise)

      sensations[col] = (location, feature)

    return sensations


  def _getRadius(self, location):
    """
    Returns the radius associated with the given location.
    """
    # TODO: find better heuristic
    return int(math.sqrt(sum([coord ** 2 for coord in location])))


  def _addNoise(self, pattern, noiseLevel):
    """
    Adds noise the given list of patterns and returns a list of noisy copies.
    """
    if pattern is None:
      return None

    newBits = []
    for bit in pattern:
      if random.random() < noiseLevel:
        newBits.append(random.randint(0, max(pattern)))
      else:
        newBits.append(bit)

    return set(newBits)


  def _generatePattern(self, numBits, totalSize):
    """
    Generates a random SDR with specified number of bits and total size.
    """
    cellsIndices = range(totalSize)
    random.shuffle(cellsIndices)
    return set(cellsIndices[:numBits])


  def _generateFeatures(self):
    """
    Generates a pool of features to be used for the experiments.

    For each index, numColumns SDR's are created, as locations for the same
    feature should be different for each column.
    """
    size = self.sensorInputSize
    bits = self.numInputBits

    self.features = []
    for _ in xrange(self.numColumns):
      self.features.append(
        [self._generatePattern(bits, size) for _ in xrange(self.numFeatures)]
    )
