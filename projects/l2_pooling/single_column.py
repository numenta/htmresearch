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

import random
import collections

from htmresearch.support.register_regions import registerAllResearchRegions
from htmresearch.frameworks.layers.laminar_network import createNetwork

NETWORK_CONFIG = {
  "networkType": "L4L2Column",
  "externalInputSize": 1024,
  "sensorInputSize": 1024,
  "L4Params": {
    "columnCount": 1024,
    "cellsPerColumn": 8,
    "formInternalConnections": 0,
    "learningMode": 1,
    "inferenceMode": 1,
    "learnOnOneCell": 0,
    "initialPermanence": 0.51,
    "connectedPermanence": 0.6,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "minThreshold": 10,
    "predictedSegmentDecrement": 0.08,
    "activationThreshold": 13,
    "maxNewSynapseCount": 20,
  },
  "L2Params": {
    "columnCount": 1024,
    "inputWidth": 1024 * 8,
    "learningMode": 1,
    "inferenceMode": 1,
    "minThreshold": 15
  }
}



class SingleColumnL4L2Experiment(object):
  """
  Single column L4-L2 experiment.

  This experiment uses the network API to test out various properties of
  inference and learning using a single sensor and an L4-L2 network. For now,
  we directly use the locations on the object.
  """

  def __init__(self,
               netConfig,
               numObjects,
               numLearningPoints=3,
               overrides=None,
               seed=42):
    """
    :param config:      (dict)  Default network parameters
    :param overrides:   (dict)  Overrides
    """
    registerAllResearchRegions()

    self.config = dict(netConfig)
    self.numLearningPoints = numLearningPoints
    self.numObjects = numObjects
    random.seed(seed)

    # update parameters with overrides
    if overrides is None:
      overrides = {}
    self.config.update(overrides)
    for key in overrides.iterkeys():
      if isinstance(overrides[key], dict):
        self.config[key].update(overrides[key])
      else:
        self.config[key] = overrides[key]

    # create network
    self.network = createNetwork(self.config)
    self.sensorInput = self.network.regions["sensorInput_0"].getSelf()
    self.externalInput = self.network.regions["externalInput_0"].getSelf()
    self.L4Column = self.network.regions["L4Column_0"].getSelf()
    self.L2Column = self.network.regions["L2Column_0"].getSelf()

    self.locations = []
    self.features = []
    self.objects = []

    self.statistics = []


  def learnObjects(self):
    """
    Learns all objects.
    """
    iterations = 0
    for object in self.objects:
      if len(object) == 0:
        continue

      # send initial signal (because of one-off problem in using ETM for L4)
      self.addPointToQueue(object[0])
      iterations += 1

      for pair in object:
        for _ in xrange(self.numLearningPoints):
          self.addPointToQueue(pair)
          iterations += 1

      # send reset signal
      self.addPointToQueue(object[-1], reset=True)
      iterations += 1

    # actually learn the objects
    if iterations > 0:
      self.network.run(iterations)


  def inferObject(self, objectIndex, order="first"):
    """
    Go over a list of locations / features and tries to recognize an object.

    It updates various statistics as it goes.
    Note: locations / feature pairs are usually sorted in descending order
    of number of objects that have them.
    """
    self.network.reset()

    pairs = self.objects[objectIndex]
    if len(pairs) == 0:
      raise ValueError("No location/feature pair for specified object")

    statistics = collections.defaultdict(list)

    # determine order to go over pairs
    idx = range(len(pairs))
    if isinstance(order, str):
      if order == "first":
        pass
      elif order == "last":
        idx = idx[::-1]
      else:
        random.shuffle(idx)
    elif isinstance(order, list):
      assert(set(order) == set(range(pairs)))
      idx = order

    # send first signal
    self.addPointToQueue(pairs[0])
    self.network.run(1)

    for i in idx[:-1]:
      self.addPointToQueue(pairs[i])
      self.network.run(1)
      self.updateInferenceStats(statistics)

    # send reset signal
    self.addPointToQueue(pairs[idx[-1]], reset=True)
    self.network.run(1)
    self.updateInferenceStats(statistics)


  def addPointToQueue(self, pair, reset=False, sequenceId=0):
    """
    :param pair:
    :param reset:
    :return:
    """
    location, feature = pair
    self.sensorInput.addDataToQueue(
      self.features[feature], int(reset), sequenceId
    )
    self.externalInput.addDataToQueue(
      self.locations[location], int(reset), sequenceId
    )


  def updateInferenceStats(self, statistics):
    """
    Updates the inference statistics.

    :param statistics:    (dict) Various statistics to save.
    """
    pass


  def plotInferenceStats(self, index, keys, path):
    """
    Plots and saves the desired inference statistics.
    :param index:
    :param keys:
    :param path:
    """
    pass


  def createObjects(self, numObjects, numPoints, numCommons=None):
    """
    :param numObjects:     (int)  Number of objects to create
    :param numPoints:      (int)  Number of points per object
    :param numCommons:
    """
    pass


  def generateLocations(self, numBits, numLocations=100, size=None):
    """
    Generates a pool of locations to be used for the experiments.
    """
    for _ in xrange(numLocations):
      self.locations.append(self.generatePattern(numBits, size))


  def generateFeatures(self, numBits, numFeatures=100, size=None):
    """
    Generates a pool of features to be used for the experiments.
    """
    for _ in xrange(numFeatures):
      self.features.append(self.generatePattern(numBits, size))


  def generatePattern(self, numBits, size=None):
    """
    Generates a pattern, represented as a set of active bits.

    :param numBits:    (int) Number of on bits.
    :param size:       (int) Total pattern size (defaults to sensorInputSize)
    :return:           (set) Indices of active bits
    """
    if size is None:
      size = self.config["sensorInputSize"]

    return set([random.randint(0, size - 1) for _ in xrange(numBits)])



if __name__ == "__main__":
  pass
