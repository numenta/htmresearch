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

try:
  import htmsanity.nupic.runner as sanity
except ImportError:
  sanity = None

from htmresearch.support.register_regions import registerAllResearchRegions
from htmresearch.frameworks.layers.laminar_network import createNetwork


NETWORK_CONFIG = {
  "networkType": "L4L2Column",
  "externalInputSize": 1024,
  "sensorInputSize": 1024,
  "L4Params": {
    "columnCount": 1024,
    "cellsPerColumn": 8,
    "formInternalBasalConnections": 0,
    "learningMode": 1,
    "inferenceMode": 1,
    "learnOnOneCell": 0,
    "initialPermanence": 0.51,
    "connectedPermanence": 0.6,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "minThreshold": 10,
    "predictedSegmentDecrement": 0.004,
    "activationThreshold": 10,
    "maxNewSynapseCount": 20,
    "maxSegmentsPerCell": 255,
    "maxSynapsesPerSegment": 255,
    "implementation": "cpp",
    "monitor": 0,
    "seed": 40
  },
  "L2Params": {
    "columnCount": 1024,
    "inputWidth": 1024 * 8,
    "learningMode": 1,
    "inferenceMode": 1,
    "minThreshold": 13,
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
               useSanity=False,
               numBits=20,
               numLearningPoints=4,
               overrides=None,
               seed=40):
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

    self.locations = self._generateLocations(numBits=numBits)
    self.features = self._generateFeatures(numBits=numBits)
    self.objects = []
    # will be populated during training
    self.objectL2Representations = []

    self.statistics = []
    self.sanity = useSanity and sanity is not None
    if useSanity:
      self.sanityStarted = False


  def learnObjects(self, debug=False):
    """
    Learns all objects.
    """
    for object in self.objects:
      iterations = 0
      if len(object) == 0:
        continue

      self._addPointToQueue(object[0])
      self.network.run(1)

      if debug and self.sanity and not self.sanityStarted:
        sanity.patchETM(self.L4Column._tm)
        self.sanityStarted = True

      self.objectL2Representations.append(self._getL2Representation())

      for pair in object:
        for _ in xrange(self.numLearningPoints):
          self._addPointToQueue(pair)
          iterations += 1

      # send reset signal
      self._addPointToQueue(object[-1], reset=True)
      iterations += 1

      # actually learn the objects
      if iterations > 0:
        self.network.run(iterations)


  def inferObject(self, objectIndex, order="first", addNoise=False):
    """
    Go over a list of locations / features and tries to recognize an object.

    It updates various statistics as it goes.
    Note: locations / feature pairs are usually sorted in descending order
    of number of objects that have them.
    """
    self._unsetLearningMode()

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
    self._addPointToQueue(pairs[idx[0]])
    self.network.run(1)

    if self.sanity and not self.sanityStarted:
      sanity.patchETM(self.L4Column._tm)
      self.sanityStarted = True

    for i in idx:
      self._addPointToQueue(pairs[i])
      self.network.run(1)
      self._updateInferenceStats(statistics, objectIndex)

    # send reset signal
    self._addPointToQueue(pairs[idx[-1]], reset=True)
    self.network.run(1)

    # save statistics
    self.statistics.append(statistics)


  def _unsetLearningMode(self):
    self.L2Column.setParameter("learningMode", 0, 0)
    self.L4Column.setParameter("learningMode", 0, 0)


  def _addPointToQueue(self, pair, reset=False, sequenceId=0):
    """
    :param pair:
    :param reset:
    :return:
    """
    location, feature = pair
    self.sensorInput.addDataToQueue(
      list(self.features[feature]), int(reset), sequenceId
    )
    self.externalInput.addDataToQueue(
      list(self.locations[location]), int(reset), sequenceId
    )


  def _updateInferenceStats(self, statistics, objectIndex):
    """
    Updates the inference statistics.

    :param statistics:    (dict) Various statistics to save.
    """
    L4Representation = self._getL4Representation()
    L4PredictiveCells = self._getL4PredictiveCells()
    L2Representation = self._getL2Representation()

    if len(L4Representation) > 100:
      print "BURSTING"
      print sorted(list(L4PredictiveCells))
      print sorted(list(L4Representation))

    objectRepresentation = self.objectL2Representations[objectIndex]
    statistics["L4_representation"].append(len(L4Representation))
    statistics["L4_predictive"].append(len(L4PredictiveCells))
    statistics["L2_representation"].append(len(L2Representation))
    statistics["L2_overlap_with_object"].append(
      len(objectRepresentation & L2Representation)
    )


  def plotInferenceStats(self, index, keys, path):
    """
    Plots and saves the desired inference statistics.
    :param index:
    :param keys:
    :param path:
    """
    pass


  def createObjects(self, numObjects, numPoints):
    """
    :param numObjects:     (int)  Number of objects to create
    :param numPoints:      (int)  Number of points per object
    :param numCommons:
    """
    objectA = [(3, 3), (1, 1), (3, 3), (2, 2), (1, 1)]
    objectB = [(1, 1), (3, 5), (10, 10)]
    self.objects = [objectA, objectB]


  def _generateLocations(self, numBits, numLocations=100, size=None):
    """
    Generates a pool of locations to be used for the experiments.

    :return:       (list(s  `et))  List of patterns
    """
    return [self.generatePattern(numBits, size) for _ in xrange(numLocations)]


  def _generateFeatures(self, numBits, numFeatures=100, size=None):
    """
    Generates a pool of features to be used for the experiments.

    :return:       (list(set))  List of patterns
    """
    return [self.generatePattern(numBits, size) for _ in xrange(numFeatures)]


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


  def _getL4Representation(self):
    return set(self.L4Column._tm.getActiveCells())


  def _getL4PredictiveCells(self):
    return set(self.L4Column._tm.getPredictiveCells())


  def _getL2Representation(self):
    return set(self.L2Column._pooler.getActiveCells())


if __name__ == "__main__":
  exp = SingleColumnL4L2Experiment(NETWORK_CONFIG,
                                   numObjects=2,
                                   numLearningPoints=20,
                                   useSanity=False)
  exp.createObjects(1, 1)

  exp.learnObjects(debug=False)
  exp.inferObject(objectIndex=0, addNoise=False)

  print exp.statistics