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

import os
import random
import collections
import matplotlib.pyplot as plt
from tabulate import tabulate

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
    "seed": 42
  },
  "L2Params": {
    "columnCount": 1024,
    "inputWidth": 1024 * 8,
    "learningMode": 1,
    "inferenceMode": 1,
    "minThreshold": 13,
  }
}



class L4L2Experiment(object):
  """
  L4-L2 experiment.

  This experiment uses the network API to test out various properties of
  inference and learning using a sensors and an L4-L2 network. For now,
  we directly use the locations on the object.
  """

  PLOT_DIRECTORY = "plots/"


  def __init__(self,
               name,
               networkConfig,
               overrides=None,
               numInputBits=20,
               numLearningPoints=3,
               seed=42):
    """
    :param netConfig:        (dict)  Default network parameters
    :param useSanity:        (bool)  Whether or not to use sanity
    :param numInputBits:     (int)   Number of input bits
    :param numLearningPoints (int)   Number of iterations to learn a pattern
                                     in L4
    :param overrides:        (dict)  Overrides
    :param seed:             (int)   Seed to use
    """
    registerAllResearchRegions()
    self.name = name

    self.config = dict(networkConfig)
    self.numLearningPoints = numLearningPoints
    self.numInputBits = numInputBits
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

    if "numCorticalColumns" in self.config:
      self.numColumns = self.config["numCorticalColumns"]
    else:
      self.numColumns = 1

    self.sensorInputs = []
    self.externalInputs = []
    self.L4Columns = []
    self.L2Columns = []

    for i in xrange(self.numColumns):
      self.sensorInputs.append(
        self.network.regions["sensorInput_" + str(i)].getSelf()
      )
      self.externalInputs.append(
        self.network.regions["externalInput_" + str(i)].getSelf()
      )
      self.L4Columns.append(
        self.network.regions["L4Column_" + str(i)].getSelf()
      )
      self.L2Columns.append(
        self.network.regions["L2Column_" + str(i)].getSelf()
      )

    self.objects = {}
    self._generateLocations()
    self._generateFeatures()
    # self.objects = []

    # will be populated during training
    self.objectL2Representations = {}
    self.statistics = []

    if not os.path.exists(self.PLOT_DIRECTORY):
      os.makedirs(self.PLOT_DIRECTORY)


  def learnAllObjects(self, profile=False):
    """
    Learns all objects in self.objects.

    :param profile:      (bool) If true, will print profile info at the end
    """
    self.network.resetProfiling()

    for object, pairs in self.objects.iteritems():
      iterations = 0
      if len(object) == 0:
        continue

      for pair in pairs:
        for _ in xrange(self.numLearningPoints):
          self._addPointToQueue(pairs)
          iterations += 1

      # actually learn the objects
      if iterations > 0:
        self.network.run(iterations)

      # update L2 representations
      self.objectL2Representations[object] = self.getL2Representations()

      # send reset signal
      self._sendResetSignal()

    if profile:
      self._prettyPrintProfile(source="learning")


  def infer(self, inferenceConfig, noise=None, profile=False):
    """
    Go over a list of locations / features and tries to recognize an object.

    It updates various statistics as it goes.
    Note: locations / feature pairs are usually sorted in descending order
    of number of objects that have them.

    :param object:       (Object)      List of pairs of location and feature
                                       indices to go over
    :param noise:        (float)       Noise level to add to the patterns
    :param profile:      (bool)        Print profile info at the end
    """
    self.network.resetProfiling()
    self._unsetLearningMode()

    statistics = collections.defaultdict(list)
    objectID = inferenceConfig["object"]
    numSteps = inferenceConfig["numSteps"]

    # some checks
    if numSteps == 0:
      raise ValueError("No inference steps were provided")
    for col in xrange(self.numColumns):
      if len(inferenceConfig[col]) != numSteps:
        raise ValueError("Incompatible numSteps and actual inference steps")

    for step in xrange(numSteps):
      pairs = [inferenceConfig[col][step] for col in xrange(self.numColumns)]
      self._addPointToQueue(pairs, noise=noise)
      self.network.run(1)
      self._updateInferenceStats(statistics, objectID)

    # send reset signal
    self._sendResetSignal()

    # save statistics
    self.statistics[objectID] = statistics

    if profile:
      self._prettyPrintProfile(source="inference")


  def plotInferenceStats(self,
                         fields,
                         objectID=0,
                         onePlot=True):
    """
    Plots and saves the desired inference statistics.
    :param index:     (int)          Experiment index in self.statistics
    :param keys:      (list(string)) Keys of statistics to plot
    :param path:      (string)       Path to save the plot
    """
    plt.figure(0)
    stats = self.statistics[objectID]
    path = self.PLOT_DIRECTORY + self.name + "_exp_" + str(objectID)

    for i in xrange(self.numColumns):
      if onePlot:
        figIdx = 0
      else:
        figIdx = i

      if not onePlot:
        plt.figure(i)

      # plot request stats
      for field in fields:
        fieldKey = field + " C" + str(i)
        plt.plot(stats[fieldKey], figure=figIdx, marker='+', label=fieldKey)

      # format
      plt.legend(loc="upper right")
      plt.xlabel("Sensation #")
      plt.xticks(range(len(stats[fields])))
      plt.ylabel("Number of active bits")
      plt.ylim(plt.ylim()[0] - 5, plt.ylim()[1] + 5)
      plt.title("Object inference")

      # save
      if not onePlot:
        path = path + "_C" + str(i) + ".png"
        plt.savefig(path)
        plt.close()

    if onePlot:
      path = path + ".png"
      plt.savefig(path)
      plt.close()


  def addObject(self, *args, **kwargs):
    """Adds an object."""
    name = kwargs.pop("name", len(self.objects))

    # check that pairs were not given as a list
    if len(args) == 1 and isinstance(args[0], list):
      return self.addObject(name, *args[0])

    self.objects[name] = [tuple(pair) for pair in args]


  def createRandomObjects(self, numObjects, numPoints):
    """
    Simply creates numObjects, each of them having numPoints feature/location
    pairs.

    The pairs would be drawn randomly, set setObjects() to create personalized
    experiments.
    """
    for _ in xrange(numObjects):
      self.addObject(
        *[(random.randint(0, numPoints),
           random.randint(0, numPoints)) for _ in xrange(numPoints)]
      )


  def generatePattern(self, numBits, size=None):
    """
    Generates a pattern, represented as a set of active bits.

    :param numBits:    (int) Number of on bits.
    :param size:       (int) Total pattern size (defaults to sensorInputSize)
    :return:           (set) Indices of active bits
    """
    if size is None:
      size = self.config["sensorInputSize"]

    cellsIndices = range(size)
    random.shuffle(cellsIndices)
    return set(cellsIndices[:numBits])


  def getL4Representations(self):
    """
    Returns the active representation in L4.
    """
    return [set(column._tm.getActiveCells()) for column in self.L4Columns]


  def getL4PredictiveCells(self):
    """
    Returns the predictive cells in L4.
    """
    return [set(column._tm.getPredictiveCells()) for column in self.L4Columns]


  def getL2Representations(self):
    """
    Returns the active representation in L2.
    """
    return [set(column._pooler.getActiveCells()) for column in self.L2Columns]


  def _unsetLearningMode(self):
    """
    Unsets the learning mode, to start inference.
    """
    for column in self.L4Columns:
      column.setParameter("learningMode", 0, 0)
    for column in self.L2Columns:
      column.setParameter("learningMode", 0, 0)


  def _setLearningMode(self):
    """
    Sets the learning mode, to start inference.
    """
    for column in self.L4Columns:
      column.setParameter("learningMode", 0, 1)
    for column in self.L2Columns:
      column.setParameter("learningMode", 0, 1)


  def _addPointToQueue(self, pairs, reset=False, sequenceId=0, noise=None):
    """
    Adds (feature, location) pairs to the network queue.

    :param pair:         (int, int) Indices of feature and location
    :param reset:        (bool)     If True, a reset signal is sent at the end
    :param sequenceId:   (int)      (Optional) Sequence ID
    :param noise:        (float)    Noise level. Set to None for no noise
    """
    for col in xrange(self.numColumns):
      locationID, featureID = pairs[col]
      feature = self.features[col][featureID]

      # generate random location if requested
      if locationID == -1:
        location = list(self.generatePattern(self.numInputBits,
                                             self.config["sensorInputSize"]))
      # generate union of locations if requested
      elif isinstance(locationID, tuple):
        location = set()
        for idx in list(locationID):
          location = location | self.locations[col][idx]
        location = list(location)
      else:
        location = self.locations[col][locationID]

      if noise is not None:
        location = self._addNoise(location, noise)
        feature = self._addNoise(feature, noise)

      self.sensorInputs[col].addDataToQueue(
        list(feature[col]), int(reset), sequenceId
      )
      self.externalInputs[col].addDataToQueue(
        list(location[col]), int(reset), sequenceId
      )


  def _sendResetSignal(self):
    """
    Sends a reset signal to the network.
    """
    for col in xrange(self.numColumns):
      self.sensorInputs[col].addDataToQueue([], 1, 0)
      self.externalInputs[col].addDataToQueue([], 1, 0)
    self.network.run(1)


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

    return newBits


  def _updateInferenceStats(self, statistics, objectIndex):
    """
    Updates the inference statistics.

    :param statistics:    (dict) Various statistics to save.
    """
    L4Representations = self.getL4Representations()
    L4PredictiveCells = self.getL4PredictiveCells()
    L2Representation = self.getL2Representations()

    objectRepresentation = self.objectL2Representations[objectIndex]
    for i in xrange(self.numColumns):
      statistics["L4 Representation C" + str(i)].append(
        len(L4Representations[i])
      )
      statistics["L4 Predictive C" + str(i)].append(
        len(L4PredictiveCells[i])
      )
      statistics["L2 Representation C" + str(i)].append(
        len(L2Representation[i])
      )
      statistics["Overlap L2 with object C" + str(i)].append(
        len(objectRepresentation & L2Representation)
      )


  def _generateLocations(self, numLocations=400, size=None):
    """
    Generates a pool of locations to be used for the experiments.

    :return:       (list(set))  List of patterns
    """
    if size is None:
      size = self.config["externalInputSize"]

    self.locations = []
    for _ in xrange(self.numColumns):
      self.locations.append(
        [self.generatePattern(self.numInputBits, size) \
         for _ in xrange(numLocations)]
      )

  def _generateFeatures(self, numFeatures=400, size=None):
    """
    Generates a pool of features to be used for the experiments.

    :return:       (list(set))  List of patterns
    """
    if size is None:
      size = self.config["sensorInputSize"]

    self.features = []
    for _ in xrange(self.numColumns):
      self.features.append(
        [self.generatePattern(self.numInputBits, size) \
         for _ in xrange(numFeatures)]
      )


  def _prettyPrintProfile(self, source):
    """
    Prints profiling information.

    :param source:    (string) Source name (e.g "learning" or "inference")
    """
    print "Profiling information for {} in {}".format(
      type(self).__name__, source
    )
    totalTime = 0.000001
    for region in self.network.regions.values():
      timer = region.computeTimer
      totalTime += timer.getElapsed()

    count = 1
    profileInfo = []
    for region in self.network.regions.values():
      timer = region.computeTimer
      count = max(timer.getStartCount(), count)
      profileInfo.append([region.name,
                          timer.getStartCount(),
                          timer.getElapsed(),
                          100.0 * timer.getElapsed() / totalTime,
                          timer.getElapsed() / max(timer.getStartCount(), 1)])

    profileInfo.append(
      ["Total time", "", totalTime, "100.0", totalTime / count])
    print tabulate(profileInfo, headers=["Region", "Count",
                                         "Elapsed", "Pct of total",
                                         "Secs/iteration"],
                   tablefmt="grid", floatfmt="6.3f")
    print ""
