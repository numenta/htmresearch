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
This class allows to easily create experiments using a L4-L2 network for
inference over objects. It uses the network API and multiple regions (raw
sensors for sensor and external input, column pooler region, extended temporal
memory region).

Here is a sample use of this class, to learn two very simple objects
and infer one of them:

  exp = L4L2Experiment(
    name="sample"
    numCorticalColumns=2,
  )

  objects = exp.addObject([(1, 2), (2, 3)], name=0, objects={})
  objects = exp.addObject([(1, 2), (4, 5)], name=1, objects=objects)

  exp.learnObjects(objects)
  exp.printProfile()

  inferConfig = {
    "object": 0,
    "numSteps": 2,
    "pairs": {
      0: [(1, 2), (2, 3)]
      1: [(2, 3), (1, 2)]
    }
  }

  exp.infer(inferConfig, noise=0.05)
  exp.plotInferenceStats(
    fields=["L2 Representation",
            "Overlap L2 with object",
            "L4 Representation"],
  )

More examples are available in projects/layers/single_column_l2l4.py and
projects/layers/multi_column_l2l4.py

"""

import os
import random
import collections
import matplotlib.pyplot as plt
from tabulate import tabulate

from htmresearch.support.register_regions import registerAllResearchRegions
from htmresearch.frameworks.layers.laminar_network import createNetwork



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
               numCorticalColumns=1,
               inputSize=1024,
               numInputBits=20,
               externalInputSize=1024,
               L2Overrides=None,
               L4Overrides=None,
               numLearningPoints=3,
               seed=42):
    """
    Creates the network.

    Parameters:
    ----------------------------
    @param   name (str)
             Experiment name

    @param   numCorticalColumns (int)
             Number of cortical columns in the network

    @param   inputSize (int)
             Size of the sensory input

    @param   numInputBits (int)
             Number of ON bits in the generated input patterns

    @param   externalInputSize (int)
             Size of the lateral input to L4 regions

    @param   L2Overrides (dict)
             Parameters to override in the L2 region

    @param   L4Overrides
             Parameters to override in the L4 region

    @param   numLearningPoints (int)
             Number of times each pair should be seen to be learnt

    """
    registerAllResearchRegions()
    self.name = name

    self.numLearningPoints = numLearningPoints
    self.numColumns = numCorticalColumns
    self.inputSize = inputSize
    self.externalInputSize = externalInputSize
    self.numInputBits = numInputBits
    random.seed(seed)

    # update parameters with overrides
    self.config = {
      "networkType": "MultipleL4L2Columns",
      "numCorticalColumns": numCorticalColumns,
      "externalInputSize": externalInputSize,
      "sensorInputSize": inputSize,
      "L4Params": self.getDefaultL4Params(inputSize),
      "L2Params": self.getDefaultL2Params(inputSize),
    }

    if L2Overrides is not None:
      self.config["L2Params"].update(L2Overrides)

    if L4Overrides is not None:
      self.config["L4Params"].update(L4Overrides)

    # create network
    self.network = createNetwork(self.config)

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

    self._generateLocations()
    self._generateFeatures()

    # will be populated during training
    self.objectL2Representations = {}
    self.statistics = []

    if not os.path.exists(self.PLOT_DIRECTORY):
      os.makedirs(self.PLOT_DIRECTORY)


  def learnObjects(self, objects):
    """
    Learns all objects in objects.
    """
    for object, pairs in objects.iteritems():
      iterations = 0
      if len(pairs) == 0:
        continue

      for pair in pairs:
        for _ in xrange(self.numLearningPoints):
          # train all columns
          # note: since their only link is in the pooled layer, the joint
          # order in which the pairs are fed does not matter
          self._addPointToQueue([pair for _ in xrange(self.numColumns)])
          iterations += 1

      # actually learn the objects
      if iterations > 0:
        self.network.run(iterations)

      # update L2 representations
      self.objectL2Representations[object] = self.getL2Representations()

      # send reset signal
      self.sendResetSignal()


  def infer(self, inferenceConfig, noise=None):
    """
    Go over a list of locations / features and tries to recognize an object.

    It updates various statistics as it goes.
    inferenceConfig should have the following format:

    inferConfig = {
      "object": 0  # objectID,
      "numSteps": 2  # number of inference steps,
      "pairs": {  # for each cortical column, list of pairs it will sense
        0: [(1, 2), (2, 3)]
        1: [(2, 3), (1, 2)]
      }
    }

    An additional parameter, giving a noise level to add to the sensed
    patterns, can be given.
    """
    self._unsetLearningMode()
    self.sendResetSignal()

    statistics = collections.defaultdict(list)
    objectID = inferenceConfig["object"]
    if "numSteps" in inferenceConfig:
      numSteps = inferenceConfig["numSteps"]
    else:
      numSteps = len(inferenceConfig["pairs"][0])

    # some checks
    if numSteps == 0:
      raise ValueError("No inference steps were provided")
    for col in xrange(self.numColumns):
      if len(inferenceConfig["pairs"][col]) != numSteps:
        raise ValueError("Incompatible numSteps and actual inference steps")

    for step in xrange(numSteps):
      pairs = [inferenceConfig["pairs"][col][step] \
               for col in xrange(self.numColumns)]
      self._addPointToQueue(pairs, noise=noise)
      self.network.run(1)
      self._updateInferenceStats(statistics, objectID)

    # send reset signal
    self.sendResetSignal()

    # save statistics
    statistics["numSteps"] = numSteps
    statistics["object"] = objectID
    self.statistics.append(statistics)


  def plotInferenceStats(self,
                         fields,
                         experimentID=0,
                         onePlot=True):
    """
    Plots and saves the desired inference statistics.

    Parameters:
    ----------------------------
    @param   fields (list(str))
             List of fields to include in the plots

    @param   experimentID (int)
             ID of the experiment (usually 0 if only one was conducted)

    @param   onePlot (bool)
             If true, all cortical columns will be merged in one plot.

    """
    plt.figure(0)
    stats = self.statistics[experimentID]
    objectID = stats["object"]
    initPath = self.PLOT_DIRECTORY + self.name + "_exp_" + str(experimentID)

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
      plt.xticks(range(stats["numSteps"]))
      plt.ylabel("Number of active bits")
      plt.ylim(plt.ylim()[0] - 5, plt.ylim()[1] + 5)
      plt.title("Object inference for object {}".format(objectID))

      # save
      if not onePlot:
        path = initPath + "_C" + str(i) + ".png"
        plt.savefig(path)
        plt.close()

    if onePlot:
      path = initPath + ".png"
      plt.savefig(path)
      plt.close()


  def getInferenceStats(self, experimentID):
    """
    Returns the statistics for the desired experiment.
    """
    return self.statistics[experimentID]


  @staticmethod
  def addObject(pairs, name=None, objects=None):
    """
    Adds an object to learn (giving the list of pairs of location, feature
    indices.

    A name can be given to the object, otherwise it will be incrementally
    indexed.
    """
    # TODO: pull out of class as part of generic object handling (RES-351)
    if objects is None:
      objects = {}

    if name is None:
      name = len(objects)

    objects[name] = pairs
    return objects


  def printProfile(self, reset=False):
    """
    Prints profiling information.
    """
    print "Profiling information for {}".format(type(self).__name__)
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
    print

    if reset:
      self.resetProfile()


  def resetProfile(self):
    """
    Resets the network profiling.
    """
    self.network.resetProfiling()


  @classmethod
  def createRandomObjects(cls, numObjects, numPoints,
                          numLocations=None, numFeatures=None):
    """
    Create numObjects, each with numPoints random location/feature pairs.

    @param  numObjects (int)
            The number of objects we are creating.

    @param  numPoints (int)
            The number of location/feature points per object.

    @param  numLocations (int or None)
            Each location index is chosen randomly from numLocations possible
            locations. If None, defaults to numPoints

    @param  numFeatures (int or None)
            Each feature index is chosen randomly from numFeatures possible
            locations. If None, defaults to numPoints

    The pairs would be drawn randomly, set setObjects() to create personalized
    experiments.
    """
    # TODO: pull out of class as part of generic object handling (RES-351)

    if numLocations is None:
      numLocations = numPoints
    if numFeatures is None:
      numFeatures = numPoints

    objects = {}
    for _ in xrange(numObjects):
      cls.addObject(
        [(random.randint(0, numLocations),
          random.randint(0, numFeatures)) for _ in xrange(numPoints)],
        objects=objects
      )

    return objects


  def generatePattern(self, numBits, size=None):
    """
    Generates a pattern, represented as a set of active bits.
    """
    if size is None:
      size = self.inputSize

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


  def getDefaultL4Params(self, inputSize):
    """
    Returns a good default set of parameters to use in the L4 region.
    """
    return {
      "columnCount": inputSize,
      "cellsPerColumn": 8,
      "formInternalConnections": 0,
      "formInternalBasalConnections": 0,  # inconsistency between CPP and PY
      "learningMode": 1,
      "inferenceMode": 1,
      "learnOnOneCell": 0,
      "initialPermanence": 0.51,
      "connectedPermanence": 0.6,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.02,
      "minThreshold": 10,
      "predictedSegmentDecrement": 0.002,
      "activationThreshold": 13,
      "maxNewSynapseCount": 20,
      "monitor": 0,
      "implementation": "cpp",
    }


  def getDefaultL2Params(self, inputSize):
    """
    Returns a good default set of parameters to use in the L4 region.
    """
    return {
      "columnCount": 1024,
      "inputWidth": inputSize * 8,
      "learningMode": 1,
      "inferenceMode": 1,
      "initialPermanence": 0.41,
      "connectedPermanence": 0.5,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.02,
      "numActiveColumnsPerInhArea": 40,
      "synPermProximalInc": 0.1,
      "synPermProximalDec": 0.001,
      "initialProximalPermanence": 0.6,
      "minThreshold": 10,
      "predictedSegmentDecrement": 0.002,
      "activationThreshold": 13,
      "maxNewSynapseCount": 20,
    }


  def sendResetSignal(self, sequenceId=0):
    """
    Sends a reset signal to the network.
    """
    for col in xrange(self.numColumns):
      self.sensorInputs[col].addResetToQueue(sequenceId)
      self.externalInputs[col].addResetToQueue(sequenceId)
    self.network.run(1)


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
    Sets the learning mode.
    """
    for column in self.L4Columns:
      column.setParameter("learningMode", 0, 1)
    for column in self.L2Columns:
      column.setParameter("learningMode", 0, 1)


  def _addPointToQueue(self, pairs, reset=False, sequenceId=0, noise=None):
    """
    Adds (feature, location) pairs to the network queue.

    Parameters:
    ----------------------------
    @param  pair list((int, int))
            List of indices of feature and location (one per column)

    @param  reset (bool)
            If True, a reset signal is sent after the pair

    @param  sequenceId (int)
            (Optional) Sequence ID

    @param  noise (float)
            Noise level. Set to None for no noise

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
        list(feature), int(reset), sequenceId
      )
      self.externalInputs[col].addDataToQueue(
        list(location), int(reset), sequenceId
      )


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


  def _updateInferenceStats(self, statistics, objectID):
    """
    Updates the inference statistics.
    """
    L4Representations = self.getL4Representations()
    L4PredictiveCells = self.getL4PredictiveCells()
    L2Representation = self.getL2Representations()

    objectRepresentation = self.objectL2Representations[objectID]
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
        len(objectRepresentation[i] & L2Representation[i])
      )


  def _generateLocations(self, numLocations=400, size=None):
    """
    Generates a pool of locations to be used for the experiments.
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
    """
    if size is None:
      size = self.config["sensorInputSize"]

    self.features = []
    for _ in xrange(self.numColumns):
      self.features.append(
        [self.generatePattern(self.numInputBits, size) \
         for _ in xrange(numFeatures)]
      )
