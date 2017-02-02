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
and infer one of them. In this case, we use a SimpleObjectMachine to generate
objects. If no object machine is used, objects and sensations should be passed
in a very specific format (cf. learnObjects() and infer() for more
information).

  exp = L4L2Experiment(
    name="sample",
    numCorticalColumns=2,
  )

  # Set up inputs for learning
  objects = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=1024,
    externalInputSize=1024,
    numCorticalColumns=2,
  )
  objects.addObject([(1, 2), (2, 3)], name=0)
  objects.addObject([(1, 2), (4, 5)], name=1)
  objectsToLearn = objects.provideObjectsToLearn()

  # Do the learning phase
  exp.learnObjects(objectsToLearn, reset=True)
  exp.printProfile()

  # Set up inputs for inference
  inferConfig = {
    "numSteps": 2,
    "noiseLevel": 0.05,
    "pairs": {
      0: [(1, 2), (2, 3)],
      1: [(2, 3), (1, 2)],
    }
  }
  objectsToInfer = objects.provideObjectToInfer(inferConfig)

  # Do the inference phase
  exp.infer(objectsToInfer,
            objectName=0, reset=True)

  exp.plotInferenceStats(
    fields=["L2 Representation",
            "Overlap L2 with object",
            "L4 Representation"],
    plotDir="plots",
  )

More examples are available in projects/layers/single_column.py and
projects/layers/multi_column.py

"""

import collections
import os
import random
import matplotlib.pyplot as plt
from tabulate import tabulate

from htmresearch.support.logging_decorator import LoggingDecorator
from htmresearch.support.register_regions import registerAllResearchRegions
from htmresearch.frameworks.layers.laminar_network import createNetwork



def rerunExperimentFromLogfile(logFilename):
  """
  Create an experiment class according to the sequence of operations in logFile
  and return resulting experiment instance.
  """
  callLog = LoggingDecorator.load(logFilename)

  # Assume first one is call to constructor

  exp = L4L2Experiment(*callLog[0][1]["args"], **callLog[0][1]["kwargs"])

  # Call subsequent methods, using stored parameters
  for call in callLog[1:]:
    method = getattr(exp, call[0])
    method(*call[1]["args"], **call[1]["kwargs"])

  return exp



class L4L2Experiment(object):
  """
  L4-L2 experiment.

  This experiment uses the network API to test out various properties of
  inference and learning using a sensors and an L4-L2 network. For now,
  we directly use the locations on the object.

  """


  @LoggingDecorator()
  def __init__(self,
               name,
               numCorticalColumns=1,
               inputSize=1024,
               numInputBits=20,
               externalInputSize=1024,
               L2Overrides=None,
               L4Overrides=None,
               numLearningPoints=3,
               seed=42,
               logCalls=False,
               enableLateralSP=False,
               lateralSPOverrides=None,
               enableFeedForwardSP=False,
               feedForwardSPOverrides=None
               ):
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

    @param   logCalls (bool)
             If true, calls to main functions will be logged internally. The
             log can then be saved with saveLogs(). This allows us to recreate
             the complete network behavior using rerunExperimentFromLogfile
             which is very useful for debugging.

    @param   enableLateralSP (bool)
             If true, Spatial Pooler will be added between external input and
             L4 lateral input

    @param   lateralSPOverrides
             Parameters to override in the lateral SP region

    @param   enableFeedForwardSP (bool)
             If true, Spatial Pooler will be added between external input and
             L4 feed-forward input

    @param   feedForwardSPOverrides
             Parameters to override in the feed-forward SP region

    """
    # Handle logging - this has to be done first
    self.logCalls = logCalls

    registerAllResearchRegions()
    self.name = name

    self.numLearningPoints = numLearningPoints
    self.numColumns = numCorticalColumns
    self.inputSize = inputSize
    self.externalInputSize = externalInputSize
    self.numInputBits = numInputBits

    # seed
    self.seed = seed
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

    if enableLateralSP:
      self.config["lateralSPParams"] = self.getDefaultLateralSPParams(inputSize)
      if lateralSPOverrides:
        self.config["lateralSPParams"].update(lateralSPOverrides)

    if enableFeedForwardSP:
      self.config["feedForwardSPParams"] = self.getDefaultFeedForwardSPParams(inputSize)
      if feedForwardSPOverrides:
        self.config["feedForwardSPParams"].update(feedForwardSPOverrides)

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

    # will be populated during training
    self.objectL2Representations = {}
    self.statistics = []


  @LoggingDecorator()
  def learnObjects(self, objects, reset=True):
    """
    Learns all provided objects, and optionally resets the network.

    The provided objects must have the canonical learning format, which is the
    following.
    objects should be a dict objectName: sensationList, where each
    sensationList is a list of sensations, and each sensation is a mapping
    from cortical column to a tuple of two SDR's respectively corresponding
    to the location in object space and the feature.

    For example, the input can look as follows, if we are learning a simple
    object with two sensations (with very few active bits for simplicity):

    objects = {
      "simple": [
        {
          0: (set([1, 5, 10]), set([6, 12, 52]),  # location, feature for CC0
          1: (set([6, 2, 15]), set([64, 1, 5]),  # location, feature for CC1
        },
        {
          0: (set([5, 46, 50]), set([8, 10, 11]),  # location, feature for CC0
          1: (set([1, 6, 45]), set([12, 17, 23]),  # location, feature for CC1
        },
      ]
    }

    In many uses cases, this object can be created by implementations of
    ObjectMachines (cf htm.research.object_machine_factory), through their
    method providedObjectsToLearn.

    Parameters:
    ----------------------------
    @param   objects (dict)
             Objects to learn, in the canonical format specified above

    @param   reset (bool)
             If set to True (which is the default value), the network will
             be reset after learning.

    """
    self._setLearningMode()

    for objectName, sensationList in objects.iteritems():

      # ignore empty sensation lists
      if len(sensationList) == 0:
        continue

      # keep track of numbers of iterations to run
      iterations = 0

      for sensations in sensationList:
        # learn each pattern multiple times
        for _ in xrange(self.numLearningPoints):

          for col in xrange(self.numColumns):
            location, feature = sensations[col]
            self.sensorInputs[col].addDataToQueue(list(feature), 0, 0)
            self.externalInputs[col].addDataToQueue(list(location), 0, 0)
          iterations += 1

      # actually learn the objects
      if iterations > 0:
        self.network.run(iterations)

      # update L2 representations
      self.objectL2Representations[objectName] = self.getL2Representations()

      if reset:
        # send reset signal
        self._sendReset()

  @LoggingDecorator()
  def infer(self, sensationList, reset=True, objectName=None):
    """
    Infer on given sensations.

    The provided sensationList is a list of sensations, and each sensation is
    a mapping from cortical column to a tuple of two SDR's respectively
    corresponding to the location in object space and the feature.

    For example, the input can look as follows, if we are inferring a simple
    object with two sensations (with very few active bits for simplicity):

    sensationList = [
      {
        0: (set([1, 5, 10]), set([6, 12, 52]),  # location, feature for CC0
        1: (set([6, 2, 15]), set([64, 1, 5]),  # location, feature for CC1
      },

      {
        0: (set([5, 46, 50]), set([8, 10, 11]),  # location, feature for CC0
        1: (set([1, 6, 45]), set([12, 17, 23]),  # location, feature for CC1
      },
    ]

    In many uses cases, this object can be created by implementations of
    ObjectMachines (cf htm.research.object_machine_factory), through their
    method providedObjectsToInfer.

    If the object is known by the caller, an object name can be specified
    as an optional argument, and must match the objects given while learning.

    Parameters:
    ----------------------------
    @param   objects (dict)
             Objects to learn, in the canonical format specified above

    @param   reset (bool)
             If set to True (which is the default value), the network will
             be reset after learning.

    @param   objectName (str)
             Name of the objects (must match the names given during learning).

    """
    self._unsetLearningMode()
    statistics = collections.defaultdict(list)

    if objectName is not None:
      if objectName not in self.objectL2Representations:
        raise ValueError("The provided objectName was not given during"
                         " learning")

    for sensations in sensationList:

      # feed all columns with sensations
      for col in xrange(self.numColumns):
        location, feature = sensations[col]
        self.sensorInputs[col].addDataToQueue(list(feature), 0, 0)
        self.externalInputs[col].addDataToQueue(list(location), 0, 0)
      self.network.run(1)
      self._updateInferenceStats(statistics, objectName)

    if reset:
      # send reset signal
      self._sendReset()

    # save statistics
    statistics["numSteps"] = len(sensationList)
    statistics["object"] = objectName if objectName is not None else "Unknown"
    self.statistics.append(statistics)

  def _sendReset(self, sequenceId=0):
    """
    Sends a reset signal to the network.
    """
    for col in xrange(self.numColumns):
      self.sensorInputs[col].addResetToQueue(sequenceId)
      self.externalInputs[col].addResetToQueue(sequenceId)
    self.network.run(1)

  @LoggingDecorator()
  def sendReset(self, *args, **kwargs):
    """
    Public interface to sends a reset signal to the network.  This is logged.
    """
    self._sendReset(*args, **kwargs)

  def plotInferenceStats(self,
                         fields,
                         plotDir="plots",
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
    if not os.path.exists(plotDir):
      os.makedirs(plotDir)

    plt.figure()
    stats = self.statistics[experimentID]
    objectName = stats["object"]

    for i in xrange(self.numColumns):
      if not onePlot:
        plt.figure()

      # plot request stats
      for field in fields:
        fieldKey = field + " C" + str(i)
        plt.plot(stats[fieldKey], marker='+', label=fieldKey)

      # format
      plt.legend(loc="upper right")
      plt.xlabel("Sensation #")
      plt.xticks(range(stats["numSteps"]))
      plt.ylabel("Number of active bits")
      plt.ylim(plt.ylim()[0] - 5, plt.ylim()[1] + 5)
      plt.title("Object inference for object {}".format(objectName))

      # save
      if not onePlot:
        relPath = "{}_exp_{}_C{}.png".format(self.name, experimentID, i)
        path = os.path.join(plotDir, relPath)
        plt.savefig(path)
        plt.close()

    if onePlot:
      relPath = "{}_exp_{}.png".format(self.name, experimentID)
      path = os.path.join(plotDir, relPath)
      plt.savefig(path)
      plt.close()


  def getInferenceStats(self, experimentID=None):
    """
    Returns the statistics for the desired experiment. If experimentID is None
    return all statistics

    Parameters:
    ----------------------------
    @param   experimentID (int)
             ID of the experiment (usually 0 if only one was conducted)

    """
    if experimentID is None:
      return self.statistics
    else:
      return self.statistics[experimentID]


  def printProfile(self, reset=False):
    """
    Prints profiling information.

    Parameters:
    ----------------------------
    @param   reset (bool)
             If set to True, the profiling will be reset.

    """
    print "Profiling information for {}".format(type(self).__name__)
    totalTime = 0.000001
    for region in self.network.regions.values():
      timer = region.getComputeTimer()
      totalTime += timer.getElapsed()

    # Sort the region names
    regionNames = list(self.network.regions.keys())
    regionNames.sort()

    count = 1
    profileInfo = []
    L2Time = 0.0
    L4Time = 0.0
    for regionName in regionNames:
      region = self.network.regions[regionName]
      timer = region.getComputeTimer()
      count = max(timer.getStartCount(), count)
      profileInfo.append([region.name,
                          timer.getStartCount(),
                          timer.getElapsed(),
                          100.0 * timer.getElapsed() / totalTime,
                          timer.getElapsed() / max(timer.getStartCount(), 1)])
      if "L2Column" in regionName:
        L2Time += timer.getElapsed()
      elif "L4Column" in regionName:
        L4Time += timer.getElapsed()

    profileInfo.append(
      ["Total time", "", totalTime, "100.0", totalTime / count])
    print tabulate(profileInfo, headers=["Region", "Count",
                                         "Elapsed", "Pct of total",
                                         "Secs/iteration"],
                   tablefmt="grid", floatfmt="6.3f")
    print
    print "Total time in L2 =", L2Time
    print "Total time in L4 =", L4Time

    if reset:
      self.resetProfile()


  def resetProfile(self):
    """
    Resets the network profiling.
    """
    self.network.resetProfiling()


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
      "formInternalBasalConnections": False,
      "learningMode": True,
      "inferenceMode": True,
      "learnOnOneCell": False,
      "initialPermanence": 0.51,
      "connectedPermanence": 0.6,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.02,
      "minThreshold": 13,
      "predictedSegmentDecrement": 0.0,
      "activationThreshold": 13,
      "maxNewSynapseCount": 30,
      "defaultOutputType": "predictedActiveCells",
      "implementation": "etm_cpp",
      "seed": self.seed
    }


  def getDefaultL2Params(self, inputSize):
    """
    Returns a good default set of parameters to use in the L2 region.
    """
    return {
      "inputWidth": inputSize * 8,
      "cellCount": 4096,
      "sdrSize": 40,
      "synPermProximalInc": 0.1,
      "synPermProximalDec": 0.001,
      "initialProximalPermanence": 0.6,
      "minThresholdProximal": 10,
      "sampleSizeProximal": 20,
      "connectedPermanenceProximal": 0.5,
      "synPermDistalInc": 0.1,
      "synPermDistalDec": 0.001,
      "initialDistalPermanence": 0.41,
      "activationThresholdDistal": 13,
      "sampleSizeDistal": 20,
      "connectedPermanenceDistal": 0.5,
      "distalSegmentInhibitionFactor": 1.5,
      "seed": self.seed,
      "learningMode": True,
    }

  def getDefaultLateralSPParams(self, inputSize):
    return {
      "spatialImp": "cpp",
      "globalInhibition": 1,
      "columnCount": 1024,
      "inputWidth": inputSize,
      "numActiveColumnsPerInhArea": 40,
      "seed": self.seed,
      "potentialPct": 0.8,
      "synPermConnected": 0.1,
      "synPermActiveInc": 0.0001,
      "synPermInactiveDec": 0.0005,
      "boostStrength": 0.0,
    }

  def getDefaultFeedForwardSPParams(self, inputSize):
    return {
      "spatialImp": "cpp",
      "globalInhibition": 1,
      "columnCount": 1024,
      "inputWidth": inputSize,
      "numActiveColumnsPerInhArea": 40,
      "seed": self.seed,
      "potentialPct": 0.8,
      "synPermConnected": 0.1,
      "synPermActiveInc": 0.0001,
      "synPermInactiveDec": 0.0005,
      "boostStrength": 0.0,
    }


  def _unsetLearningMode(self):
    """
    Unsets the learning mode, to start inference.
    """
    for column in self.L4Columns:
      column.setParameter("learningMode", 0, False)
    for column in self.L2Columns:
      column.setParameter("learningMode", 0, False)


  def _setLearningMode(self):
    """
    Sets the learning mode.
    """
    for column in self.L4Columns:
      column.setParameter("learningMode", 0, True)
    for column in self.L2Columns:
      column.setParameter("learningMode", 0, True)


  def _updateInferenceStats(self, statistics, objectName=None):
    """
    Updates the inference statistics.

    Parameters:
    ----------------------------
    @param  statistics (dict)
            Dictionary in which to write the statistics

    @param  objectName (str)
            Name of the inferred object, if known. Otherwise, set to None.

    """
    L4Representations = self.getL4Representations()
    L4PredictiveCells = self.getL4PredictiveCells()
    L2Representation = self.getL2Representations()

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

      # add true overlap if objectName was provided
      if objectName is not None:
        objectRepresentation = self.objectL2Representations[objectName]
        statistics["Overlap L2 with object C" + str(i)].append(
          len(objectRepresentation[i] & L2Representation[i])
        )
