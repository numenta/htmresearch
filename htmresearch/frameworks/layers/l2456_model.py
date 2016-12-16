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
This class allows to easily create experiments using a L2456 network for
inference over objects. It uses the network API and multiple regions (raw
sensors for sensor and external input, column pooler region, extended temporal
memory region).

Here is a sample use of this class, to learn objects and infer one of them. The
object creation details are TBD.

  exp = L2456Model(
    name="sample",
    numCorticalColumns=2,
  )

  # Set up objects (TBD)
  objects = createObjectMachine()

  # Do the learning phase
  exp.learnObjects(objects, reset=True)
  exp.printProfile()

  # Do the inference phase for one object
  exp.infer(objects[0], reset=True)

  exp.plotInferenceStats(
    fields=["L2 Representation",
            "Overlap L2 with object",
            "L4 Representation"],
    plotDir="plots",
  )

"""

import os
import random
import collections
import inspect
import cPickle
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

  exp = L2456Model(*callLog[0][1]["args"], **callLog[0][1]["kwargs"])

  # Call subsequent methods, using stored parameters
  for call in callLog[1:]:
    method = getattr(exp, call[0])
    method(*call[1]["args"], **call[1]["kwargs"])

  return exp


class L2456Model(object):
  """
  L2456 experiment.

  This experiment uses the network API to test out various properties of
  inference and learning using a sensors and an L4-L2 network. For now,
  we directly use the locations on the object.

  """


  @LoggingDecorator()
  def __init__(self,
               name,
               numCorticalColumns=1,
               L2Overrides={},
               L4Overrides={},
               L5Overrides={},
               L6Overrides={},
               numLearningPoints=3,
               seed=42,
               logCalls = False
               ):
    """
    Creates the network.

    Parameters:
    ----------------------------
    @param   name (str)
             Experiment name

    @param   numCorticalColumns (int)
             Number of cortical columns in the network

    @param   L2Overrides (dict)
             Parameters to override in the L2 region

    @param   L4Overrides (dict)
             Parameters to override in the L4 region

    @param   L5Overrides (dict)
             Parameters to override in the L5 region

    @param   L6Overrides (dict)
             Parameters to override in the L6 region

    @param   numLearningPoints (int)
             Number of times each pair should be seen to be learnt

    @param   logCalls (bool)
             If true, calls to main functions will be logged internally. The
             log can then be saved with saveLogs(). This allows us to recreate
             the complete network behavior using rerunExperimentFromLogfile
             which is very useful for debugging.
    """
    # Handle logging - this has to be done first
    self.logCalls = logCalls

    registerAllResearchRegions()
    self.name = name

    self.numLearningPoints = numLearningPoints
    self.numColumns = numCorticalColumns
    self.sensorInputSize = 2048
    self.numInputBits = 40

    # seed
    self.seed = seed
    random.seed(seed)

    # Get network parameters and update with overrides
    self.config = {
      "networkType": "L2456Columns",
      "numCorticalColumns": numCorticalColumns,
      "randomSeedBase": self.seed,
    }
    self.config.update(self.getDefaultParams())

    self.config["L2Params"].update(L2Overrides)
    self.config["L4Params"].update(L4Overrides)
    self.config["L5Params"].update(L5Overrides)
    self.config["L6Params"].update(L6Overrides)

    # create network and retrieve regions
    self.network = createNetwork(self.config)
    self._retrieveRegions()

    # will be populated during training
    self.objectRepresentationsL2 = {}
    self.objectRepresentationsL5 = {}
    self.statistics = []


  @LoggingDecorator()
  def learnObjects(self, objects, reset=True):
    """
    Learns all provided objects, and optionally resets the network.

    The provided objects must have the canonical learning format, which is the
    following.

    objects should be a dict objectName: sensationList, where the sensationList
    is a list of sensations, and each sensation is a mapping from cortical
    column to a tuple of three SDR's respectively corresponding to the
    locationInput, the coarseSensorInput, and the sensorInput.

    The model presents each sensation for numLearningPoints iterations before
    moving on to the next sensation.  Once the network has been trained on an
    object, the L2 and L5 representations for it are stored. A reset signal is
    sent whenever there is a new object if reset=True.

    An example input is as follows, assuming we are learning a simple object
    with a sequence of two sensations (with very few active bits for
    simplicity):

    objects = {
      "simple": [
        {
          # location, coarse feature, fine feature for CC0, sensation 1
          0: ( [1, 5, 10], [9, 32, 75], [6, 12, 52] ),
          # location, coarse feature, fine feature for CC1, sensation 1
          1: ( [6, 2, 15], [11, 42, 92], [7, 11, 50] ),
        },
        {
          # location, coarse feature, fine feature for CC0, sensation 2
          0: ( [2, 9, 10], [10, 35, 78], [6, 12, 52] ),
          # location, coarse feature, fine feature for CC1, sensation 2
          1: ( [1, 4, 12], [10, 32, 52], [6, 10, 52] ),
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
            location, coarseFeature, fineFeature = sensations[col]
            self.locationInputs[col].addDataToQueue(list(location), 0, 0)
            self.coarseSensors[col].addDataToQueue(list(coarseFeature), 0, 0)
            self.sensors[col].addDataToQueue(list(fineFeature), 0, 0)
          iterations += 1

      # actually learn the objects
      if iterations > 0:
        self.network.run(iterations)

      # update L2 and L5 representations for this object
      self.objectRepresentationsL2[objectName] = self.getL2Representations()
      self.objectRepresentationsL5[objectName] = self.getL5Representations()

      if reset:
        # send reset signal
        self._sendReset()


  @LoggingDecorator()
  def infer(self, sensationList, reset=True, objectName=None):
    """
    Infer on a given set of sensations for a single object.

    The provided sensationList is a list of sensations, and each sensation is a
    mapping from cortical column to a tuple of three SDR's respectively
    corresponding to the locationInput, the coarseSensorInput, and the
    sensorInput.

    For example, the input can look as follows, if we are inferring a simple
    object with two sensations (with very few active bits for simplicity):

    sensationList = [
        {
          # location, coarse feature, fine feature for CC0, sensation 1
          0: ( [1, 5, 10], [9, 32, 75], [6, 12, 52] ),
          # location, coarse feature, fine feature for CC1, sensation 1
          1: ( [6, 2, 15], [11, 42, 92], [7, 11, 50] ),
        },
        {
          # location, coarse feature, fine feature for CC0, sensation 2
          0: ( [2, 9, 10], [10, 35, 78], [6, 12, 52] ),
          # location, coarse feature, fine feature for CC1, sensation 2
          1: ( [1, 4, 12], [10, 32, 52], [6, 10, 52] ),
        },
    ]

    If the object is known by the caller, an object name can be specified
    as an optional argument, and must match the objects given while learning.
    This is used later when evaluating inference statistics.

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
      if objectName not in self.objectRepresentationsL2:
        raise ValueError("The provided objectName was not given during"
                         " learning")

    for sensations in sensationList:

      # feed all columns with sensations
      for col in xrange(self.numColumns):
        location, coarseFeature, fineFeature = sensations[col]
        self.locationInputs[col].addDataToQueue(list(location), 0, 0)
        self.coarseSensors[col].addDataToQueue(list(coarseFeature), 0, 0)
        self.sensors[col].addDataToQueue(list(fineFeature), 0, 0)
      self.network.run(1)
      self._updateInferenceStats(statistics, objectName)

    if reset:
      # send reset signal
      self._sendReset()

    # save statistics
    statistics["numSteps"] = len(sensationList)
    statistics["object"] = objectName if objectName is not None else "Unknown"
    self.statistics.append(statistics)


  @LoggingDecorator()
  def sendReset(self, *args, **kwargs):
    """
    Public interface to sends a reset signal to the network.  This is logged.
    """
    self._sendReset(*args, **kwargs)


  def _sendReset(self, sequenceId=0):
    """
    Sends a reset signal to the network.
    """
    # Handle logging - this has to be done first
    if self.logCalls:
      frame = inspect.currentframe()
      args, _, _, values = inspect.getargvalues(frame)
      values.pop('frame')
      values.pop('self')
      (_, filename,
       _, _, _, _) = inspect.getouterframes(inspect.currentframe())[1]
      if os.path.splitext(os.path.basename(__file__))[0] != \
         os.path.splitext(os.path.basename(filename))[0]:
        self.callLog.append([inspect.getframeinfo(frame)[2], values])

    for col in xrange(self.numColumns):
      self.locationInputs[col].addResetToQueue(sequenceId)
      self.coarseSensors[col].addResetToQueue(sequenceId)
      self.sensors[col].addResetToQueue(sequenceId)
    self.network.run(1)


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
    Returns a list of active cells in L2 for each column.
    """
    return [set(column._pooler.getActiveCells()) for column in self.L2Columns]


  def getL5Representations(self):
    """
    Returns a list of active cells in L5 for each column.
    """
    return [set(column._pooler.getActiveCells()) for column in self.L5Columns]


  def getL6Representations(self):
    """
    Returns the active representation in L4.
    """
    return [set(column._tm.getActiveCells()) for column in self.L6Columns]


  def getL6PredictiveCells(self):
    """
    Returns the predictive cells in L4.
    """
    return [set(column._tm.getPredictiveCells()) for column in self.L6Columns]


  def getDefaultParams(self):
    """
    Returns a good default set of parameters to use in L2456 regions
    """
    return   {
      "sensorParams": {
        "outputWidth": self.sensorInputSize,
      },

      "coarseSensorParams": {
        "outputWidth": self.sensorInputSize,
      },

      "locationParams": {
        "activeBits": 41,
        "outputWidth": self.sensorInputSize,
        "radius": 2,
        "verbosity": 0,
      },

      "L4Params": {
        "columnCount": self.sensorInputSize,
        "cellsPerColumn": 8,
        "formInternalBasalConnections": False,
        "learningMode": True,
        "inferenceMode": True,
        "learnOnOneCell": False,
        "initialPermanence": 0.51,
        "connectedPermanence": 0.6,
        "permanenceIncrement": 0.1,
        "permanenceDecrement": 0.02,
        "minThreshold": 10,
        "predictedSegmentDecrement": 0.002,
        "activationThreshold": 13,
        "maxNewSynapseCount": 20,
        "defaultOutputType": "predictedActiveCells",
        "implementation": "etm_cpp",
      },

      "L2Params": {
        "inputWidth": self.sensorInputSize * 8,
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
        "learningMode": True,
      },

      "L6Params": {
        "columnCount": self.sensorInputSize,
        "cellsPerColumn": 8,
        "formInternalBasalConnections": False,
        "learningMode": True,
        "inferenceMode": True,
        "learnOnOneCell": False,
        "initialPermanence": 0.51,
        "connectedPermanence": 0.6,
        "permanenceIncrement": 0.1,
        "permanenceDecrement": 0.02,
        "minThreshold": 10,
        "predictedSegmentDecrement": 0.004,
        "activationThreshold": 13,
        "maxNewSynapseCount": 20,
      },

      "L5Params": {
        "inputWidth": self.sensorInputSize * 8,
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
        "learningMode": True,
      },

    }


  def _retrieveRegions(self):
    """
    Retrieve and store Python region instances for each column
    """
    self.sensors = []
    self.coarseSensors = []
    self.locationInputs = []
    self.L4Columns = []
    self.L2Columns = []
    self.L5Columns = []
    self.L6Columns = []

    for i in xrange(self.numColumns):
      self.sensors.append(
        self.network.regions["sensorInput_" + str(i)].getSelf()
      )
      self.coarseSensors.append(
        self.network.regions["coarseSensorInput_" + str(i)].getSelf()
      )
      self.locationInputs.append(
        self.network.regions["locationInput_" + str(i)].getSelf()
      )
      self.L4Columns.append(
        self.network.regions["L4Column_" + str(i)].getSelf()
      )
      self.L2Columns.append(
        self.network.regions["L2Column_" + str(i)].getSelf()
      )
      self.L5Columns.append(
        self.network.regions["L5Column_" + str(i)].getSelf()
      )
      self.L6Columns.append(
        self.network.regions["L6Column_" + str(i)].getSelf()
      )


  def _unsetLearningMode(self):
    """
    Unsets the learning mode, to start inference.
    """
    for column in self.L4Columns + self.L2Columns + \
                  self.L5Columns + self.L6Columns:
      column.setParameter("learningMode", 0, False)


  def _setLearningMode(self):
    """
    Sets the learning mode.
    """
    for column in self.L4Columns + self.L2Columns + \
                  self.L5Columns + self.L6Columns:
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
    L2Representations = self.getL2Representations()
    L5Representations = self.getL5Representations()
    L6Representations = self.getL6Representations()
    L6PredictiveCells = self.getL6PredictiveCells()

    for i in xrange(self.numColumns):
      statistics["L4 Representation C" + str(i)].append(
        len(L4Representations[i])
      )
      statistics["L4 Predictive C" + str(i)].append(
        len(L4PredictiveCells[i])
      )
      statistics["L2 Representation C" + str(i)].append(
        len(L2Representations[i])
      )

      statistics["L6 Representation C" + str(i)].append(
        len(L6Representations[i])
      )
      statistics["L6 Predictive C" + str(i)].append(
        len(L6PredictiveCells[i])
      )
      statistics["L5 Representation C" + str(i)].append(
        len(L5Representations[i])
      )

      # add true overlap if objectName was provided
      if objectName is not None:
        objectRepresentationL2 = self.objectRepresentationsL2[objectName]
        statistics["Overlap L2 with object C" + str(i)].append(
          len(objectRepresentationL2[i] & L2Representations[i])
        )

        objectRepresentationL5 = self.objectRepresentationsL5[objectName]
        statistics["Overlap L5 with object C" + str(i)].append(
          len(objectRepresentationL5[i] & L5Representations[i])
        )

