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
This class implements an L4-L2 network.

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
# Disable variable/field name restrictions
# pylint: disable=C0103

import collections
import os
import random
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

from nupic.bindings.algorithms import SpatialPooler

from htmresearch_core.experimental import ExtendedTemporalMemory

from htmresearch.algorithms.column_pooler import ColumnPooler
from htmresearch.support.logging_decorator import LoggingDecorator



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
  This class implements an L4-L2 network.

  We use it to test out various properties of inference and learning using a
  sensors and an L4-L2 network. For now, we directly use the locations on the
  object.

  """


  @LoggingDecorator()
  def __init__(self,
               name,
               numCorticalColumns=1,
               inputSize=1024,
               numInputBits=20,
               externalInputSize=1024,
               numExternalInputBits=20,
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

    @param   numExternalInputBits (int)
             Number of ON bits in the external input patterns

    @param   L2Overrides (dict)
             Parameters to override in the L2 region

    @param   L4RegionType (string)
             The type of region to use for L4

    @param   L4Overrides (dict)
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

    self.name = name

    self.numLearningPoints = numLearningPoints
    self.numColumns = numCorticalColumns
    self.inputSize = inputSize
    self.externalInputSize = externalInputSize
    self.numInputBits = numInputBits

    # seed
    self.seed = seed
    random.seed(seed)

    # SP on L4's sensory input
    if enableFeedForwardSP:
      SPParams = self.getDefaultFeedForwardSPParams(inputSize)
      if feedForwardSPOverrides:
        SPParams.update(feedForwardSPOverrides)

      self.L4FeedforwardSPs = [SpatialPooler(**SPParams)
                               for _ in xrange(numCorticalColumns)]
    else:
      self.L4FeedforwardSPs = None

    # SP on L4's location input
    if enableLateralSP:
      SPParams = self.getDefaultLateralSPParams(externalInputSize)
      if lateralSPOverrides:
        SPParams.update(lateralSPOverrides)

      self.L4LateralSPs = [SpatialPooler(**SPParams)
                           for _ in xrange(numCorticalColumns)]
    else:
      self.L4LateralSPs = None


    # L4
    self.L4Params = self.getDefaultL4Params(inputSize, externalInputSize,
                                            numExternalInputBits)
    if L4Overrides is not None:
      self.L4Params.update(L4Overrides)
    self.L4Columns = [ExtendedTemporalMemory(**self.L4Params)
                      for _ in xrange(numCorticalColumns)]


    # L2
    self.L2Params = self.getDefaultL2Params(numCorticalColumns, inputSize,
                                            numInputBits)
    if L2Overrides is not None:
      self.L2Params.update(L2Overrides)
    self.L2Columns = [ColumnPooler(**self.L2Params)
                      for _ in xrange(numCorticalColumns)]

    # will be populated during training
    self.objectL2Representations = {}
    self.statistics = []


  def doTimestep(self, sensations, learn):
    """
    Run the network for one timestep.

    The format of sensations is:

    sensations = {
      0: (set([1, 5, 10]), set([6, 12, 52]),  # location, feature for CC0
      1: (set([6, 2, 15]), set([64, 1, 5]),  # location, feature for CC1
    }

    Parameters:
    ----------------------------
    @param   sensations (dict)
             The feature/location pair for each column.

    @param   learn (bool)
             Whether to allow learning for this timestep
    """

    prevL2Representations = [L2.getActiveCells() for L2 in self.L2Columns]

    for col in xrange(self.numColumns):
      location, feature = sensations[col]
      location = sorted(location)
      feature = sorted(feature)
      L2 = self.L2Columns[col]
      L4 = self.L4Columns[col]

      # Compute L4's active columns
      if self.L4FeedforwardSPs is not None:
        featureDense = np.zeros(self.inputSize, dtype="uint32")
        featureDense[feature] = 1

        spOutput = np.zeros(self.inputSize, dtype="uint32")
        self.L4FeedforwardSPs[col].compute(featureDense, learn, spOutput)

        activeColumns = spOutput.nonzero()[0]
      else:
        activeColumns = np.asarray(feature, dtype="uint32")

      # Compute L4's distal basal input
      if self.L4LateralSPs is not None:
        locationDense = np.zeros(self.externalInputSize, dtype="uint32")
        locationDense[location] = 1

        spOutput = np.zeros(self.externalInputSize, dtype="uint32")
        self.L4LateralSPs[col].compute(locationDense, learn, spOutput)

        basalInput = spOutput.nonzero()[0]
      else:
        basalInput = np.asarray(location, dtype="uint32")

      # Compute L4's active cells
      apicalInput = L2.getActiveCells()
      L4.depolarizeCells(basalInput, apicalInput, learn=learn)
      L4.activateCells(activeColumns,
                       reinforceCandidatesExternalBasal=basalInput,
                       reinforceCandidatesExternalApical=apicalInput,
                       growthCandidatesExternalBasal=basalInput,
                       growthCandidatesExternalApical=apicalInput,
                       learn=learn)

      # Compute L2's active cells
      lateralInputs = [prevActiveCells
                       for i, prevActiveCells in enumerate(
                           prevL2Representations)
                       if i != col]
      L2.compute(feedforwardInput=L4.getActiveCells(),
                 feedforwardGrowthCandidates=L4.getPredictedActiveCells(),
                 lateralInputs=lateralInputs,
                 learn=learn)


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
    for objectName, sensationList in objects.iteritems():

      # ignore empty sensation lists
      if len(sensationList) == 0:
        continue

      for sensations in sensationList:
        # learn each pattern multiple times
        for _ in xrange(self.numLearningPoints):
          self.doTimestep(sensations, learn=True)

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
    statistics = collections.defaultdict(list)

    if objectName is not None:
      if objectName not in self.objectL2Representations:
        raise ValueError("The provided objectName was not given during"
                         " learning")

    for sensations in sensationList:
      self.doTimestep(sensations, learn=False)
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
      self.L4Columns[col].reset()
      self.L2Columns[col].reset()


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


  def getL4Representations(self):
    """
    Returns the active representation in L4.
    """
    return [set(L4.getActiveCells()) for L4 in self.L4Columns]


  def getL4PredictiveCells(self):
    """
    Returns the predictive cells in L4.
    """
    return [set(L4.getPredictiveCells()) for L4 in self.L4Columns]


  def getL2Representations(self):
    """
    Returns the active representation in L2.
    """
    return [set(L2.getActiveCells()) for L2 in self.L2Columns]


  def getCurrentClassification(self, minOverlap=None):
    """
    A dict with a score for each object. Score goes from 0 to 1. A 1 means
    every col (that has received input since the last reset) currently has
    overlap >= minOverlap with the representation for that object.

    :param minOverlap: min overlap to consider the object as recognized.
                       Defaults to half of the SDR size
    :return: dict of object names and their score
    """
    results = {}
    l2sdr = self.getL2Representations()
    sdrSize = self.L2Params["sdrSize"]
    if minOverlap is None:
      minOverlap = sdrSize / 2

    for objectName, objectSdr in self.objectL2Representations.iteritems():
      count = 0
      score = 0.0
      for i in xrange(self.numColumns):
        # Ignore inactive column
        if len(l2sdr[i]) == 0:
          continue

        count += 1
        overlap = len(l2sdr[i] & objectSdr[i])
        if overlap >= minOverlap:
          score += 1

      if count == 0:
        results[objectName] = 0
      else:
        results[objectName] = score / count

    return results


  def getDefaultL4Params(self, inputSize, externalInputSize, numInputBits):
    """
    Returns a good default set of parameters to use in the L4 region.
    """
    sampleSize = int(1.5 * numInputBits)

    if numInputBits == 20:
      activationThreshold = 13
      minThreshold = 13
    elif numInputBits == 10:
      activationThreshold = 8
      minThreshold = 8
    else:
      activationThreshold = int(numInputBits * .6)
      minThreshold = activationThreshold

    return {
      "columnDimensions": (inputSize,),
      "basalInputDimensions": (externalInputSize,),
      "apicalInputDimensions": (4096,), # Keep synced with L2 "cellCount"
      "cellsPerColumn": 16, # Keep synced with L2 "inputWidth"
      "formInternalBasalConnections": False,
      "learnOnOneCell": False,
      "initialPermanence": 0.51,
      "connectedPermanence": 0.6,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.02,
      "minThreshold": minThreshold,
      "predictedSegmentDecrement": 0.0,
      "activationThreshold": activationThreshold,
      "maxNewSynapseCount": sampleSize,
      "seed": self.seed,
      "checkInputs": False,
    }


  def getDefaultL2Params(self, numCorticalColumns, inputSize, numInputBits):
    """
    Returns a good default set of parameters to use in the L2 region.
    """
    if numInputBits == 20:
      sampleSizeProximal = 10
      minThresholdProximal = 6
    elif numInputBits == 10:
      sampleSizeProximal = 6
      minThresholdProximal = 3
    else:
      sampleSizeProximal = int(numInputBits * .6)
      minThresholdProximal = int(sampleSizeProximal * .6)

    return {
      "inputWidth": inputSize * 16, # Keep synced with L4 "cellsPerColumn"
      "cellCount": 4096, # Keep synced with L4 "apicalInputDimensions"
      "lateralInputWidths": [4096]*(numCorticalColumns-1),
      "sdrSize": 40,
      "synPermProximalInc": 0.1,
      "synPermProximalDec": 0.001,
      "initialProximalPermanence": 0.6,
      "minThresholdProximal": minThresholdProximal,
      "sampleSizeProximal": sampleSizeProximal,
      "connectedPermanenceProximal": 0.5,
      "synPermDistalInc": 0.1,
      "synPermDistalDec": 0.001,
      "initialDistalPermanence": 0.41,
      "activationThresholdDistal": 13,
      "sampleSizeDistal": 20,
      "connectedPermanenceDistal": 0.5,
      "distalSegmentInhibitionFactor": 1.5,
    }


  def getDefaultLateralSPParams(self, inputSize):
    return {
      "globalInhibition": True,
      "columnDimensions": (inputSize,),
      "inputDimensions": (inputSize,),
      "potentialRadius": inputSize,
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
      "globalInhibition": True,
      "columnDimensions": (inputSize,),
      "inputDimensions": (inputSize,),
      "potentialRadius": inputSize,
      "numActiveColumnsPerInhArea": 40,
      "seed": self.seed,
      "potentialPct": 0.8,
      "synPermConnected": 0.1,
      "synPermActiveInc": 0.0001,
      "synPermInactiveDec": 0.0005,
      "boostStrength": 0.0,
    }


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
      statistics["L4 Apical Segments C" + str(i)].append(
        len(self.L4Columns[i].getActiveApicalSegments())
      )

      # add true overlap if objectName was provided
      if objectName is not None:
        objectRepresentation = self.objectL2Representations[objectName]
        statistics["Overlap L2 with object C" + str(i)].append(
          len(objectRepresentation[i] & L2Representation[i])
        )
