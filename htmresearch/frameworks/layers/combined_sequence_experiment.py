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
This class allows to easily run the L4-TM combined sequences experiment. Its use
is identical to the use of its superclass.
"""
# Disable variable/field name restrictions
# pylint: disable=C0103

import pprint
import random

from nupic.bindings.math import SparseMatrix
from htmresearch.support.register_regions import registerAllResearchRegions
from htmresearch.frameworks.layers.laminar_network import createNetwork
from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment


class L4TMExperiment(L4L2Experiment):
  """
  L4-TM combined sequences experiment.
  """

  def __init__(self,
               name,
               numCorticalColumns=1,
               inputSize=1024,
               numInputBits=20,
               externalInputSize=1024,
               numExternalInputBits=20,
               L2Overrides=None,
               networkType = "L4L2TMColumn",
               L4Overrides=None,
               seed=42,
               logCalls=False,
               objectNamesAreIndices=False,
               TMOverrides=None,
               ):
    """
    Creates the network.

    Parameters:
    ----------------------------
    @param   TMOverrides (dict)
             Parameters to override in the TM region
    """

    # Handle logging - this has to be done first
    self.logCalls = logCalls

    registerAllResearchRegions()
    self.name = name

    self.numLearningPoints = 1
    self.numColumns = numCorticalColumns
    self.inputSize = inputSize
    self.externalInputSize = externalInputSize
    self.numInputBits = numInputBits
    self.objectNamesAreIndices = objectNamesAreIndices

    # seed
    self.seed = seed
    random.seed(seed)

    # update parameters with overrides
    self.config = {
      "networkType": networkType,
      "numCorticalColumns": numCorticalColumns,
      "externalInputSize": externalInputSize,
      "sensorInputSize": inputSize,
      "enableFeedback": False,
      "L4Params": self.getDefaultL4Params(inputSize, numExternalInputBits),
      "L2Params": self.getDefaultL2Params(inputSize, numInputBits),
      "TMParams": self.getDefaultTMParams(self.inputSize, self.numInputBits),
    }


    if L2Overrides is not None:
      self.config["L2Params"].update(L2Overrides)

    if L4Overrides is not None:
      self.config["L4Params"].update(L4Overrides)

    if TMOverrides is not None:
      self.config["TMParams"].update(TMOverrides)

    # Recreate network including TM parameters
    self.network = createNetwork(self.config)
    self.sensorInputs = []
    self.externalInputs = []
    self.L4Regions = []
    self.L2Regions = []
    self.TMRegions = []

    for i in xrange(self.numColumns):
      self.sensorInputs.append(
        self.network.regions["sensorInput_" + str(i)].getSelf()
      )
      self.externalInputs.append(
        self.network.regions["externalInput_" + str(i)].getSelf()
      )
      self.L4Regions.append(
        self.network.regions["L4Column_" + str(i)]
      )
      self.L2Regions.append(
        self.network.regions["L2Column_" + str(i)]
      )
      self.TMRegions.append(
        self.network.regions["TMColumn_" + str(i)]
      )

    self.L4Columns = [region.getSelf() for region in self.L4Regions]
    self.L2Columns = [region.getSelf() for region in self.L2Regions]
    self.TMColumns = [region.getSelf() for region in self.TMRegions]

    # will be populated during training
    self.objectL2Representations = {}
    self.objectL2RepresentationsMatrices = [
      SparseMatrix(0, self.config["L2Params"]["cellCount"])
      for _ in xrange(self.numColumns)]
    self.objectNameToIndex = {}
    self.statistics = []


  def getTMRepresentations(self):
    """
    Returns the active representation in TM.
    """
    return [set(column.getOutputData("activeCells").nonzero()[0])
            for column in self.TMRegions]


  def getTMNextPredictedCells(self):
    """
    Returns the cells in TM that were predicted at the end of the most recent
    call to 'compute'.
    """
    return [set(column.getOutputData("nextPredictedCells").nonzero()[0])
            for column in self.TMRegions]


  def getTMPredictedActiveCells(self):
    """
    Returns the cells in TM that were predicted at the beginning of the most
    recent call to 'compute' and are currently active.
    """
    return [set(column.getOutputData("predictedActiveCells").nonzero()[0])
            for column in self.TMRegions]


  def getDefaultTMParams(self, inputSize, numInputBits):
    """
    Returns a good default set of parameters to use in the TM region.
    """
    sampleSize = int(1.5 * numInputBits)

    if numInputBits == 20:
      activationThreshold = 18
      minThreshold = 18
    elif numInputBits == 10:
      activationThreshold = 8
      minThreshold = 8
    else:
      activationThreshold = int(numInputBits * .6)
      minThreshold = activationThreshold

    return {
      "columnCount": inputSize,
      "cellsPerColumn": 16,
      "learn": True,
      "learnOnOneCell": False,
      "initialPermanence": 0.41,
      "connectedPermanence": 0.6,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.03,
      "minThreshold": minThreshold,
      "basalPredictedSegmentDecrement": 0.003,
      "apicalPredictedSegmentDecrement": 0.0,
      "reducedBasalThreshold": int(activationThreshold*0.6),
      "activationThreshold": activationThreshold,
      "sampleSize": sampleSize,
      "implementation": "ApicalTiebreak",
      "seed": self.seed
    }


  def averageSequenceAccuracy(self, minOverlap, maxOverlap,
                              firstStat=0, lastStat=None):
    """
    For each object, decide whether the TM uniquely classified it by checking
    that the number of predictedActive cells are in an acceptable range.
    """
    numCorrect = 0.0
    numStats = 0.0
    prefix = "TM PredictedActive"

    # For each object
    for stats in self.statistics[firstStat:lastStat]:

      # Keep running total of how often the number of predictedActive cells are
      # in the range.  We always skip the first (unpredictable) count.
      for key in stats.iterkeys():
        if prefix in key:
          for numCells in stats[key][1:]:
            numStats += 1.0
            if numCells in range(minOverlap, maxOverlap + 1):
              numCorrect += 1.0
          # print stats["object"], key, stats[key]
          # print

    return numCorrect / numStats


  def _unsetLearningMode(self):
    """
    Unsets the learning mode, to start inference.
    """
    for region in self.TMRegions:
      region.setParameter("learn", False)
    super(L4TMExperiment, self)._unsetLearningMode()


  def _setLearningMode(self):
    """
    Sets the learning mode.
    """
    for region in self.TMRegions:
      region.setParameter("learn", True)
    super(L4TMExperiment, self)._setLearningMode()


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
    L4PredictedCells = self.getL4PredictedCells()
    L4PredictedActiveCells = self.getL4PredictedActiveCells()
    L2Representation = self.getL2Representations()
    TMPredictedActive = self.getTMPredictedActiveCells()
    TMNextPredicted = self.getTMNextPredictedCells()
    TMRepresentation = self.getTMRepresentations()

    for i in xrange(self.numColumns):
      statistics["L4 Representation C" + str(i)].append(
        len(L4Representations[i])
      )
      statistics["L4 Predicted C" + str(i)].append(
        len(L4PredictedCells[i])
      )
      statistics["L4 PredictedActive C" + str(i)].append(
        len(L4PredictedActiveCells[i])
      )
      statistics["L2 Representation C" + str(i)].append(
        len(L2Representation[i])
      )
      statistics["L4 Apical Segments C" + str(i)].append(
        len(self.L4Columns[i]._tm.getActiveApicalSegments())
      )
      statistics["L4 Basal Segments C" + str(i)].append(
        len(self.L4Columns[i]._tm.getActiveBasalSegments())
      )
      statistics["TM Basal Segments C" + str(i)].append(
        len(self.TMColumns[i]._tm.getActiveBasalSegments())
      )
      statistics["TM PredictedActive C" + str(i)].append(
        len(TMPredictedActive[i])
      )
      statistics["TM NextPredicted C" + str(i)].append(
        len(TMNextPredicted[i])
      )
      statistics["TM Representation C" + str(i)].append(
        len(TMRepresentation[i])
      )

      # add true overlap if objectName was provided
      if objectName in self.objectL2Representations:
        objectRepresentation = self.objectL2Representations[objectName]
        statistics["Overlap L2 with object C" + str(i)].append(
          len(objectRepresentation[i] & L2Representation[i])
        )
