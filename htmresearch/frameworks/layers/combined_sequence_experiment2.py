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


class CombinedSequenceExperiment(L4L2Experiment):
  """
  L2-TM combined sequences experiment.
  """

  def __init__(self,
               name,
               numCorticalColumns=1,
               inputSize=1024,
               numInputBits=20,
               externalInputSize=1024,
               numExternalInputBits=20,
               L2Overrides=None,
               L4Overrides=None,
               seed=42,
               logCalls=False,
               objectNamesAreIndices=False,
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
    self.numExternalInputBits = numExternalInputBits

    # seed
    self.seed = seed
    random.seed(seed)

    # Create default parameters and then update with overrides
    self.config = {
      "networkType": "CombinedSequenceColumn",
      "numCorticalColumns": numCorticalColumns,
      "externalInputSize": externalInputSize,
      "sensorInputSize": inputSize,
      "enableFeedback": False,
      "L2Params": self.getDefaultL2Params(inputSize, numInputBits),
    }
    self.config["L4Params"] = self._getDefaultCombinedL4Params(
      self.numInputBits, self.inputSize,
      self.numExternalInputBits, self.externalInputSize,
      self.config["L2Params"]["cellCount"])

    if L2Overrides is not None:
      self.config["L2Params"].update(L2Overrides)

    if L4Overrides is not None:
      self.config["L4Params"].update(L4Overrides)

    pprint.pprint(self.config)

    # Recreate network including TM parameters
    self.network = createNetwork(self.config)
    self.sensorInputs = []
    self.externalInputs = []
    self.L2Regions = []
    self.L4Regions = []

    for i in xrange(self.numColumns):
      self.sensorInputs.append(
        self.network.regions["sensorInput_" + str(i)].getSelf()
      )
      self.externalInputs.append(
        self.network.regions["externalInput_" + str(i)].getSelf()
      )
      self.L2Regions.append(
        self.network.regions["L2Column_" + str(i)]
      )
      self.L4Regions.append(
        self.network.regions["L4Column_" + str(i)]
      )

    self.L2Columns = [region.getSelf() for region in self.L2Regions]
    self.L4Columns = [region.getSelf() for region in self.L4Regions]

    # will be populated during training
    self.objectL2Representations = {}
    self.objectL2RepresentationsMatrices = [
      SparseMatrix(0, self.config["L2Params"]["cellCount"])
      for _ in xrange(self.numColumns)]
    self.objectNameToIndex = {}
    self.statistics = []


  def _getDefaultCombinedL4Params(self, numInputBits, inputSize,
                                  numExternalInputBits, externalInputSize,
                                  L2CellCount):
    """
    Returns a good default set of parameters to use in a combined L4 region.
    """
    sampleSize = numExternalInputBits + numInputBits
    activationThreshold = int(max(numExternalInputBits, numInputBits) * .6)
    minThreshold = activationThreshold

    return {
      "columnCount": inputSize,
      "cellsPerColumn": 16,
      "learn": True,
      "learnOnOneCell": False,
      "initialPermanence": 0.41,
      "connectedPermanence": 0.6,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.02,
      "minThreshold": minThreshold,
      "basalPredictedSegmentDecrement": 0.001,
      "apicalPredictedSegmentDecrement": 0.0,
      "reducedBasalThreshold": int(activationThreshold*0.6),
      "activationThreshold": activationThreshold,
      "sampleSize": sampleSize,
      "implementation": "ApicalTiebreak",
      "seed": self.seed,
      "basalInputWidth": inputSize*16 + externalInputSize,
      "apicalInputWidth": L2CellCount,
    }


  def averageSequenceAccuracy(self, minOverlap, maxOverlap):
    """
    For each object, decide whether the TM uniquely classified it by checking
    that the number of predictedActive cells are in an acceptable range.
    """
    numCorrect = 0.0
    numStats = 0.0
    prefix = "L4 PredictedActive"

    # For each object
    for stats in self.statistics:

      # Keep running total of how often the number of predictedActive cells are
      # in the range.
      for key in stats.iterkeys():
        if prefix in key:
          for numCells in stats[key]:
            numStats += 1.0
            if numCells in range(minOverlap, maxOverlap + 1):
              numCorrect += 1.0

    return numCorrect / numStats


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

      # add true overlap if objectName was provided
      if objectName in self.objectL2Representations:
        objectRepresentation = self.objectL2Representations[objectName]
        statistics["Overlap L2 with object C" + str(i)].append(
          len(objectRepresentation[i] & L2Representation[i])
        )
