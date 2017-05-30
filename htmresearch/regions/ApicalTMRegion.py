# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
Region for Temporal Memory with various apical implementations.
"""

import numpy as np

from nupic.bindings.regions.PyRegion import PyRegion



class ApicalTMRegion(PyRegion):
  """
  Implements temporal memory for the HTM network API. The temporal memory uses
  basal and apical dendrites.
  """

  @classmethod
  def getSpec(cls):
    """
    Return the Spec for ApicalTMRegion.
    """

    spec = {
      "description": ApicalTMRegion.__doc__,
      "singleNodeOnly": True,
      "inputs": {
        "activeColumns": {
          "description": ("An array of 0's and 1's representing the active "
                          "minicolumns, i.e. the input to the TemporalMemory"),
          "dataType": "Real32",
          "count": 0,
          "required": True,
          "regionLevel": True,
          "isDefaultInput": True,
          "requireSplitterMap": False
        },
        "basalInput": {
          "description": "An array of 0's and 1's representing basal input",
          "dataType": "Real32",
          "count": 0,
          "required": False,
          "regionLevel": True,
          "isDefaultInput": False,
          "requireSplitterMap": False
        },
        "basalGrowthCandidates": {
          "description": ("An array of 0's and 1's representing basal input " +
                          "that can be learned on new synapses on basal " +
                          "segments. If this input is a length-0 array, the " +
                          "whole basalInput is used."),
          "dataType": "Real32",
          "count": 0,
          "required": False,
          "regionLevel": True,
          "isDefaultInput": False,
          "requireSplitterMap": False
        },
        "apicalInput": {
          "description": "An array of 0's and 1's representing top down input."
          " The input will be provided to apical dendrites.",
          "dataType": "Real32",
          "count": 0,
          "required": False,
          "regionLevel": True,
          "isDefaultInput": False,
          "requireSplitterMap": False
        },
        "apicalGrowthCandidates": {
          "description": ("An array of 0's and 1's representing apical input " +
                          "that can be learned on new synapses on apical " +
                          "segments. If this input is a length-0 array, the " +
                          "whole apicalInput is used."),
          "dataType": "Real32",
          "count": 0,
          "required": False,
          "regionLevel": True,
          "isDefaultInput": False,
          "requireSplitterMap": False},
      },
      "outputs": {
        "predictedCells": {
          "description": ("A binary output containing a 1 for every "
                          "cell that was predicted for this timestep."),
          "dataType": "Real32",
          "count": 0,
          "regionLevel": True,
          "isDefaultOutput": False
        },

        "predictedActiveCells": {
          "description": ("A binary output containing a 1 for every "
                          "cell that transitioned from predicted to active."),
          "dataType": "Real32",
          "count": 0,
          "regionLevel": True,
          "isDefaultOutput": False
        },

        "activeCells": {
          "description": ("A binary output containing a 1 for every "
                          "cell that is currently active."),
          "dataType": "Real32",
          "count": 0,
          "regionLevel": True,
          "isDefaultOutput": True
        },

        "winnerCells": {
          "description": ("A binary output containing a 1 for every "
                          "'winner' cell in the TM."),
          "dataType": "Real32",
          "count": 0,
          "regionLevel": True,
          "isDefaultOutput": False
        },
      },

      "parameters": {

        # Input sizes (the network API doesn't provide these during initialize)
        "columnCount": {
          "description": ("The size of the 'activeColumns' input " +
                          "(i.e. the number of columns)"),
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1,
          "constraints": ""
        },

        "basalInputWidth": {
          "description": "The size of the 'basalInput' input",
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1,
          "constraints": ""
        },

        "apicalInputWidth": {
          "description": "The size of the 'apicalInput' input",
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1,
          "constraints": ""
        },

        "learn": {
          "description": "True if the TM should learn.",
          "accessMode": "ReadWrite",
          "dataType": "Bool",
          "count": 1,
          "defaultValue": "true"
        },
        "cellsPerColumn": {
          "description": "Number of cells per column",
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1,
          "constraints": ""
        },
        "activationThreshold": {
          "description": ("If the number of active connected synapses on a "
                          "segment is at least this threshold, the segment "
                          "is said to be active."),
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1,
          "constraints": ""
        },
        "reducedThresholdBasal": {
          "description": ("Activation threshold of basal segments for cells "
                          "with active apical segments (with apicalTiebreak). "),
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1,
          "constraints": ""
        },
        "initialPermanence": {
          "description": "Initial permanence of a new synapse.",
          "accessMode": "Read",
          "dataType": "Real32",
          "count": 1,
          "constraints": ""
        },
        "connectedPermanence": {
          "description": ("If the permanence value for a synapse is greater "
                          "than this value, it is said to be connected."),
          "accessMode": "Read",
          "dataType": "Real32",
          "count": 1,
          "constraints": ""
        },
        "minThreshold": {
          "description": ("If the number of synapses active on a segment is at "
                          "least this threshold, it is selected as the best "
                          "matching cell in a bursting column."),
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1,
          "constraints": ""
        },
        "sampleSize": {
          "description": ("The desired number of active synapses for an " +
                          "active cell"),
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1
        },
        "maxSynapsesPerSegment": {
          "description": "The maximum number of synapses per segment",
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1
        },
        "permanenceIncrement": {
          "description": ("Amount by which permanences of synapses are "
                          "incremented during learning."),
          "accessMode": "Read",
          "dataType": "Real32",
          "count": 1
        },
        "permanenceDecrement": {
          "description": ("Amount by which permanences of synapses are "
                          "decremented during learning."),
          "accessMode": "Read",
          "dataType": "Real32",
          "count": 1
        },
        "basalPredictedSegmentDecrement": {
          "description": ("Amount by which active permanences of synapses of "
                          "previously predicted but inactive segments are "
                          "decremented."),
          "accessMode": "Read",
          "dataType": "Real32",
          "count": 1
        },
        "apicalPredictedSegmentDecrement": {
          "description": ("Amount by which active permanences of synapses of "
                          "previously predicted but inactive segments are "
                          "decremented."),
          "accessMode": "Read",
          "dataType": "Real32",
          "count": 1
        },
        "seed": {
          "description": "Seed for the random number generator.",
          "accessMode": "Read",
          "dataType": "UInt32",
          "count": 1
        },
        "implementation": {
          "description": "Apical implementation",
          "accessMode": "Read",
          "dataType": "Byte",
          "count": 0,
          "constraints": ("enum: ApicalTiebreak, ApicalDependent, ApicalModulation"),
          "defaultValue": "ApicalTiebreak"
        },
      },
    }

    return spec


  def __init__(self,

               # Input sizes
               columnCount,
               basalInputWidth,
               apicalInputWidth,

               # TM params
               cellsPerColumn=32,
               activationThreshold=13,
               initialPermanence=0.21,
               connectedPermanence=0.50,
               minThreshold=10,
               reducedThresholdBasal=11, # Only used for apicalTiebreak implementation
               sampleSize=20,
               permanenceIncrement=0.10,
               permanenceDecrement=0.10,
               basalPredictedSegmentDecrement=0.0,
               apicalPredictedSegmentDecrement=0.0,
               maxSynapsesPerSegment=255,
               seed=42,

               # Region params
               implementation="ApicalTiebreak",
               learn=True,
               **kwargs):

    # Input sizes (the network API doesn't provide these during initialize)
    self.columnCount = columnCount
    self.basalInputWidth = basalInputWidth
    self.apicalInputWidth = apicalInputWidth

    # TM params
    self.cellsPerColumn = cellsPerColumn
    self.activationThreshold = activationThreshold
    self.reducedThresholdBasal = reducedThresholdBasal
    self.initialPermanence = initialPermanence
    self.connectedPermanence = connectedPermanence
    self.minThreshold = minThreshold
    self.sampleSize = sampleSize
    self.permanenceIncrement = permanenceIncrement
    self.permanenceDecrement = permanenceDecrement
    self.basalPredictedSegmentDecrement = basalPredictedSegmentDecrement
    self.apicalPredictedSegmentDecrement = apicalPredictedSegmentDecrement
    self.maxSynapsesPerSegment = maxSynapsesPerSegment
    self.seed = seed

    # Region params
    self.implementation = implementation
    self.learn = learn

    PyRegion.__init__(self, **kwargs)

    # TM instance
    self._tm = None


  def initialize(self):
    """
    Initialize the self._tm if not already initialized.
    """

    if self._tm is None:
      params = {
        "columnCount": self.columnCount,
        "basalInputSize": self.basalInputWidth,
        "apicalInputSize": self.apicalInputWidth,
        "cellsPerColumn": self.cellsPerColumn,
        "activationThreshold": self.activationThreshold,
        "reducedThresholdBasal": self.reducedThresholdBasal,
        "initialPermanence": self.initialPermanence,
        "connectedPermanence": self.connectedPermanence,
        "minThreshold": self.minThreshold,
        "sampleSize": self.sampleSize,
        "permanenceIncrement": self.permanenceIncrement,
        "permanenceDecrement": self.permanenceDecrement,
        "basalPredictedSegmentDecrement": self.basalPredictedSegmentDecrement,
        "apicalPredictedSegmentDecrement": self.apicalPredictedSegmentDecrement,
        "maxSynapsesPerSegment": self.maxSynapsesPerSegment,
        "seed": self.seed,
      }

      if self.implementation == "ApicalTiebreak":
        from htmresearch.algorithms.apical_tiebreak_temporal_memory import (
          ApicalTiebreakTemporalMemory)
        self._tm = ApicalTiebreakTemporalMemory(**params)
      elif self.implementation == "ApicalModulation":
        from htmresearch.algorithms.apical_modulation_temporal_memory import (
          ApicalModulationTemporalMemory)
        self._tm = ApicalModulationTemporalMemory(**params)
      elif self.implementation == "ApicalDependent":
        from htmresearch.algorithms.apical_dependent_temporal_memory import (
          ApicalDependentTemporalMemory)
        self._tm = ApicalDependentTemporalMemory(**params)
      else:
        raise ValueError("Unrecognized implementation %s" % self.implementation)


  def compute(self, inputs, outputs):
    """
    Run one iteration of TM's compute.
    """

    activeColumns = inputs["activeColumns"].nonzero()[0]

    if "basalInput" in inputs:
      basalInput = inputs["basalInput"].nonzero()[0]
    else:
      basalInput = np.empty(0, dtype="uint32")

    if "apicalInput" in inputs:
      apicalInput = inputs["apicalInput"].nonzero()[0]
    else:
      apicalInput = np.empty(0, dtype="uint32")

    if "basalGrowthCandidates" in inputs:
      basalGrowthCandidates = inputs["basalGrowthCandidates"].nonzero()[0]
    else:
      basalGrowthCandidates = basalInput

    if "apicalGrowthCandidates" in inputs:
      apicalGrowthCandidates = inputs["apicalGrowthCandidates"].nonzero()[0]
    else:
      apicalGrowthCandidates = apicalInput

    self._tm.compute(activeColumns, basalInput, apicalInput,
                     basalGrowthCandidates, apicalGrowthCandidates, self.learn)

    # Extract the active / predicted cells and put them into binary arrays.
    outputs["activeCells"][:] = 0
    outputs["activeCells"][self._tm.getActiveCells()] = 1
    outputs["predictedCells"][:] = 0
    outputs["predictedCells"][
      self._tm.getPredictedCells()] = 1
    outputs["predictedActiveCells"][:] = (outputs["activeCells"] *
                                          outputs["predictedCells"])
    outputs["winnerCells"][:] = 0
    outputs["winnerCells"][self._tm.getWinnerCells()] = 1


  def getParameter(self, parameterName, index=-1):
    """
      Get the value of a NodeSpec parameter. Most parameters are handled
      automatically by PyRegion's parameter get mechanism. The ones that need
      special treatment are explicitly handled here.
    """
    return PyRegion.getParameter(self, parameterName, index)


  def setParameter(self, parameterName, index, parameterValue):
    """
    Set the value of a Spec parameter. Most parameters are handled
    automatically by PyRegion's parameter set mechanism. The ones that need
    special treatment are explicitly handled here.
    """
    if hasattr(self, parameterName):
      setattr(self, parameterName, parameterValue)
    else:
      raise Exception("Unknown parameter: " + parameterName)


  def getOutputElementCount(self, name):
    """
    Return the number of elements for the given output.
    """
    if name in ["activeCells", "predictedCells", "predictedActiveCells",
                "winnerCells"]:
      return self.cellsPerColumn * self.columnCount
    else:
      raise Exception("Invalid output name specified: %s" % name)
