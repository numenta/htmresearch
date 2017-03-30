# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
Run the apical tiebreak tests on the Python ExtendedTemporalMemory.
"""

import unittest

from htmresearch.algorithms.extended_temporal_memory import ExtendedTemporalMemory
from htmresearch.support.shared_tests.apical_tiebreak_test_base import (
  ApicalTiebreakTestBase)


class ExtendedTMPY_ApicalTiebreakTests(ApicalTiebreakTestBase,
                                       unittest.TestCase):
  """
  Run the apical tiebreak tests on the Python ExtendedTemporalMemory.
  """

  def constructTM(self, columnCount, basalInputSize, apicalInputSize,
                  cellsPerColumn, initialPermanence, connectedPermanence,
                  minThreshold, sampleSize, permanenceIncrement,
                  permanenceDecrement, predictedSegmentDecrement,
                  activationThreshold, seed):

    params = {
      "columnDimensions": (columnCount,),
      "basalInputDimensions": (basalInputSize,),
      "apicalInputDimensions": (apicalInputSize,),
      "cellsPerColumn": cellsPerColumn,
      "initialPermanence": initialPermanence,
      "connectedPermanence": connectedPermanence,
      "minThreshold": minThreshold,
      "maxNewSynapseCount": sampleSize,
      "permanenceIncrement": permanenceIncrement,
      "permanenceDecrement": permanenceDecrement,
      "predictedSegmentDecrement": predictedSegmentDecrement,
      "activationThreshold": activationThreshold,
      "seed": seed,
      "learnOnOneCell": False,
      "formInternalBasalConnections": False,
    }

    self.tm = ExtendedTemporalMemory(**params)


  def compute(self, activeColumns, basalInput, apicalInput, learn):

    activeColumns = sorted(activeColumns)
    basalInput = sorted(basalInput)
    apicalInput = sorted(apicalInput)

    # Use depolarizeCells + activateCells rather than tm.compute so that
    # getPredictiveCells returns predictions for the current timestep.
    self.tm.depolarizeCells(activeCellsExternalBasal=basalInput,
                            activeCellsExternalApical=apicalInput,
                            learn=learn)
    self.tm.activateCells(activeColumns,
                          reinforceCandidatesExternalBasal=basalInput,
                          growthCandidatesExternalBasal=basalInput,
                          reinforceCandidatesExternalApical=apicalInput,
                          growthCandidatesExternalApical=apicalInput,
                          learn=learn)


  def getActiveCells(self):
    return self.tm.getActiveCells()


  def getPreviouslyPredictedCells(self):
    return self.tm.getPredictiveCells()
