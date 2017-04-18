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
Run the 'apical tiebreak sequences' tests on the ApicalTiebreakTemporalMemory
"""

import unittest

import numpy as np

from htmresearch.algorithms.apical_tiebreak_temporal_memory import (
  ApicalTiebreakTemporalMemory)
from htmresearch.support.shared_tests.apical_tiebreak_sequences_test_base import (
  ApicalTiebreakSequencesTestBase)


class ApicalTiebreakTM_ApicalTiebreakSequencesTests(ApicalTiebreakSequencesTestBase,
                                                    unittest.TestCase):
  """
  Runs the "apical tiebreak sequences" tests on the ApicalTiebreakTemporalMemory
  """

  def constructTM(self, columnCount, apicalInputSize, cellsPerColumn,
                  initialPermanence, connectedPermanence, minThreshold,
                  sampleSize, permanenceIncrement, permanenceDecrement,
                  predictedSegmentDecrement, activationThreshold, seed):

    params = {
      "columnCount": columnCount,
      "cellsPerColumn": cellsPerColumn,
      "initialPermanence": initialPermanence,
      "connectedPermanence": connectedPermanence,
      "minThreshold": minThreshold,
      "sampleSize": sampleSize,
      "permanenceIncrement": permanenceIncrement,
      "permanenceDecrement": permanenceDecrement,
      "basalPredictedSegmentDecrement": predictedSegmentDecrement,
      "apicalPredictedSegmentDecrement": 0.0,
      "activationThreshold": activationThreshold,
      "seed": seed,
      "basalInputSize": columnCount*cellsPerColumn,
      "apicalInputSize": apicalInputSize,
    }

    self.tm = ApicalTiebreakTemporalMemory(**params)


  def compute(self, activeColumns, apicalInput, learn):
    activeColumns = np.array(sorted(activeColumns), dtype="uint32")
    apicalInput = sorted(apicalInput)

    self.tm.compute(activeColumns,
                    basalInput=self.tm.getActiveCells(),
                    basalGrowthCandidates=self.tm.getWinnerCells(),
                    apicalInput=apicalInput,
                    apicalGrowthCandidates=apicalInput,
                    learn=learn)


  def reset(self):
    self.tm.reset()


  def getActiveCells(self):
    return self.tm.getActiveCells()


  def getPredictedCells(self):
    return self.tm.getPredictedCells()
