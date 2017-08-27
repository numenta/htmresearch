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
Run the apical tiebreak tests on the ExtendedTemporalMemory.
"""

import unittest

from htmresearch_core.experimental import ApicalTiebreakPairMemory
from htmresearch.support.shared_tests.apical_tiebreak_test_base import (
  ApicalTiebreakTestBase)


class ExtendedTM_ApicalTiebreakTests(ApicalTiebreakTestBase,
                                     unittest.TestCase):
  """
  Run the apical tiebreak tests on the C++ ExtendedTemporalMemory.
  """

  def constructTM(self, columnCount, basalInputSize, apicalInputSize,
                  cellsPerColumn, initialPermanence, connectedPermanence,
                  minThreshold, sampleSize, permanenceIncrement,
                  permanenceDecrement, predictedSegmentDecrement,
                  activationThreshold, seed):

    params = {
      "columnCount": columnCount,
      "basalInputSize": basalInputSize,
      "apicalInputSize": apicalInputSize,
      "cellsPerColumn": cellsPerColumn,
      "initialPermanence": initialPermanence,
      "connectedPermanence": connectedPermanence,
      "minThreshold": minThreshold,
      "sampleSize": sampleSize,
      "permanenceIncrement": permanenceIncrement,
      "permanenceDecrement": permanenceDecrement,
      "basalPredictedSegmentDecrement": predictedSegmentDecrement,
      "activationThreshold": activationThreshold,
      "seed": seed,
      "learnOnOneCell": False,
    }

    self.tm = ApicalTiebreakPairMemory(**params)


  def compute(self, activeColumns, basalInput, apicalInput, learn):

    activeColumns = sorted(activeColumns)
    basalInput = sorted(basalInput)
    apicalInput = sorted(apicalInput)

    self.tm.compute(activeColumns, basalInput, apicalInput)


  def getActiveCells(self):
    return self.tm.getActiveCells()


  def getPredictedCells(self):
    return self.tm.getPredictedCells()
