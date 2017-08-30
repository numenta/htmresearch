# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016-2017, Numenta, Inc.  Unless you have an agreement
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
Run the sequence memory tests on the C++ ExtendedTemporalMemory.
"""

import unittest

from htmresearch_core.experimental import ApicalTiebreakSequenceMemory

from htmresearch.support.shared_tests.sequence_memory_test_base import (
  SequenceMemoryTestBase)


class ExtendedTM_SequenceMemoryTests(SequenceMemoryTestBase,
                                     unittest.TestCase):
  """
  Run the sequence memory tests on the C++ ExtendedTemporalMemory.
  """

  def constructTM(self, columnCount, cellsPerColumn, initialPermanence,
                  connectedPermanence, minThreshold, sampleSize,
                  permanenceIncrement, permanenceDecrement,
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
      "activationThreshold": activationThreshold,
      "seed": seed,
      "learnOnOneCell": False,
    }

    self.tm = ApicalTiebreakSequenceMemory(**params)


  def compute(self, activeColumns, learn):
    self.tm.compute(sorted(activeColumns), learn=learn)


  def reset(self):
    self.tm.reset()


  def getActiveCells(self):
    return self.tm.getActiveCells()


  def getPredictedCells(self):
    return self.tm.getPredictedCells()
