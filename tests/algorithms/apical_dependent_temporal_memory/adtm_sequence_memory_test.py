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
Run the sequence memory tests on the ApicalDependentTemporalMemory
"""

import random
import unittest

import numpy as np

from htmresearch.algorithms.apical_dependent_temporal_memory import (
  TripleMemory)
from htmresearch.support.shared_tests.sequence_memory_test_base import(
  SequenceMemoryTestBase)


class ApicalDependentTM_BasalSequenceMemoryTests(SequenceMemoryTestBase,
                                                 unittest.TestCase):
  """
  Run the sequence memory tests on the ApicalDependentTemporalMemory,
  passing the sequences in through basal input.
  """

  def constructTM(self, columnCount, cellsPerColumn, initialPermanence,
                  connectedPermanence, minThreshold, sampleSize,
                  permanenceIncrement, permanenceDecrement,
                  predictedSegmentDecrement, activationThreshold, seed):

    # Use the same apical input on every compute. This is like running the whole
    # experiment in one "world" or on one "object". It makes the
    # ApicalDependentTemporalMemory behave like traditional sequence memory.
    apicalInputSize = 1024
    self.constantApicalInput = np.array(
      sorted(random.sample(xrange(apicalInputSize), 40)),
      dtype="uint32")

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

      # This parameter wreaks havoc if we're holding the apical input constant.
      "apicalPredictedSegmentDecrement": 0.0,

      "activationThreshold": activationThreshold,
      "seed": seed,
      "basalInputSize": columnCount*cellsPerColumn,
      "apicalInputSize": apicalInputSize,
    }

    self.tm = TripleMemory(**params)


  def compute(self, activeColumns, learn):
    activeColumns = np.array(sorted(activeColumns), dtype="uint32")

    self.tm.compute(activeColumns,
                    basalInput=self.tm.getActiveCells(),
                    basalGrowthCandidates=self.tm.getWinnerCells(),
                    apicalInput=self.constantApicalInput,
                    apicalGrowthCandidates=self.constantApicalInput,
                    learn=learn)


  def reset(self):
    self.tm.reset()


  def getActiveCells(self):
    return self.tm.getActiveCells()


  def getPredictedCells(self):
    return self.tm.getPredictedCells()



class ApicalDependentTM_ApicalSequenceMemoryTests(SequenceMemoryTestBase,
                                                  unittest.TestCase):
  """
  Run the sequence memory tests on the ApicalDependentTemporalMemory,
  passing the sequences in through apical input.
  """

  def constructTM(self, columnCount, cellsPerColumn, initialPermanence,
                  connectedPermanence, minThreshold, sampleSize,
                  permanenceIncrement, permanenceDecrement,
                  predictedSegmentDecrement, activationThreshold, seed):

    # Use the same basal input on every compute. With this algorithm, basal and
    # apical segments are treated equally, so you can do sequence memory on the
    # apical segments. There might not be a good reason to do this, but it's
    # worth testing that the code is working as expected.
    basalInputSize = 1024
    self.constantBasalInput = np.array(
      sorted(random.sample(xrange(basalInputSize), 40)),
      dtype="uint32")

    params = {
      "columnCount": columnCount,
      "cellsPerColumn": cellsPerColumn,
      "initialPermanence": initialPermanence,
      "connectedPermanence": connectedPermanence,
      "minThreshold": minThreshold,
      "sampleSize": sampleSize,
      "permanenceIncrement": permanenceIncrement,
      "permanenceDecrement": permanenceDecrement,

      # This parameter wreaks havoc if we're holding the basal input constant.
      "basalPredictedSegmentDecrement": 0.0,

      "apicalPredictedSegmentDecrement": predictedSegmentDecrement,
      "activationThreshold": activationThreshold,
      "seed": seed,
      "basalInputSize": basalInputSize,
      "apicalInputSize": columnCount*cellsPerColumn,
    }

    self.tm = TripleMemory(**params)


  def compute(self, activeColumns, learn):
    activeColumns = np.array(sorted(activeColumns), dtype="uint32")

    self.tm.compute(activeColumns,
                    basalInput=self.constantBasalInput,
                    basalGrowthCandidates=self.constantBasalInput,
                    apicalInput=self.tm.getActiveCells(),
                    apicalGrowthCandidates=self.tm.getWinnerCells(),
                    learn=learn)


  def reset(self):
    self.tm.reset()


  def getActiveCells(self):
    return self.tm.getActiveCells()


  def getPredictedCells(self):
    return self.tm.getPredictedCells()
