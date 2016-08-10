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

import unittest
import numpy

from htmresearch.algorithms.column_pooler import ColumnPooler


class ColumnPoolerTest(unittest.TestCase):
  """ Super simple test of the ColumnPooler region."""


  def testConstructor(self):
    """Create a simple instance and test the constructor."""

    pooler = ColumnPooler(
      inputWidth=2048*8,
      columnDimensions=[2048, 1],
      maxSynapsesPerSegment=2048*8
    )

    self.assertEqual(pooler.numberOfCells(), 2048, "Incorrect number of cells")

    self.assertEqual(pooler.numberOfInputs(), 16384,
                     "Incorrect number of inputs")


  def testInitialNullInputLearnMode(self):
    """Tests with no input in the beginning. """

    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      columnDimensions=[2048, 1],
      maxSynapsesPerSegment=2048 * 8
    )
    activatedCells = numpy.zeros(pooler.numberOfCells())

    # Should be no active cells in beginning
    self.assertEqual(len(pooler.getActiveCells()), 0,
                     "Incorrect number of active cells")

    # After computing with no input should have 40 active cells
    pooler.compute(feedforwardInput=set(), learn=True)
    activatedCells[pooler.getActiveCells()] = 1
    self.assertEqual(activatedCells.sum(), 40,
                     "Incorrect number of active cells")

    # Should be no active cells after reset
    pooler.reset()
    self.assertEqual(len(pooler.getActiveCells()), 0,
                     "Incorrect number of active cells")

    # After computing again with no input should have different 40 active cells
    pooler.compute(feedforwardInput=set(), learn=True)
    activatedCells[pooler.getActiveCells()] += 1
    self.assertLess((activatedCells>=2).sum(), 5,
                    "SDRs not sufficiently different")



if __name__ == "__main__":
  unittest.main()

