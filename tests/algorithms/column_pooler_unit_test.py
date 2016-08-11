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

import scipy.sparse as sparse

from htmresearch.algorithms.column_pooler import ColumnPooler, realDType


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

    # Computing again with no input should lead to different 40 active cells
    pooler.compute(feedforwardInput=set(), learn=True)
    activatedCells[pooler.getActiveCells()] += 1
    self.assertLess((activatedCells>=2).sum(), 5,
                    "SDRs not sufficiently different")


  def testInitialProximalLearning(self):
    """Tests the first few steps of proximal learning. """

    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      columnDimensions=[2048, 1],
      maxSynapsesPerSegment=2048 * 8
    )
    activatedCells = numpy.zeros(pooler.numberOfCells())

    # Get initial activity
    pooler.compute(feedforwardInput=set(range(0,40)), learn=True)
    activatedCells[pooler.getActiveCells()] = 1
    self.assertEqual(activatedCells.sum(), 40,
                     "Incorrect number of active cells")
    sum1 = sum(pooler.getActiveCells())

    # Ensure we've added correct number synapses on the active cells
    self.assertEqual(
      pooler.numberOfSynapses(pooler.getActiveCells()),
      800,
      "Incorrect number of nonzero permanences on active cells"
    )

    # Ensure they are all connected
    self.assertEqual(
      pooler.numberOfConnectedSynapses(pooler.getActiveCells()),
      800,
      "Incorrect number of connected synapses on active cells"
    )

    # If we call compute with different feedforward input we should
    # get the same set of active cells
    pooler.compute(feedforwardInput=set(range(100,140)), learn=True)
    self.assertEqual(sum1, sum(pooler.getActiveCells()),
                     "Activity is not consistent for same input")

    # Ensure we've added correct number of new synapses on the active cells
    self.assertEqual(
      pooler.numberOfSynapses(pooler.getActiveCells()),
      1600,
      "Incorrect number of nonzero permanences on active cells"
    )

    # Ensure they are all connected
    self.assertEqual(
      pooler.numberOfConnectedSynapses(pooler.getActiveCells()),
      1600,
      "Incorrect number of connected synapses on active cells"
    )

    # If we call compute with no input we should still
    # get the same set of active cells
    pooler.compute(feedforwardInput=set(), learn=True)
    self.assertEqual(sum1, sum(pooler.getActiveCells()),
                     "Activity is not consistent for same input")

    # Ensure we do actually add the number of synapses we want

    # In "learn new object mode", if we call compute with the same feedforward
    # input after reset we should not get the same set of active cells
    pooler.reset()
    pooler.compute(feedforwardInput=set(range(0,40)), learn=True)
    self.assertNotEqual(sum1, sum(pooler.getActiveCells()),
               "Activity should not be consistent for same input after reset")
    self.assertEqual(len(pooler.getActiveCells()), 40,
               "Incorrect number of active cells after reset")


  def testPickProximalInputsToLearnOn(self):
    """Test _pickProximalInputsToLearnOn method"""

    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      columnDimensions=[2048, 1],
      maxSynapsesPerSegment=2048 * 8
    )

    proximalPermanences = sparse.lil_matrix(
                (pooler.numberOfCells(), pooler.inputWidth),
                dtype=realDType)
    proximalPermanences[42,0:10] = 0.21

    # With no existing synapses, and number of inputs = newSynapseCount, should
    # return the full list
    inputs,existing = pooler._pickProximalInputsToLearnOn(newSynapseCount=10,
                                        cell=100,
                                        activeInputs=set(range(10)),
                                        proximalPermanences=proximalPermanences)
    self.assertEqual(sum(inputs),45,"Did not select correct inputs")
    self.assertEqual(len(existing),0,"Did not return correct existing inputs")

    # With no existing synapses, and number of inputs < newSynapseCount, should
    # return all inputs as synapses
    inputs,existing = pooler._pickProximalInputsToLearnOn(newSynapseCount=11,
                                        cell=100,
                                        activeInputs=set(range(10)),
                                        proximalPermanences=proximalPermanences)
    self.assertEqual(sum(inputs),45,"Did not select correct inputs")
    self.assertEqual(len(existing),0,"Did not return correct existing inputs")

    # With no existing synapses, and number of inputs > newSynapseCount
    # should return newSynapseCount indices
    inputs,existing = pooler._pickProximalInputsToLearnOn(newSynapseCount=9,
                                        cell=100,
                                        activeInputs=set(range(10)),
                                        proximalPermanences=proximalPermanences)
    self.assertEqual(len(inputs),9,"Did not select correct inputs")
    self.assertEqual(len(existing),0,"Did not return correct existing inputs")

    # With existing inputs to [0..9], should return [10..19]
    inputs,existing = pooler._pickProximalInputsToLearnOn(newSynapseCount=10,
                                        cell=42,
                                        activeInputs=set(range(20)),
                                        proximalPermanences=proximalPermanences)
    self.assertEqual(sum(inputs),145,"Did not select correct inputs")
    self.assertEqual(sum(existing),45,"Did not return correct existing inputs")

    # With existing inputs to [0..9], and active inputs [0..9] should
    # return none
    inputs,existing = pooler._pickProximalInputsToLearnOn(newSynapseCount=10,
                                        cell=42,
                                        activeInputs=set(range(10)),
                                        proximalPermanences=proximalPermanences)
    self.assertEqual(len(inputs),0,"Did not select correct inputs")
    self.assertEqual(sum(existing),45,"Did not return correct existing inputs")


  def testLearnProximal(self):
    """Test _learnProximal method"""

    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      columnDimensions=[2048, 1],
      maxSynapsesPerSegment=2048 * 8
    )

    proximalPermanences = sparse.lil_matrix(
                (pooler.numberOfCells(), pooler.inputWidth),
                dtype=realDType)
    proximalConnections = sparse.lil_matrix(
                (pooler.numberOfCells(), pooler.inputWidth),
                dtype=realDType)

    pooler._learnProximal(
      activeInputs=set(range(20)), activeCells=set(range(10)),
      maxNewSynapseCount=7, proximalPermanences=proximalPermanences,
      proximalConnections=proximalConnections,
      initialPermanence=0.2, synPermProximalInc=0.2, synPermProximalDec=0.1,
      connectedPermanence=0.3
    )

    # There should be exactly 7 * 10 new connections, each with permanence 0.2
    self.assertEqual(proximalPermanences.nnz, 70, "Incorrect number of synapses")
    self.assertAlmostEqual(proximalPermanences.sum(), 0.2*70,
                     msg="Incorrect permanence total", places=4)

    # Ensure the correct indices are there and there are no extra ones
    for cell in range(10):
      nz = proximalPermanences[cell].nonzero()[1]
      for i in nz:
        self.assertTrue(i in range(20), "Incorrect input index")

    self.assertEqual(proximalPermanences[10,:].sum(), 0.0, "Extra synapses exist")

    # Do another learning step to ensure increments and decrements are handled
    pooler._learnProximal(
      activeInputs=set(range(5,15)), activeCells=set(range(10)),
      maxNewSynapseCount=10, proximalPermanences=proximalPermanences,
      proximalConnections=proximalConnections,
      initialPermanence=0.2, synPermProximalInc=0.2, synPermProximalDec=0.1,
      connectedPermanence=0.3
    )

    # print proximalPermanences


if __name__ == "__main__":
  unittest.main()

