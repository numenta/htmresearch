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

    self.assertEqual(
      pooler.numberOfSynapses(range(2048)),
      0,
      "Should be no synapses on initialization"
    )

    self.assertEqual(
      pooler.numberOfConnectedSynapses(range(2048)),
      0,
      "Should be no connected synapses on initialization"
    )


  def testInitialNullInputLearnMode(self):
    """Tests with no input in the beginning. """

    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      columnDimensions=[2048, 1],
      maxSynapsesPerSegment=2048 * 8
    )
    activatedCells = numpy.zeros(pooler.numberOfCells())

    # Should be no active cells in beginning
    self.assertEqual(
      len(pooler.getActiveCells()),
      0,
      "Incorrect number of active cells")

    # After computing with no input should have 40 active cells
    pooler.compute(feedforwardInput=set(), learn=True)
    activatedCells[pooler.getActiveCells()] = 1
    self.assertEqual(
      activatedCells.sum(),
      40,
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


  def testInitialInference(self):
    """Tests inference after learning one pattern. """

    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      columnDimensions=[2048, 1],
      maxSynapsesPerSegment=2048 * 8
    )
    activatedCells = numpy.zeros(pooler.numberOfCells())

    # Learn one pattern
    pooler.compute(feedforwardInput=set(range(0,40)), learn=True)
    activatedCells[pooler.getActiveCells()] = 1
    sum1 = sum(pooler.getActiveCells())

    # Inferring on same pattern should lead to same result
    pooler.reset()
    pooler.compute(feedforwardInput=set(range(0,40)), learn=False)
    self.assertEqual(sum1,
                     sum(pooler.getActiveCells()),
                     "Inference on pattern after learning it is incorrect")

    # Inferring with no inputs should maintain same pattern
    pooler.compute(feedforwardInput=set(), learn=False)
    self.assertEqual(sum1,
                     sum(pooler.getActiveCells()),
                     "Inference doesn't maintain activity with no input.")


  def testShortInferenceSequence(self):
    """Tests inference after learning two objects with two patterns. """

    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      columnDimensions=[2048, 1],
      maxSynapsesPerSegment=2048 * 8
    )
    activatedCells = numpy.zeros(pooler.numberOfCells())

    # Learn object one
    pooler.compute(feedforwardInput=set(range(0,40)), learn=True)
    activatedCells[pooler.getActiveCells()] = 1
    sum1 = sum(pooler.getActiveCells())

    pooler.compute(feedforwardInput=set(range(100,140)), learn=True)
    activatedCells[pooler.getActiveCells()] = 1
    self.assertEqual(sum1,
                     sum(pooler.getActiveCells()),
                     "Activity for second pattern is incorrect")

    # Learn object two
    pooler.reset()
    pooler.compute(feedforwardInput=set(range(1000,1040)), learn=True)
    activatedCells[pooler.getActiveCells()] = 1
    sum2 = sum(pooler.getActiveCells())

    pooler.compute(feedforwardInput=set(range(1100,1140)), learn=True)
    activatedCells[pooler.getActiveCells()] = 1
    self.assertEqual(sum2,
                     sum(pooler.getActiveCells()),
                     "Activity for second pattern is incorrect")

    # Inferring on patterns in first object should lead to same result, even
    # after gap
    pooler.reset()
    pooler.compute(feedforwardInput=set(range(100,140)), learn=False)
    self.assertEqual(sum1,
                     sum(pooler.getActiveCells()),
                     "Inference on pattern after learning it is incorrect")

    # Inferring with no inputs should maintain same pattern
    pooler.compute(feedforwardInput=set(), learn=False)
    self.assertEqual(sum1,
                     sum(pooler.getActiveCells()),
                     "Inference doesn't maintain activity with no input.")

    pooler.reset()
    pooler.compute(feedforwardInput=set(range(0,40)), learn=False)
    self.assertEqual(sum1,
                     sum(pooler.getActiveCells()),
                     "Inference on pattern after learning it is incorrect")

    # Inferring on patterns in second object should lead to same result, even
    # after gap
    pooler.reset()
    pooler.compute(feedforwardInput=set(range(1100,1140)), learn=False)
    self.assertEqual(sum2,
                     sum(pooler.getActiveCells()),
                     "Inference on pattern after learning it is incorrect")

    # Inferring with no inputs should maintain same pattern
    pooler.compute(feedforwardInput=set(), learn=False)
    self.assertEqual(sum2,
                     sum(pooler.getActiveCells()),
                     "Inference doesn't maintain activity with no input.")

    pooler.reset()
    pooler.compute(feedforwardInput=set(range(1000,1040)), learn=False)
    self.assertEqual(sum2,
                     sum(pooler.getActiveCells()),
                     "Inference on pattern after learning it is incorrect")


  def testPickProximalInputsToLearnOn(self):
    """Test _pickProximalInputsToLearnOn method"""

    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      columnDimensions=[2048, 1],
      maxSynapsesPerSegment=2048 * 8
    )

    proximalPermanences = pooler.proximalPermanences
    a = numpy.zeros(pooler.inputWidth, dtype=realDType)
    a[0:10] = 0.21
    proximalPermanences.setRowFromDense(42,a)

    cellNonZeros42,_ = proximalPermanences.rowNonZeros(42)
    cellNonZeros100,_ = proximalPermanences.rowNonZeros(100)

    # With no existing synapses, and number of inputs = newSynapseCount, should
    # return the full list
    inputs,existing = pooler._pickProximalInputsToLearnOn(newSynapseCount=10,
                                        activeInputs=set(range(10)),
                                        cellNonZeros=cellNonZeros100)
    self.assertEqual(sum(inputs),45,"Did not select correct inputs")
    self.assertEqual(len(existing),0,"Did not return correct existing inputs")

    # With no existing synapses, and number of inputs < newSynapseCount, should
    # return all inputs as synapses
    inputs,existing = pooler._pickProximalInputsToLearnOn(newSynapseCount=11,
                                        activeInputs=set(range(10)),
                                        cellNonZeros=cellNonZeros100)
    self.assertEqual(sum(inputs),45,"Did not select correct inputs")
    self.assertEqual(len(existing),0,"Did not return correct existing inputs")

    # With no existing synapses, and number of inputs > newSynapseCount
    # should return newSynapseCount indices
    inputs,existing = pooler._pickProximalInputsToLearnOn(newSynapseCount=9,
                                        activeInputs=set(range(10)),
                                        cellNonZeros=cellNonZeros100)
    self.assertEqual(len(inputs),9,"Did not select correct inputs")
    self.assertEqual(len(existing),0,"Did not return correct existing inputs")

    # With existing inputs to [0..9], should return [10..19]
    inputs,existing = pooler._pickProximalInputsToLearnOn(newSynapseCount=10,
                                        activeInputs=set(range(20)),
                                        cellNonZeros=cellNonZeros42)
    self.assertEqual(sum(inputs),145,"Did not select correct inputs")
    self.assertEqual(sum(existing),45,"Did not return correct existing inputs")

    # With existing inputs to [0..9], and active inputs [0..9] should
    # return none
    inputs,existing = pooler._pickProximalInputsToLearnOn(newSynapseCount=10,
                                        activeInputs=set(range(10)),
                                        cellNonZeros=cellNonZeros42)
    self.assertEqual(len(inputs),0,"Did not select correct inputs")
    self.assertEqual(sum(existing),45,"Did not return correct existing inputs")


  def testLearnProximal(self):
    """Test _learnProximal method"""

    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      columnDimensions=[2048, 1],
      maxSynapsesPerSegment=2048 * 8
    )
    proximalPermanences = pooler.proximalPermanences
    proximalConnections = pooler.proximalConnections

    pooler._learnProximal(
      activeInputs=set(range(20)), activeCells=set(range(10)),
      maxNewSynapseCount=7, proximalPermanences=proximalPermanences,
      proximalConnections=proximalConnections,
      initialPermanence=0.2, synPermProximalInc=0.2, synPermProximalDec=0.1,
      connectedPermanence=0.3
    )

    # There should be exactly 7 * 10 new connections, each with permanence 0.2
    self.assertEqual(proximalPermanences.nNonZeros(),
                     70,
                     "Incorrect number of synapses")

    self.assertAlmostEqual(proximalPermanences.sum(), 0.2*70,
                     msg="Incorrect permanence total", places=4)

    # Ensure the correct indices are there and there are no extra ones
    for cell in range(10):
      nz,_ = proximalPermanences.rowNonZeros(cell)
      for i in nz:
        self.assertTrue(i in range(20), "Incorrect input index")

    self.assertEqual(pooler.numberOfSynapses(range(10,2048)),
                     0,
                     "Extra synapses exist")

    # Do another learning step to ensure increments and decrements are handled
    pooler._learnProximal(
      activeInputs=set(range(5,15)), activeCells=set(range(10)),
      maxNewSynapseCount=5, proximalPermanences=proximalPermanences,
      proximalConnections=proximalConnections,
      initialPermanence=0.2, synPermProximalInc=0.2, synPermProximalDec=0.1,
      connectedPermanence=0.3
    )

    # Should be no synapses on cells that were never active
    self.assertEqual(pooler.numberOfSynapses(range(10,2048)),
                     0,
                     "Extra synapses exist")

    # Should be 12 synapses on cells that were active
    # Number of connected cells can vary, depending on how many from range
    # 10-14 were selected in the first learning step
    for cell in range(10):
      self.assertEqual(pooler.numberOfSynapses([cell]),
                       12,
                       "Incorrect number of synapses on active cell")

      self.assertGreater(pooler.numberOfConnectedSynapses([cell]),
                         0,
                         "Must be at least one connected synapse on cell.")

      cellNonZeroIndices, cellPerms = proximalPermanences.rowNonZeros(cell)
      self.assertAlmostEqual(min(cellPerms), 0.1, 3,
                             "Must be at least one decremented permanence.")


  def testLearningWithLateralInputs(self):
    """
    With lateral inputs from other columns, test that some distal segments are
    learned on a stable set of SDRs for each new feed forward object
    """
    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      columnDimensions=[2048, 1],
      numNeighboringColumns=2,
      initialPermanence=0.41,
    )

    # Get initial SDR for first object from pooler
    pooler.compute(feedforwardInput=set(range(0,40)),
                   activeExternalCells=set(range(100,140)),
                   learn=True)
    activeCells = pooler.getActiveCells()

    # Cells corresponding to that initial SDR should now start learning
    # on their distal segments.
    pooler.compute(feedforwardInput=set(range(40,80)),
                   activeExternalCells=set(range(100,140)),
                   learn=True)
    self.assertEqual(pooler.numberOfDistalSegments(activeCells),
                     40,
                     "Incorrect number of segments after learning")
    self.assertEqual(pooler.numberOfDistalSynapses(activeCells),
                     40*20,
                     "Incorrect number of synapses after learning")

    # Cells corresponding to that initial SDR should continue to learn new
    # synapses on that same set of segments. There should be no
    # segments on any other cells
    pooler.compute(feedforwardInput=set(range(80,120)),
                   activeExternalCells=set(range(100,140)),
                   learn=True)

    self.assertEqual(pooler.numberOfDistalSegments(activeCells),
                     40,
                     "Incorrect number of segments after learning")
    self.assertEqual(pooler.numberOfDistalSegments(range(2048)),
                     40,
                     "Extra segments on other cells after learning")
    self.assertEqual(pooler.numberOfDistalSynapses(activeCells),
                     40*40,
                     "Incorrect number of synapses after learning")


    # Get SDR for second object from pooler
    pooler.reset()
    pooler.compute(feedforwardInput=set(range(120,160)),
                   activeExternalCells=set(range(200,240)),
                   learn=True)
    activeCellsObject2 = pooler.getActiveCells()
    uniqueCellsObject2 = set(activeCellsObject2) - set(activeCells)
    numCommonCells = len(set(activeCells).intersection(set(activeCellsObject2)))

    # Cells corresponding to that initial SDR should now start learning
    # on their distal segments.
    pooler.compute(feedforwardInput=set(range(160,200)),
                   activeExternalCells=set(range(200,240)),
                   learn=True)
    self.assertEqual(pooler.numberOfDistalSegments(uniqueCellsObject2),
                     len(uniqueCellsObject2),
                     "Incorrect number of segments after learning")
    self.assertEqual(pooler.numberOfDistalSynapses(uniqueCellsObject2),
                     len(uniqueCellsObject2)*20,
                     "Incorrect number of synapses after learning")
    self.assertLess(numCommonCells, 5, "Too many common cells across objects")


    # Cells corresponding to that initial SDR should continue to learn new
    # synapses on that same set of segments. There should be no
    # segments on any other cells
    pooler.compute(feedforwardInput=set(range(200,240)),
                   activeExternalCells=set(range(200,240)),
                   learn=True)

    self.assertEqual(pooler.numberOfDistalSegments(uniqueCellsObject2),
                     len(uniqueCellsObject2),
                     "Incorrect number of segments after learning")
    self.assertEqual(pooler.numberOfDistalSegments(range(2048)),
                     40*2,
                     "Extra segments on other cells after learning")
    self.assertEqual(pooler.numberOfDistalSynapses(uniqueCellsObject2),
                     len(uniqueCellsObject2)*40,
                     "Incorrect number of synapses after learning")



  @unittest.skip("Method might not be required")
  def testNumberOfActiveDistalSegments(self):
    """Tests the function counting the number of active distal segments."""

    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      columnDimensions=[2048, 1],
      maxSynapsesPerDistalSegment=50,
      numNeighboringColumns=4,
      distalActivationThreshold=1
    )

    for connections in pooler.distalConnections:
      seg = connections.createSegment(1)
      _ = connections.createSynapse(seg, 4, 0.7)
      _ = connections.createSynapse(seg, 5, 0.7)

    numSegments = pooler._numberOfActiveDistalSegments(
      cell=1,
      lateralInput=[{4, 5}] * 4,
      connectedPermanence=pooler.connectedPermanence,
      activationThreshold=pooler.distalActivationThreshold,
    )
    self.assertEqual(numSegments, 4)

    for connections in pooler.distalConnections:
      seg = connections.createSegment(2)
      _ = connections.createSynapse(seg, 4, 0.7)
      _ = connections.createSynapse(seg, 3, 0.3)

    numSegments = pooler._numberOfActiveDistalSegments(
      cell=1,
      lateralInput=[{4, 5}, {3}, {4}, {4, 5}],
      connectedPermanence=pooler.connectedPermanence,
      activationThreshold=pooler.distalActivationThreshold,
    )
    self.assertEqual(numSegments, 3)

    numSegments = pooler._numberOfActiveDistalSegments(
      cell=2,
      lateralInput=[{4}, {3}, {1, 2, 3}, {7, 8, 9}],
      connectedPermanence=pooler.connectedPermanence,
      activationThreshold=pooler.distalActivationThreshold,
    )
    self.assertEqual(numSegments, 1)

    numSegments = pooler._numberOfActiveDistalSegments(
      cell=3,
      lateralInput=[{4}, {3}, {1, 2, 3}, {7, 8, 9}],
      connectedPermanence=pooler.connectedPermanence,
      activationThreshold=pooler.distalActivationThreshold,
    )
    self.assertEqual(numSegments, 0)


  @unittest.skip("While working on algorithm")
  def testWinnersBasedOnLateralActivity(self):
    """Tests that the correct winners always get chosen."""

    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      columnDimensions=[2048, 1],
      maxSynapsesPerDistalSegment=50,
      numNeighboringColumns=4,
      distalActivationThreshold=1
    )

    proximallyActivatedCells = {1, 2 ,6}
    previouslyActiveCells = {2, 3, 4}
    pooler.activeCells = previouslyActiveCells

    # no lateral input, proximally activated cells should win
    cells = pooler._winnersBasedOnLateralActivity(
      activeCells=proximallyActivatedCells,
      lateralInput=[],
      minThreshold=pooler.distalMinThreshold
    )
    self.assertEqual(cells, proximallyActivatedCells)

    # create the same synapses as previously
    for connections in pooler.distalConnections:
      seg = connections.createSegment(1)
      _ = connections.createSynapse(seg, 4, 0.7)
      _ = connections.createSynapse(seg, 5, 0.7)

    for connections in pooler.distalConnections:
      seg = connections.createSegment(2)
      _ = connections.createSynapse(seg, 4, 0.7)
      _ = connections.createSynapse(seg, 3, 0.3)

    lateralInput = [{4, 5}, {3}, {3}, {4, 5}]
    pooler.activeCells = previouslyActiveCells
    cells = pooler._winnersBasedOnLateralActivity(
      activeCells=proximallyActivatedCells,
      lateralInput=lateralInput,
      minThreshold=pooler.distalMinThreshold
    )
    self.assertEqual(cells, {1, 2})

    # test competition between lateral input
    lateralInput = [{2, 5}, {3}, {1}, {2, 5}]
    pooler.activeCells = previouslyActiveCells
    cells = pooler._winnersBasedOnLateralActivity(
      activeCells=proximallyActivatedCells,
      lateralInput=lateralInput,
      minThreshold=pooler.distalMinThreshold
    )
    self.assertEqual(cells, {1})


if __name__ == "__main__":
  unittest.main()

