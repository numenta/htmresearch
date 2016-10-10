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

from htmresearch.algorithms.column_pooler import ColumnPooler, realDType
from htmresearch.support.column_pooler_mixin import ColumnPoolerMonitorMixin


class MonitoredColumnPooler(ColumnPoolerMonitorMixin, ColumnPooler):
  pass


class ColumnPoolerTest(unittest.TestCase):
  """
  Simplistic tests of the ColumnPooler region, focusing on underlying
  implementation.
  """

  def _initializeDefaultPooler(self):
    """Initialize and return a default ColumnPooler """
    return ColumnPooler(
      inputWidth=2048 * 8,
      lateralInputWidth=4096,
      columnDimensions=(2048,),
      initialPermanence=0.41,
      # Temporary: a high maxNewSynapseCount is in place until NUP #3268 is
      # addressed
      maxNewProximalSynapseCount=40,
      maxNewDistalSynapseCount=40,
    )


  def testConstructor(self):
    """Create a simple instance and test the constructor."""

    pooler = self._initializeDefaultPooler()

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

    pooler = self._initializeDefaultPooler()
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

    pooler = MonitoredColumnPooler(
      inputWidth=2048 * 8,
      lateralInputWidth=512,
      columnDimensions=[2048, 1],
      initialPermanence=0.41,
      # Temporary: a high maxNewSynapseCount is in place until NUP #3268 is
      # addressed
      maxNewProximalSynapseCount=40,
      maxNewDistalSynapseCount=40,
    )

    activatedCells = numpy.zeros(pooler.numberOfCells())

    # Get initial activity
    pooler.compute(feedforwardInput=set(range(0, 40)), learn=True)
    activatedCells[pooler.getActiveCells()] = 1
    self.assertEqual(activatedCells.sum(), 40,
                     "Incorrect number of active cells")
    sum1 = sum(pooler.getActiveCells())

    # Ensure we've added correct number synapses on the active cells
    self.assertEqual(
      pooler.mmGetTraceNumProximalSynapses().data[-1],
      1600,
      "Incorrect number of nonzero permanences on active cells"
    )

    # Ensure they are all connected
    self.assertEqual(
      pooler.numberOfConnectedSynapses(pooler.getActiveCells()),
      1600,
      "Incorrect number of connected synapses on active cells"
    )

    # As multiple different feedforward inputs come in, the same set of cells
    # should be active.
    pooler.compute(feedforwardInput=set(range(100, 140)), learn=True)
    self.assertEqual(sum1, sum(pooler.getActiveCells()),
                     "Activity is not consistent for same input")

    # Ensure we've added correct number of new synapses on the active cells
    self.assertEqual(
      pooler.mmGetTraceNumProximalSynapses().data[-1],
      3200,
      "Incorrect number of nonzero permanences on active cells"
    )

    # Ensure they are all connected
    self.assertEqual(
      pooler.numberOfConnectedSynapses(pooler.getActiveCells()),
      3200,
      "Incorrect number of connected synapses on active cells"
    )

    # If there is no feedforward input we should still get the same set of
    # active cells
    pooler.compute(feedforwardInput=set(), learn=True)
    self.assertEqual(sum1, sum(pooler.getActiveCells()),
                     "Activity is not consistent for same input")

    # Ensure we do actually add the number of synapses we want

    # In "learn new object mode", given a familiar feedforward input after reset
    # we should not get the same set of active cells
    pooler.reset()
    pooler.compute(feedforwardInput=set(range(0, 40)), learn=True)
    self.assertNotEqual(sum1, sum(pooler.getActiveCells()),
               "Activity should not be consistent for same input after reset")
    self.assertEqual(len(pooler.getActiveCells()), 40,
               "Incorrect number of active cells after reset")


  def testInitialInference(self):
    """Tests inference after learning one pattern. """

    pooler = self._initializeDefaultPooler()
    activatedCells = numpy.zeros(pooler.numberOfCells())

    # Learn one pattern
    pooler.compute(feedforwardInput=set(range(0, 40)), learn=True)
    activatedCells[pooler.getActiveCells()] = 1
    sum1 = sum(pooler.getActiveCells())

    # Inferring on same pattern should lead to same result
    pooler.reset()
    pooler.compute(feedforwardInput=set(range(0, 40)), learn=False)
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

    pooler = self._initializeDefaultPooler()
    activatedCells = numpy.zeros(pooler.numberOfCells())

    # Learn object one
    pooler.compute(feedforwardInput=set(range(0, 40)), learn=True)
    activatedCells[pooler.getActiveCells()] = 1
    sum1 = sum(pooler.getActiveCells())

    pooler.compute(feedforwardInput=set(range(100, 140)), learn=True)
    activatedCells[pooler.getActiveCells()] = 1
    self.assertEqual(sum1,
                     sum(pooler.getActiveCells()),
                     "Activity for second pattern is incorrect")

    # Learn object two
    pooler.reset()
    pooler.compute(feedforwardInput=set(range(1000, 1040)), learn=True)
    activatedCells[pooler.getActiveCells()] = 1
    sum2 = sum(pooler.getActiveCells())

    pooler.compute(feedforwardInput=set(range(1100, 1140)), learn=True)
    activatedCells[pooler.getActiveCells()] = 1
    self.assertEqual(sum2,
                     sum(pooler.getActiveCells()),
                     "Activity for second pattern is incorrect")

    # Inferring on patterns in first object should lead to same result, even
    # after gap
    pooler.reset()
    pooler.compute(feedforwardInput=set(range(100, 140)), learn=False)
    self.assertEqual(sum1,
                     sum(pooler.getActiveCells()),
                     "Inference on pattern after learning it is incorrect")

    # Inferring with no inputs should maintain same pattern
    pooler.compute(feedforwardInput=set(), learn=False)
    self.assertEqual(sum1,
                     sum(pooler.getActiveCells()),
                     "Inference doesn't maintain activity with no input.")

    pooler.reset()
    pooler.compute(feedforwardInput=set(range(0, 40)), learn=False)
    self.assertEqual(sum1,
                     sum(pooler.getActiveCells()),
                     "Inference on pattern after learning it is incorrect")

    # Inferring on patterns in second object should lead to same result, even
    # after gap
    pooler.reset()
    pooler.compute(feedforwardInput=set(range(1100, 1140)), learn=False)
    self.assertEqual(sum2,
                     sum(pooler.getActiveCells()),
                     "Inference on pattern after learning it is incorrect")

    # Inferring with no inputs should maintain same pattern
    pooler.compute(feedforwardInput=set(), learn=False)
    self.assertEqual(sum2,
                     sum(pooler.getActiveCells()),
                     "Inference doesn't maintain activity with no input.")

    pooler.reset()
    pooler.compute(feedforwardInput=set(range(1000, 1040)), learn=False)
    self.assertEqual(sum2,
                     sum(pooler.getActiveCells()),
                     "Inference on pattern after learning it is incorrect")


  def testProximalLearning_Growth_MaxNewSynapseCount(self):
    """
    When the number of available active input bits is = maxNewSynapseCount,
    cells should grow synapses to every active input bit.

    """
    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      lateralInputWidth=512,
      numActiveColumnsPerInhArea=12,
      initialProximalPermanence=0.60,
      connectedPermanence=0.50,
      maxNewProximalSynapseCount=10,
      maxNewDistalSynapseCount=10,
    )

    feedforwardInput = set(range(10))

    pooler.compute(feedforwardInput, learn=True)

    activeCells = pooler.getActiveCells()
    self.assertEqual(len(activeCells), 12)

    for cell in activeCells:
      self.assertEqual(pooler.numberOfSynapses([cell]), 10,
                       "Should connect to every active input bit.")
      self.assertEqual(pooler.numberOfConnectedSynapses([cell]), 10,
                       "Each synapse should be marked as connected.")

      (presynapticCells,
       permanences) = pooler.proximalPermanences.rowNonZeros(cell)

      self.assertEqual(set(presynapticCells), feedforwardInput,
                       "Should connect to every active input bit.")
      for perm in permanences:
        self.assertAlmostEqual(perm, 0.60,
                               msg="Should use 'initialProximalPermanence'.")


  def testProximalLearning_Growth_FewActiveInputBits(self):
    """
    When the number of available active input bits is < maxNewSynapseCount,
    cells should grow synapses to every active input bit.

    """
    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      lateralInputWidth=512,
      numActiveColumnsPerInhArea=12,
      initialProximalPermanence=0.60,
      connectedPermanence=0.50,
      maxNewProximalSynapseCount=10,
      maxNewDistalSynapseCount=10,
    )

    feedforwardInput = set(range(9))

    pooler.compute(feedforwardInput, learn=True)

    activeCells = pooler.getActiveCells()
    self.assertEqual(len(activeCells), 12)

    for cell in activeCells:
      self.assertEqual(pooler.numberOfSynapses([cell]), 9,
                       "Should connect to every active input bit.")
      self.assertEqual(pooler.numberOfConnectedSynapses([cell]), 9,
                       "Each synapse should be marked as connected.")

      (presynapticCells,
       permanences) = pooler.proximalPermanences.rowNonZeros(cell)

      self.assertEqual(set(presynapticCells), feedforwardInput,
                       "Should connect to every active input bit.")
      for perm in permanences:
        self.assertAlmostEqual(perm, 0.60,
                               msg="Should use 'initialProximalPermanence'.")


  def testProximalLearning_Growth_ManyActiveInputBits(self):
    """
    When the number of available active input bits is > maxNewSynapseCount,
    each cell should grow 'maxNewSynapseCount' synapses.

    """
    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      lateralInputWidth=512,
      numActiveColumnsPerInhArea=12,
      initialProximalPermanence=0.60,
      connectedPermanence=0.50,
      maxNewProximalSynapseCount=10,
      maxNewDistalSynapseCount=10,
    )

    feedforwardInput = set(range(11))

    pooler.compute(feedforwardInput, learn=True)

    activeCells = pooler.getActiveCells()
    self.assertEqual(len(activeCells), 12)

    for cell in activeCells:
      self.assertEqual(pooler.numberOfSynapses([cell]), 10,
                       "Should connect to every active input bit.")
      self.assertEqual(pooler.numberOfConnectedSynapses([cell]), 10,
                       "Each synapse should be marked as connected.")

      (presynapticCells,
       permanences) = pooler.proximalPermanences.rowNonZeros(cell)

      self.assertTrue(set(presynapticCells).issubset(feedforwardInput),
                      "Should connect to a subset of the active input bits.")
      for perm in permanences:
        self.assertAlmostEqual(perm, 0.60,
                               msg="Should use 'initialProximalPermanence'.")


  def testProximalLearning_SubsequentGrowth(self):
    """
    When all of the active input bits are synapsed, don't grow new synapses.
    When some of them are not synapsed, grow new synapses to them.

    """
    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      lateralInputWidth=512,
      numActiveColumnsPerInhArea=12,
      synPermProximalInc=0.0,
      synPermProximalDec=0.0,
      initialProximalPermanence=0.60,
      connectedPermanence=0.50,
      maxNewProximalSynapseCount=10,
      maxNewDistalSynapseCount=10,
    )

    # Grow synapses.
    pooler.compute(set(range(10)), learn=True)
    for cell in pooler.getActiveCells():
      self.assertEqual(pooler.numberOfSynapses([cell]), 10,
                       "Should connect to every active input bit.")

    # Given the same input, no new synapses should form.
    pooler.compute(set(range(10)), learn=True)
    for cell in pooler.getActiveCells():
      self.assertEqual(pooler.numberOfSynapses([cell]), 10,
                       "No new synapses should form.")

    # Given a superset of the input, some new synapses should form.
    pooler.compute(set(range(20)), learn=True)
    for cell in pooler.getActiveCells():
      self.assertEqual(pooler.numberOfSynapses([cell]), 20,
                       "Should connect to the new active input bits.")
      self.assertEqual(pooler.numberOfConnectedSynapses([cell]), 20,
                       "Each synapse should be marked as connected.")


  def testProximalLearning_InitiallyDisconnected(self):
    """
    If the initialProximalPermanence is below the connectedPermanence, new
    synapses should not be marked as connected.

    """
    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      lateralInputWidth=512,
      numActiveColumnsPerInhArea=12,
      initialProximalPermanence=0.45,
      connectedPermanence=0.50,
      maxNewProximalSynapseCount=10,
      maxNewDistalSynapseCount=10,
    )

    feedforwardInput = set(range(10))

    pooler.compute(feedforwardInput, learn=True)

    activeCells = pooler.getActiveCells()
    self.assertEqual(len(activeCells), 12)

    for cell in activeCells:
      self.assertEqual(pooler.numberOfSynapses([cell]), 10,
                       "Should connect to every active input bit.")
      self.assertEqual(pooler.numberOfConnectedSynapses([cell]), 0,
                       "The synapses shouldn't have a high enough permanence"
                       " to be connected.")


  def testProximalLearning_ReinforceExisting(self):
    """
    When a cell has a synapse to an active input bit, increase its permanence by
    'synPermProximalInc'.

    """
    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      lateralInputWidth=512,
      numActiveColumnsPerInhArea=12,
      synPermProximalInc=0.1,
      synPermProximalDec=0.0,
      initialProximalPermanence=0.45,
      connectedPermanence=0.50,
      maxNewProximalSynapseCount=10,
      maxNewDistalSynapseCount=10,
    )

    # Grow some synapses.
    pooler.compute(set(range(0, 10)), learn=True)
    pooler.compute(set(range(10, 20)), learn=True)

    # Reinforce some of them.
    pooler.compute(set(range(0, 15)), learn=True)

    activeCells = pooler.getActiveCells()
    self.assertEqual(len(activeCells), 12)

    for cell in activeCells:
      self.assertEqual(pooler.numberOfSynapses([cell]), 20,
                       "Should connect to every active input bit.")
      self.assertEqual(pooler.numberOfConnectedSynapses([cell]), 15,
                       "Each reinforced synapse should be marked as connected.")

      (presynapticCells,
       permanences) = pooler.proximalPermanences.rowNonZeros(cell)

      d = dict(zip(presynapticCells, permanences))
      for presynapticCell in xrange(0, 15):
        perm = d[presynapticCell]
        self.assertAlmostEqual(
          perm, 0.55,
          msg=("Should have permanence of 'initialProximalPermanence'"
               " + 'synPermProximalInc'."))
      for presynapticCell in xrange(15, 20):
        perm = d[presynapticCell]
        self.assertAlmostEqual(
          perm, 0.45,
          msg="Should have permanence of 'initialProximalPermanence'")


  def testProximalLearning_PunishExisting(self):
    """
    When a cell has a synapse to an inactive input bit, decrease its permanence
    by 'synPermProximalDec'.

    """
    pooler = ColumnPooler(
      inputWidth=2048 * 8,
      lateralInputWidth=512,
      numActiveColumnsPerInhArea=12,
      synPermProximalInc=0.0,
      synPermProximalDec=0.1,
      initialProximalPermanence=0.55,
      connectedPermanence=0.50,
      maxNewProximalSynapseCount=10,
      maxNewDistalSynapseCount=10,
    )

    # Grow some synapses.
    pooler.compute(set(range(0, 10)), learn=True)

    # Punish some of them.
    pooler.compute(set(range(0, 5)), learn=True)

    activeCells = pooler.getActiveCells()
    self.assertEqual(len(activeCells), 12)

    for cell in activeCells:
      self.assertEqual(pooler.numberOfSynapses([cell]), 10,
                       "Should connect to every active input bit.")
      self.assertEqual(pooler.numberOfConnectedSynapses([cell]), 5,
                       "Each punished synapse should no longer be marked as"
                       " connected.")

      (presynapticCells,
       permanences) = pooler.proximalPermanences.rowNonZeros(cell)

      d = dict(zip(presynapticCells, permanences))
      for presynapticCell in xrange(0, 5):
        perm = d[presynapticCell]
        self.assertAlmostEqual(
          perm, 0.55,
          msg="Should have permanence of 'initialProximalPermanence'")
      for presynapticCell in xrange(5, 10):
        perm = d[presynapticCell]
        self.assertAlmostEqual(
          perm, 0.45,
          msg=("Should have permanence of 'initialProximalPermanence'"
               " - 'synPermProximalDec'."))


  def testLearnProximal(self):
    """Test _learnProximal method"""

    pooler = self._initializeDefaultPooler()
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

    self.assertEqual(pooler.numberOfSynapses(range(10, 2048)),
                     0,
                     "Extra synapses exist")

    # Do another learning step to ensure increments and decrements are handled
    pooler._learnProximal(
      activeInputs=set(range(5, 15)), activeCells=set(range(10)),
      maxNewSynapseCount=5, proximalPermanences=proximalPermanences,
      proximalConnections=proximalConnections,
      initialPermanence=0.2, synPermProximalInc=0.2, synPermProximalDec=0.1,
      connectedPermanence=0.3
    )

    # Should be no synapses on cells that were never active
    self.assertEqual(pooler.numberOfSynapses(range(10, 2048)),
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
    pooler = self._initializeDefaultPooler()

    # Object 1
    lateralInput1 = set(xrange(100, 140))
    pooler.compute(set(xrange(0, 40)), lateralInput1, learn=True)

    # Get initial SDR for first object from pooler.
    activeCells = pooler.getActiveCells()

    # Cells corresponding to that initial SDR should have started learning on
    # their distal segments.
    self.assertEqual(pooler.numberOfDistalSegments(activeCells),
                     40,
                     "Incorrect number of segments after learning")
    self.assertEqual(pooler.numberOfDistalSynapses(activeCells),
                     40*40,
                     "Incorrect number of synapses after learning")

    # Cells corresponding to that initial SDR should continue to learn new
    # synapses on that same set of segments. There should be no segments on any
    # other cells.
    pooler.compute(set(xrange(40, 80)), lateralInput1, learn=True)

    self.assertEqual(pooler.numberOfDistalSegments(activeCells),
                     40,
                     "Incorrect number of segments after learning")
    self.assertEqual(pooler.numberOfDistalSegments(range(2048)),
                     40,
                     "Extra segments on other cells after learning")
    self.assertEqual(pooler.numberOfDistalSynapses(activeCells),
                     40*40,
                     "Incorrect number of synapses after learning")


    # Object 2
    pooler.reset()
    lateralInput2 = set(xrange(200, 240))
    pooler.compute(set(xrange(120, 160)), lateralInput2, learn=True)

    # Get initial SDR for second object from pooler.
    activeCellsObject2 = pooler.getActiveCells()
    uniqueCellsObject2 = set(activeCellsObject2) - set(activeCells)
    numCommonCells = len(set(activeCells).intersection(set(activeCellsObject2)))


    # Cells corresponding to that initial SDR should have started learning
    # on their distal segments.
    self.assertEqual(pooler.numberOfDistalSegments(uniqueCellsObject2),
                     len(uniqueCellsObject2),
                     "Incorrect number of segments after learning")
    self.assertEqual(pooler.numberOfDistalSynapses(uniqueCellsObject2),
                     len(uniqueCellsObject2)*40,
                     "Incorrect number of synapses after learning")
    self.assertLess(numCommonCells, 5, "Too many common cells across objects")


    # Cells corresponding to that initial SDR should continue to learn new
    # synapses on that same set of segments. There should be no segments on any
    # other cells.
    pooler.compute(set(xrange(160, 200)), lateralInput2, learn=True)
    self.assertEqual(pooler.numberOfDistalSegments(uniqueCellsObject2),
                     len(uniqueCellsObject2),
                     "Incorrect number of segments after learning")
    self.assertEqual(pooler.numberOfDistalSynapses(uniqueCellsObject2),
                     len(uniqueCellsObject2)*40,
                     "Incorrect number of synapses after learning")


  def testInferenceWithLateralInputs(self):
    """
    After learning two objects, test that inference behaves as expected in
    a variety of scenarios.
    """
    pooler = self._initializeDefaultPooler()

    # Feed-forward representations:
    # Object 1 = union(range(0,40), range(40,80), range(80,120))
    # Object 2 = union(range(120, 160), range(160,200), range(200,240))
    feedforwardInputs = [
      [set(range(0, 40)), set(range(40, 80)), set(range(80, 120))],
      [set(range(120, 160)), set(range(160, 200)), set(range(200, 240))]
    ]

    # Lateral representations:
    # Object 1, Col 1 = range(200,240)
    # Object 2, Col 1 = range(240,280)
    # Object 1, Col 2 = range(2300,2340)
    # Object 2, Col 2 = range(2340,2380)
    lateralInputs = [
      [set(range(200, 240)), set(range(240, 280))],      # External column 1
      [set(range(2300, 2340)), set(range(2340, 2380))]   # External column 2
    ]

    # Train pooler on two objects, three iterations per object
    objectRepresentations = []
    for obj in range(2):
      pooler.reset()
      lateralInput = lateralInputs[0][obj] | lateralInputs[1][obj]
      for i in range(3): # three iterations
        for f in range(3): # three features per object
          pooler.compute(feedforwardInputs[obj][f], lateralInput, learn=True)

      objectRepresentations += [set(pooler.getActiveCells())]

    # With no lateral support, BU for O1 feature 0 + O2 feature 1.
    # End up with object representations for O1+O2.
    pooler.reset()
    pooler.compute(feedforwardInput= (feedforwardInputs[0][0] |
                                      feedforwardInputs[1][1]),
                   learn=False)
    self.assertEqual(set(pooler.getActiveCells()),
                     objectRepresentations[0] | objectRepresentations[1],
           "Incorrect object representations - expecting union of objects")

    # If you now get no input, should maintain the representation
    pooler.compute(feedforwardInput=set(), learn=False)
    self.assertEqual(set(pooler.getActiveCells()),
                     objectRepresentations[0] | objectRepresentations[1],
           "Incorrect object representations - expecting union is maintained")


    # Test case where you have two objects in bottom up representation, but
    # only one in lateral. In this case the laterally supported object
    # should dominate.

    # Test lateral from first column
    pooler.reset()
    pooler.compute(feedforwardInput=(feedforwardInputs[0][0] |
                                     feedforwardInputs[1][1]),
                   lateralInput=lateralInputs[0][0],
                   learn=False)
    self.assertEqual(set(pooler.getActiveCells()),
                     objectRepresentations[0],
           "Incorrect object representations - expecting single object")

    # Test lateral from second column
    pooler.reset()
    pooler.compute(feedforwardInput=(feedforwardInputs[0][0] |
                                     feedforwardInputs[1][1]),
                   lateralInput=lateralInputs[1][0],
                   learn=False)
    self.assertEqual(set(pooler.getActiveCells()),
                     objectRepresentations[0],
           "Incorrect object representations - expecting single object")

    # Test lateral from both columns
    pooler.reset()
    pooler.compute(feedforwardInput=(feedforwardInputs[0][0] |
                                     feedforwardInputs[1][1]),
                   lateralInput=(lateralInputs[0][0] |
                                 lateralInputs[1][0]),
                   learn=False)
    self.assertEqual(set(pooler.getActiveCells()),
                     objectRepresentations[0],
           "Incorrect object representations - expecting single object")


    # Test case where you have bottom up for O1, and lateral for O2. In
    # this case the bottom up one, O1, should dominate.
    pooler.reset()
    pooler.compute(feedforwardInput=feedforwardInputs[0][0],
                   lateralInput=lateralInputs[1][1],
                   learn=False)
    self.assertEqual(set(pooler.getActiveCells()),
                     objectRepresentations[0],
           "Incorrect object representations - expecting first object")

    # Test case where you have BU support O1+O2 with no lateral input Then see
    # no input but get lateral support for O1. Should converge to O1 only.
    pooler.reset()
    pooler.compute(feedforwardInput=(feedforwardInputs[0][0] |
                                     feedforwardInputs[1][1]),
                   lateralInput=set(),
                   learn=False)

    # No bottom input, but lateral support for O1
    pooler.compute(feedforwardInput=set(),
                   lateralInput=(lateralInputs[0][0] | lateralInputs[1][0]),
                   learn=False)

    self.assertEqual(set(pooler.getActiveCells()),
                     objectRepresentations[0],
           "Incorrect object representations - expecting first object")


    # TODO: more tests we could write:
    # Test case where you have two objects in bottom up representation, and
    # same two in lateral. End up with both active.

    # Test case where you have O1, O2 in bottom up representation, but
    # O1, O3 in lateral. In this case should end up with O1.

    # Test case where you have BU support for two objects, less than adequate
    # lateral support (below threshold) for one of them. Should end up with both
    # BU objects.


  def testInferenceWithChangingLateralInputs(self):
    """
    # Test case where the lateral inputs change while learning an object.
    # The same distal segments should continue to sample from the new inputs.
    # During inference any of these lateral inputs should cause the pooler
    # to disambiguate appropriately.
    """
    pooler = self._initializeDefaultPooler()

    # Feed-forward representations:
    # Object 1 = union(range(0,40), range(40,80), range(80,120))
    # Object 2 = union(range(120, 160), range(160,200), range(200,240))
    feedforwardInputs = [
      [set(range(0, 40)), set(range(40, 80)), set(range(80, 120))],
      [set(range(120, 160)), set(range(160, 200)), set(range(200, 240))]
    ]

    # Lateral representations:
    # Object 1, Col 1 = range(200,240)
    # Object 2, Col 1 = range(240,280)
    # Object 1, Col 2 = range(2300,2340)
    # Object 2, Col 2 = range(2340,2380)
    lateralInputs = [
      [set(range(200, 240)), set(range(240, 280))],      # External column 1
      [set(range(2300, 2340)), set(range(2340, 2380))]   # External column 2
    ]

    # Train pooler on two objects. For each object we go through three
    # iterations using just lateral input from first column. Then repeat with
    # second column.
    objectRepresentations = []
    for obj in range(2):
      pooler.reset()
      for col in range(2):
        lateralInput = lateralInputs[col][obj]
        for i in range(3): # three iterations
          for f in range(3): # three features per object
            pooler.compute(feedforwardInputs[obj][f], lateralInput, learn=True)
      objectRepresentations += [set(pooler.getActiveCells())]

    # We want to ensure that the learning for each cell happens on one distal
    # segment only. Some cells could in theory be common across both
    # representations so we just check the unique ones.
    # TODO: this test currently fails due to NuPIC issue #3268
    # uniqueCells = objectRepresentations[0].symmetric_difference(
    #                                                   objectRepresentations[1])
    # connections = pooler.tm.connections
    # for cell in uniqueCells:
    #   self.assertEqual(connections.segmentsForCell(cell), 1,
    #                    "Too many segments")

    # Test case where both objects are present in bottom up representation, but
    # only one in lateral. In this case the laterally supported object
    # should dominate after second iteration.

    # Test where lateral input is from first column
    pooler.reset()
    pooler.compute(feedforwardInput=(feedforwardInputs[0][0] |
                                     feedforwardInputs[1][1]),
                   lateralInput=lateralInputs[0][0],
                   learn=False)
    self.assertEqual(set(pooler.getActiveCells()),
                     objectRepresentations[0],
           "Incorrect object representations - expecting single object")

    # Test lateral from second column
    pooler.reset()
    pooler.compute(feedforwardInput=(feedforwardInputs[0][0] |
                                     feedforwardInputs[1][1]),
                   lateralInput=lateralInputs[1][0],
                   learn=False)
    self.assertEqual(set(pooler.getActiveCells()),
                     objectRepresentations[0],
           "Incorrect object representations - expecting single object")


  def testWinnersBasedOnLateralActivity(self):
    """Tests internal pooler method _winnersBasedOnLateralActivity()."""

    pooler = self._initializeDefaultPooler()

    # With no lateral support end up with bottom up activity
    overlaps = numpy.zeros(pooler.numberOfColumns(), dtype=realDType)
    overlaps[range(0, 40)] = 10
    active = pooler._winnersBasedOnLateralActivity(
      activeCells=set(range(0, 40)),
      predictiveCells=set(),
      overlaps=overlaps,
      targetActiveCells=40
    )
    self.assertEqual(sum(active), sum(range(0, 40)),
                     "Incorrect active cells with no lateral support")


    # Test case where you have two objects in bottom up representation, but
    # only one in lateral. In this case the laterally supported object
    # should dominate.
    overlaps = numpy.zeros(pooler.numberOfColumns(), dtype=realDType)
    overlaps[range(0, 80)] = 10
    active = pooler._winnersBasedOnLateralActivity(
      activeCells=set(range(0, 80)),
      predictiveCells=set(range(0, 40)),
      overlaps=overlaps,
      targetActiveCells=40
    )
    self.assertEqual(sum(active), sum(range(0, 40)),
                     "Incorrect active cells with bottom up union "
                     "and some lateral support")


    # Test case where BU has support for O1+O2, and lateral support for O2+O3
    # Should end up with O2
    overlaps = numpy.zeros(pooler.numberOfColumns(), dtype=realDType)
    overlaps[range(0, 80)] = 10
    active = pooler._winnersBasedOnLateralActivity(
      activeCells=set(range(0, 80)),
      predictiveCells=set(range(40, 120)),
      overlaps=overlaps,
      targetActiveCells=40
    )
    self.assertEqual(sum(active), sum(range(40, 80)),
                     "Incorrect active cells with bottom up union "
                     "and lateral support that includes other objects")

    # Test case where BU has support for O1, and lateral support for O2
    # Should end up with O1
    overlaps = numpy.zeros(pooler.numberOfColumns(), dtype=realDType)
    overlaps[range(0, 40)] = 10
    active = pooler._winnersBasedOnLateralActivity(
      activeCells=set(range(0, 40)),
      predictiveCells=set(range(40, 80)),
      overlaps=overlaps,
      targetActiveCells=40
    )
    self.assertEqual(sum(active), sum(range(0, 40)),
                     "Incorrect active cells with bottom up union "
                     "and conflicting lateral support")

    # Test case where you have partial lateral support for O1 and a
    # bottom up union that includes O1+O2, but higher overlap scores for O2.
    # In this case you should end up the laterally predicted cells in O1, plus
    # cells corresponding to O2.
    overlaps = numpy.zeros(pooler.numberOfColumns(), dtype=realDType)
    overlaps[range(0, 80)] = range(0, 80)
    active = pooler._winnersBasedOnLateralActivity(
      activeCells=set(range(0, 80)),
      predictiveCells=set(range(0, 10)),
      overlaps=overlaps,
      targetActiveCells=40
    )
    self.assertEqual(sum(active), sum(range(0, 10)) + sum(range(50, 80)),
                     "Incorrect active cells with bottom up activity "
                     "and partial lateral support")


if __name__ == "__main__":
  unittest.main()

