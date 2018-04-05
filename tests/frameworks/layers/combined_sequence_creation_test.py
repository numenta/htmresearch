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
import random

from htmresearch.support.register_regions import registerAllResearchRegions
from htmresearch.frameworks.layers.laminar_network import createNetwork


networkConfig1 = {
  "networkType": "L4L2TMColumn",
  "externalInputSize": 1024,
  "sensorInputSize": 1024,
  "enableFeedback": False,
  "L4Params": {
    "columnCount": 1024,
    "cellsPerColumn": 8,
    "learn": True,
    "learnOnOneCell": False,
    "initialPermanence": 0.51,
    "connectedPermanence": 0.6,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "minThreshold": 10,
    "basalPredictedSegmentDecrement": 0.004,
    "activationThreshold": 13,
    "sampleSize": 20,
  },
  "L2Params": {
    "inputWidth": 1024 * 8,
    "cellCount": 4096,
    "sdrSize": 40,
    "synPermProximalInc": 0.1,
    "synPermProximalDec": 0.001,
    "initialProximalPermanence": 0.6,
    "minThresholdProximal": 10,
    "sampleSizeProximal": 20,
    "connectedPermanenceProximal": 0.5,
    "synPermDistalInc": 0.1,
    "synPermDistalDec": 0.001,
    "initialDistalPermanence": 0.41,
    "activationThresholdDistal": 13,
    "sampleSizeDistal": 20,
    "connectedPermanenceDistal": 0.5,
    "learningMode": True,
  },
  "TMParams": {
    "columnCount": 1024,
    "cellsPerColumn": 8,
    "learn": True,
    "learnOnOneCell": False,
    "initialPermanence": 0.51,
    "connectedPermanence": 0.6,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "minThreshold": 10,
    "basalPredictedSegmentDecrement": 0.004,
    "activationThreshold": 13,
    "sampleSize": 20,
  },
}


class CombinedSequenceNetworkTest(unittest.TestCase):
  """Simple test of combined sequence network creation"""

  @classmethod
  def setUpClass(cls):
    random.seed(42)
    registerAllResearchRegions()


  def testCreate(self):
    """
    In this simplistic test we just create a network, ensure it has the
    right number of regions and try to run some inputs through it without
    crashing.
    """

    # Create a simple network to test the sensor
    net = createNetwork(networkConfig1)

    self.assertEqual(len(net.regions.keys()),5,
                     "Incorrect number of regions")

    # Add some input vectors to the queue
    externalInput = net.regions["externalInput_0"].getSelf()
    sensorInput = net.regions["sensorInput_0"].getSelf()

    # Add 3 input vectors
    externalInput.addDataToQueue([2, 42, 1023], 0, 9)
    sensorInput.addDataToQueue([2, 42, 1023], 0, 0)

    externalInput.addDataToQueue([1, 42, 1022], 0, 0)
    sensorInput.addDataToQueue([1, 42, 1022], 0, 0)

    externalInput.addDataToQueue([3, 42, 1021], 0, 0)
    sensorInput.addDataToQueue([3, 42, 1021], 0, 0)

    # Run the network and ensure nothing crashes
    net.run(3)


  def testLinks(self):
    """
    In this simplistic test we create a network and ensure that it has the
    correct links between regions.
    """

    # Create a simple network to check its architecture
    net = createNetwork(networkConfig1)

    # These are exactly all the links we expect
    desired_links = {
      "sensorInput_0.dataOut-->L4Column_0.activeColumns",
      "sensorInput_0.dataOut-->TMColumn_0.activeColumns",
      "externalInput_0.dataOut-->L4Column_0.basalInput",
      "L4Column_0.predictedActiveCells-->L2Column_0.feedforwardGrowthCandidates",
      "L4Column_0.activeCells-->L2Column_0.feedforwardInput",
      "sensorInput_0.resetOut-->L2Column_0.resetIn",
      "sensorInput_0.resetOut-->L4Column_0.resetIn",
      "sensorInput_0.resetOut-->TMColumn_0.resetIn",
      "externalInput_0.dataOut-->L4Column_0.basalGrowthCandidates",
    }

    links = net.getLinks()

    # This gets textual representations of the links.
    links = set([link.second.getMoniker() for link in links])

    # Build a descriptive error message to pass to the user
    error_message = "Error: Links incorrectly formed in simple network: \n"
    for link in desired_links:
      if not link in links:
        error_message += "Failed to find link: {}\n".format(link)

    for link in links:
      if not link in desired_links:
        error_message += "Found unexpected link: {}\n".format(link)

    self.assertSetEqual(desired_links, links, error_message)


  def testDataFlowL2L4(self):
    """
    This test trains a network with a few (feature, location) pairs and checks
    the data flows correctly, and that each intermediate representation is
    correct.
    """

    # Create a simple network to test the sensor
    net = createNetwork(networkConfig1)

    # Get various regions
    externalInput = net.regions["externalInput_0"].getSelf()
    sensorInput = net.regions["sensorInput_0"].getSelf()
    L4Column = net.regions["L4Column_0"].getSelf()
    L2Column = net.regions["L2Column_0"].getSelf()

    # create a feature and location pool
    features = [self.generatePattern(1024, 20) for _ in xrange(2)]
    locations = [self.generatePattern(1024, 20) for _ in xrange(3)]

    # train with following pairs:
    # (F0, L0) (F1, L1) on object A
    # (F0, L2) (F1, L1) on object B

    # Object A

    # start with an object 1 input to get L2 representation for object 1
    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[0], 0, 0)
    net.run(1)

    # get L2 representation for object A
    L2RepresentationA = self.getCurrentL2Representation(L2Column)
    self.assertEqual(len(L2RepresentationA), 40)

    for _ in xrange(4):
      sensorInput.addDataToQueue(features[0], 0, 0)
      externalInput.addDataToQueue(locations[0], 0, 0)
      net.run(1)

      # check L2
      self.assertEqual(
        self.getCurrentL2Representation(L2Column),
        L2RepresentationA
      )
      sensorInput.addDataToQueue(features[1], 0, 0)
      externalInput.addDataToQueue(locations[1], 0, 0)
      net.run(1)

      # check L2
      self.assertEqual(
        self.getCurrentL2Representation(L2Column),
        L2RepresentationA
      )

    # get L4 representations when they are stable
    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[0], 0, 0)
    net.run(1)

    L4Representation00 = self.getPredictedActiveCells(L4Column)
    self.assertEqual(len(L4Representation00), 20)

    # send reset signal
    sensorInput.addResetToQueue(0)
    externalInput.addResetToQueue(0)
    net.run(1)

    # Object B

    # start with empty input
    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[2], 0, 0)
    net.run(1)

    # get L2 representation for object B
    L2RepresentationB = self.getCurrentL2Representation(L2Column)
    self.assertEqual(len(L2RepresentationB), 40)
    # check that it is very different from object A
    self.assertLessEqual(len(L2RepresentationA & L2RepresentationB), 5)

    for _ in xrange(4):
      sensorInput.addDataToQueue(features[0], 0, 0)
      externalInput.addDataToQueue(locations[2], 0, 0)
      net.run(1)

      # check L2
      self.assertEqual(
        self.getCurrentL2Representation(L2Column),
        L2RepresentationB
      )

      sensorInput.addDataToQueue(features[1], 0, 0)
      externalInput.addDataToQueue(locations[1], 0, 0)
      net.run(1)

      # check L2
      self.assertEqual(
        self.getCurrentL2Representation(L2Column),
        L2RepresentationB
      )

    # get L4 representations when they are stable
    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[2], 0, 0)
    net.run(1)

    L4Representation02 = self.getPredictedActiveCells(L4Column)
    self.assertEqual(len(L4Representation02), 20)

    sensorInput.addDataToQueue(features[1], 0, 0)
    externalInput.addDataToQueue(locations[1], 0, 0)
    net.run(1)

    L4Representation11 = self.getPredictedActiveCells(L4Column)
    self.assertEqual(len(L4Representation11), 20)

    # send reset signal
    sensorInput.addResetToQueue(0)
    externalInput.addResetToQueue(0)
    net.run(1)

    # check inference with each (feature, location) pair
    L2Column.setParameter("learningMode", 0, False)
    L4Column.setParameter("learn", 0, False)

    # (F0, L0)
    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[0], 0, 0)
    net.run(1)

    # check L2 representation, L4 representation, no bursting
    self.assertEqual(
      self.getCurrentL2Representation(L2Column),
      L2RepresentationA
    )
    self.assertEqual(
      self.getPredictedActiveCells(L4Column),
      L4Representation00
    )
    self.assertEqual(len(self.getBurstingCells(L4Column)), 0)

    # send reset signal
    sensorInput.addResetToQueue(0)
    externalInput.addResetToQueue(0)
    net.run(1)

    # (F0, L2)
    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[2], 0, 0)
    net.run(1)

    # check L2 representation, L4 representation, no bursting
    self.assertEqual(
      self.getCurrentL2Representation(L2Column),
      L2RepresentationB
    )
    self.assertEqual(
      self.getPredictedActiveCells(L4Column),
      L4Representation02
    )
    self.assertEqual(len(self.getBurstingCells(L4Column)), 0)

    # send reset signal
    sensorInput.addResetToQueue(0)
    externalInput.addResetToQueue(0)
    net.run(1)

    # (F1, L1)
    sensorInput.addDataToQueue(features[1], 0, 0)
    externalInput.addDataToQueue(locations[1], 0, 0)
    net.run(1)

    # check L2 representation, L4 representation, no bursting
    self.assertEqual(
      self.getCurrentL2Representation(L2Column),
      L2RepresentationA | L2RepresentationB
    )
    self.assertEqual(
      self.getPredictedActiveCells(L4Column),
      L4Representation11
    )
    self.assertEqual(len(self.getBurstingCells(L4Column)), 0)
    sensorInput.addDataToQueue(features[1], 0, 0)
    externalInput.addDataToQueue(locations[2], 0, 0)
    net.run(1)

    # check bursting (representation in L2 should be like in a random SP)
    self.assertEqual(len(self.getPredictedActiveCells(L4Column)), 0)
    self.assertEqual(len(self.getBurstingCells(L4Column)), 20 * 8)


  def testDataFlowTM(self):
    """
    This test trains a network with two high order sequences and checks
    the data flows correctly, and that the TM learns them correctly.
    """

    # Create a simple network to test the sensor
    net = createNetwork(networkConfig1)

    # Get various regions
    externalInput = net.regions["externalInput_0"].getSelf()
    sensorInput = net.regions["sensorInput_0"].getSelf()
    L4Column = net.regions["L4Column_0"].getSelf()
    L2Column = net.regions["L2Column_0"].getSelf()
    TMColumn = net.regions["TMColumn_0"].getSelf()

    # create a feature and location pool
    features = [self.generatePattern(1024, 20) for _ in xrange(5)]

    # train with following sequences:
    # 1 : F0 F1 F2
    # 2 : F3 F1 F4

    # Sequence A, three repeats with a reset in between.  We add nothing for
    # location signal
    for _ in range(3):
      sensorInput.addDataToQueue(features[0], 0, 0)
      sensorInput.addDataToQueue(features[1], 0, 0)
      sensorInput.addDataToQueue(features[2], 0, 0)

      externalInput.addDataToQueue([], 0, 0)
      externalInput.addDataToQueue([], 0, 0)
      externalInput.addDataToQueue([], 0, 0)

      sensorInput.addResetToQueue(0)
      externalInput.addResetToQueue(0)

    net.run(4*3) # Includes reset

    # Sequence B, three repeats with a reset in between.
    for _ in range(3):
      sensorInput.addDataToQueue(features[3], 0, 0)
      sensorInput.addDataToQueue(features[1], 0, 0)
      sensorInput.addDataToQueue(features[4], 0, 0)

      externalInput.addDataToQueue([], 0, 0)
      externalInput.addDataToQueue([], 0, 0)
      externalInput.addDataToQueue([], 0, 0)

      sensorInput.addResetToQueue(0)
      externalInput.addResetToQueue(0)

    net.run(4*3) # Includes reset


    # check inference
    L2Column.setParameter("learningMode", 0, False)
    L4Column.setParameter("learn", 0, False)
    TMColumn.setParameter("learn", 0, False)

    # Sequence A with a reset in between.
    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue([], 0, 0)
    net.run(1)
    self.assertEqual(len(self.getPredictedActiveCells(TMColumn)), 0)
    self.assertEqual(len(self.getActiveCells(TMColumn)),
                     TMColumn.cellsPerColumn*20)

    sensorInput.addDataToQueue(features[1], 0, 0)
    externalInput.addDataToQueue([], 0, 0)
    net.run(1)
    self.assertEqual(len(self.getPredictedActiveCells(TMColumn)), 20)
    self.assertEqual(len(self.getActiveCells(TMColumn)), 20)
    predictedActiveCellsS1 = self.getPredictedActiveCells(TMColumn)

    sensorInput.addDataToQueue(features[2], 0, 0)
    externalInput.addDataToQueue([], 0, 0)
    net.run(1)
    self.assertEqual(len(self.getPredictedActiveCells(TMColumn)), 20)
    self.assertEqual(len(self.getActiveCells(TMColumn)), 20)

    sensorInput.addResetToQueue(0)
    externalInput.addResetToQueue(0)
    net.run(1)

    # Sequence B with a reset
    sensorInput.addDataToQueue(features[3], 0, 0)
    externalInput.addDataToQueue([], 0, 0)
    net.run(1)
    self.assertEqual(len(self.getPredictedActiveCells(TMColumn)), 0)
    self.assertEqual(len(self.getActiveCells(TMColumn)),
                     TMColumn.cellsPerColumn*20)

    sensorInput.addDataToQueue(features[1], 0, 0)
    externalInput.addDataToQueue([], 0, 0)
    net.run(1)
    self.assertEqual(len(self.getPredictedActiveCells(TMColumn)), 20)
    self.assertEqual(len(self.getActiveCells(TMColumn)), 20)
    predictedActiveCellsS2 = self.getPredictedActiveCells(TMColumn)

    sensorInput.addDataToQueue(features[4], 0, 0)
    externalInput.addDataToQueue([], 0, 0)
    net.run(1)
    self.assertEqual(len(self.getPredictedActiveCells(TMColumn)), 20)
    self.assertEqual(len(self.getActiveCells(TMColumn)), 20)

    # Ensure representation for ambiguous element is different
    self.assertFalse(predictedActiveCellsS1 == predictedActiveCellsS2)


  def generatePattern(self, max, size):
    """Generates a random feedback pattern."""
    cellsIndices = range(max)
    random.shuffle(cellsIndices)
    return cellsIndices[:size]


  def getPredictedCells(self, column):
    """
    Returns the cells in L4 or TM that were predicted at the beginning of the
    last call to 'compute'.
    """
    return set(column._tm.getPredictedCells())


  def getActiveCells(self, column):
    """
    Returns the active cells in L4 or TM.
    """
    return set(column._tm.getActiveCells())


  def getPredictedActiveCells(self, column):
    """Returns the predicted active cells in L4 or TM."""
    activeCells = set(column._tm.getActiveCells())
    predictedCells = set(column._tm.getPredictedCells())
    return activeCells & predictedCells


  def getBurstingCells(self, column):
    """Returns the bursting cells in L4 or TM."""
    activeCells = set(column._tm.getActiveCells())
    predictedCells = set(column._tm.getPredictedCells())
    return activeCells - predictedCells


  def getWinnerCells(self, column):
    """Returns the winner cells in L4 or TM."""
    return set(column._tm.getWinnerCells())


  def getCurrentL2Representation(self, column):
    """Returns the current active representation in a given L2 column."""
    return set(column._pooler.activeCells)



if __name__ == "__main__":
  unittest.main()
