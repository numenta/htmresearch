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
  "networkType": "L4L2Column",
  "externalInputSize": 1024,
  "sensorInputSize": 1024,
  "L4Params": {
    "columnCount": 1024,
    "cellsPerColumn": 8,
    "formInternalConnections": 0,
    "learningMode": 1,
    "inferenceMode": 1,
    "learnOnOneCell": 0,
    "initialPermanence": 0.51,
    "connectedPermanence": 0.6,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "minThreshold": 10,
    "predictedSegmentDecrement": 0.004,
    "activationThreshold": 13,
    "maxNewSynapseCount": 20,
  },
  "L2Params": {
    "columnCount": 1024,
    "inputWidth": 1024 * 8,
    "learningMode": 1,
    "inferenceMode": 1,
    "minThreshold": 10
  }
}

networkConfig2 = {
  "networkType": "MultipleL4L2Columns",
  "numCorticalColumns": 3,
  "externalInputSize": 1024,
  "sensorInputSize": 1024,
  "L4Params": {
    "columnCount": 1024,
    "cellsPerColumn": 8,
  },
  "L2Params": {
    "columnCount": 1024,
    "inputWidth": 1024 * 8,
  }
}


class LaminarNetworkTest(unittest.TestCase):
  """ Super simple test of laminar network factory"""

  @classmethod
  def setUpClass(cls):
    random.seed(42)
    registerAllResearchRegions()


  def testL4L2ColumnCreate(self):
    """
    In this simplistic test we just create a network, ensure it has the
    right number of regions and try to run some inputs through it without
    crashing.
    """

    # Create a simple network to test the sensor
    net = createNetwork(networkConfig1)

    self.assertEqual(len(net.regions.keys()),4,
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

    # Run the network and check outputs are as expected
    net.run(3)

  @unittest.skip("Skipped until lateral connections in L2 are stable")
  def testMultipleL4L2ColumnsCreate(self):
    """
    In this simplistic test we create a network with 3 L4L2Columns, ensure it
    has the right number of regions and try to run some inputs through it
    without crashing.
    """

    net = createNetwork(networkConfig2)
    self.assertEqual(len(net.regions.keys()),4*3,
                     "Incorrect number of regions")

    # Add some input vectors to the queue
    externalInput0 = net.regions["externalInput_0"].getSelf()
    sensorInput0 = net.regions["sensorInput_0"].getSelf()
    externalInput1 = net.regions["externalInput_1"].getSelf()
    sensorInput1 = net.regions["sensorInput_1"].getSelf()
    externalInput2 = net.regions["externalInput_2"].getSelf()
    sensorInput2 = net.regions["sensorInput_2"].getSelf()

    externalInput0.addDataToQueue([2, 42, 1023], 0, 9)
    sensorInput0.addDataToQueue([2, 42, 1023], 0, 0)
    externalInput1.addDataToQueue([2, 42, 1023], 0, 9)
    sensorInput1.addDataToQueue([2, 42, 1023], 0, 0)
    externalInput2.addDataToQueue([2, 42, 1023], 0, 9)
    sensorInput2.addDataToQueue([2, 42, 1023], 0, 0)

    # Run the network and check outputs are as expected
    net.run(1)

    # Spotcheck some of the phases
    self.assertEqual(net.getPhases("externalInput_0"),(0,),
                     "Incorrect phase externalInput_0")
    self.assertEqual(net.getPhases("externalInput_1"),(0,),
                     "Incorrect phase for externalInput_1")
    self.assertEqual(net.getPhases("L4Column_0"),(1,),
                     "Incorrect phase for L4Column_0")
    self.assertEqual(net.getPhases("L4Column_1"),(1,),
                     "Incorrect phase for L4Column_1")


  def testL4L2DataFlow(self):
    """
    This test trains a network with a few (feature, location) pairs and checks
    the data flows correctly, and that each intermediate representation is
    correct.
    """

    # Create a simple network to test the sensor
    net = createNetwork(networkConfig1)

    self.assertEqual(
      len(net.regions.keys()), 4,
      "Incorrect number of regions"
    )

    # Get various regions
    externalInput = net.regions["externalInput_0"].getSelf()
    sensorInput = net.regions["sensorInput_0"].getSelf()
    L4Column = net.regions["L4Column_0"].getSelf()
    L2Column = net.regions["L2Column_0"].getSelf()

    # create a feature and location pool
    features = [self.generatePattern(1024, 40) for _ in xrange(2)]
    locations = [self.generatePattern(1024, 40) for _ in xrange(3)]

    # train with following pairs:
    # (F0, L0) (F1, L1) on object 1
    # (F0, L2) (F1, L1) on object 2

    # Object 1

    # start with an object 1 input to get L2 representation for object 1
    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[0], 0, 0)
    net.run(1)

    # get L2 representation for object B
    L2RepresentationA = self.getCurrentL2Representation(L2Column)

    for _ in xrange(3):
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

    L4Representation00 = self.getL4PredictedActiveCells(L4Column)

    # send reset signal
    sensorInput.addDataToQueue(features[1], 1, 0)
    externalInput.addDataToQueue(locations[1], 0, 0)
    net.run(1)

    # Object B

    # start with empty input
    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[2], 0, 0)
    net.run(1)

    # get L2 representation for object B
    L2RepresentationB = self.getCurrentL2Representation(L2Column)
    # check that it is very different from object A
    self.assertLessEqual(len(L2RepresentationA & L2RepresentationB), 5)

    for _ in xrange(3):
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

    L4Representation02 = self.getL4PredictedActiveCells(L4Column)


    sensorInput.addDataToQueue(features[1], 0, 0)
    externalInput.addDataToQueue(locations[1], 0, 0)
    net.run(1)

    L4Representation11 = self.getL4PredictedActiveCells(L4Column)

    # send reset signal
    sensorInput.addDataToQueue(features[1], 1, 0)
    externalInput.addDataToQueue(locations[1], 0, 0)
    net.run(1)

    # check inference with each (feature, location) pair
    L2Column.setParameter("learningMode", 0, 0)
    L4Column.setParameter("learningMode", 0, 0)

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
      self.getL4PredictedActiveCells(L4Column),
      L4Representation00
    )
    self.assertEqual(len(self.getL4BurstingCells(L4Column)), 0)

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
      self.getL4PredictedActiveCells(L4Column),
      L4Representation02
    )
    self.assertEqual(len(self.getL4BurstingCells(L4Column)), 0)

    # (F2, L2)
    sensorInput.addDataToQueue(features[1], 0, 0)
    externalInput.addDataToQueue(locations[1], 0, 0)
    net.run(1)

    # check L2 representation, L4 representation, no bursting
    self.assertEqual(
      self.getCurrentL2Representation(L2Column),
      L2RepresentationA | L2RepresentationB
    )
    self.assertEqual(
      self.getL4PredictedActiveCells(L4Column),
      L4Representation11
    )
    self.assertEqual(len(self.getL4BurstingCells(L4Column)), 0)
    sensorInput.addDataToQueue(features[1], 0, 0)
    externalInput.addDataToQueue(locations[2], 0, 0)
    net.run(1)

    # check bursting (representation in L2 should be like in a random SP)
    self.assertEqual(len(self.getL4PredictedActiveCells(L4Column)), 0)
    self.assertEqual(len(self.getL4BurstingCells(L4Column)), 40 * 8)


  def generatePattern(self, max, size):
    """Generates a random feedback pattern."""
    return [random.randint(0, max-1) for _ in range(size)]


  def getL4PredictiveCells(self, column):
    """Returns the predictive cells in L4."""
    return set(column._tm.getPredictiveCells())


  def getL4PredictedActiveCells(self, column):
    """Returns the predicted active cells in L4."""
    activeCells = set(column._tm.getActiveCells())
    predictiveCells = set(column._tm.getPredictiveCells())
    return activeCells & predictiveCells


  def getL4BurstingCells(self, column):
    """Returns the bursting cells in L4."""
    activeCells = set(column._tm.getActiveCells())
    predictiveCells = set(column._tm.getPredictiveCells())
    return activeCells - predictiveCells


  def getCurrentL2Representation(self, column):
    return set(column._pooler.getActiveCells())



if __name__ == "__main__":
  unittest.main()
