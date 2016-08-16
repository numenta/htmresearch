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
    "predictedSegmentDecrement": 0.08,
    "activationThreshold": 13,
    "maxNewSynapseCount": 20,
  },
  "L2Params": {
    "columnCount": 1024,
    "inputWidth": 1024 * 8,
    "learningMode": 1,
    "inferenceMode": 1,
    "minThreshold": 15
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

    In the current implementation, each call to compute is doubled to be able
    to correctly fetch the representations in order to test them laster (as
    the use of the feedforward input is delayed by the region in order to have
    the location predict the feature at the same time step).
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
    # (F1, L1) (F2, L2) on object 1
    # (F1, L3) (F2, L2) on object 2

    # Object 1

    # start with empty input
    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[0], 0, 0)
    net.run(1)

    # get L2 representation for object B
    L2RepresentationA = self.getCurrentL2Representation(L2Column)

    for _ in xrange(3):
      sensorInput.addDataToQueue(features[0], 0, 0)
      externalInput.addDataToQueue(locations[0], 0, 0)
      net.run(1)

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

      sensorInput.addDataToQueue(features[1], 0, 0)
      externalInput.addDataToQueue(locations[1], 0, 0)
      net.run(1)

      # check L2
      self.assertEqual(
        self.getCurrentL2Representation(L2Column),
        L2RepresentationA
      )

    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[0], 0, 0)
    net.run(1)

    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[0], 0, 0)
    net.run(1)

    L4Representation00 = self.getL4PredictedActiveCells(L4Column, -1)

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

      sensorInput.addDataToQueue(features[1], 0, 0)
      externalInput.addDataToQueue(locations[1], 0, 0)
      net.run(1)

      # check L2
      self.assertEqual(
        self.getCurrentL2Representation(L2Column),
        L2RepresentationB
      )

    sensorInput.addDataToQueue(features[1], 0, 0)
    externalInput.addDataToQueue(locations[1], 0, 0)
    net.run(1)

    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[2], 0, 0)

    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[2], 0, 0)
    net.run(2)

    L4Representation11 = self.getL4PredictedActiveCells(L4Column, -2)
    L4Representation02 = self.getL4PredictedActiveCells(L4Column, -1)

    # send reset signal
    sensorInput.addDataToQueue(features[1], 1, 0)
    externalInput.addDataToQueue(locations[1], 0, 0)
    net.run(1)

    # check inference with each (feature, location) pair
    L2Column.setParameter("learningMode", 0, 0)
    L4Column.setParameter("learningMode", 0, 0)

    # start with empty input
    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[0], 0, 0)
    net.run(1)

    # (F1, L1)
    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[0], 0, 0)
    net.run(1)

    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[0], 0, 0)
    net.run(1)

    # check L2 representation, L4 representation, no bursting
    self.assertEqual(
      self.getCurrentL2Representation(L2Column),
      L2RepresentationA
    )
    self.assertEqual(
      self.getL4PredictedActiveCells(L4Column, -1),
      L4Representation00
    )
    self.assertEqual(len(self.getL4BurstingCells(L4Column, -1)), 0)

    # (F1, L3)
    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[2], 0, 0)
    net.run(1)

    # (F1, L3)
    sensorInput.addDataToQueue(features[0], 0, 0)
    externalInput.addDataToQueue(locations[2], 0, 0)
    net.run(1)

    # check L2 representation, L4 representation, no bursting
    self.assertEqual(
      self.getCurrentL2Representation(L2Column),
      L2RepresentationB
    )
    self.assertEqual(
      self.getL4PredictedActiveCells(L4Column, -1),
      L4Representation02
    )
    self.assertEqual(len(self.getL4BurstingCells(L4Column, -1)), 0)

    # (F2, L2)
    sensorInput.addDataToQueue(features[1], 0, 0)
    externalInput.addDataToQueue(locations[1], 0, 0)
    net.run(1)

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
      self.getL4PredictedActiveCells(L4Column, -1),
      L4Representation11
    )
    self.assertEqual(len(self.getL4BurstingCells(L4Column, -1)), 0)

    # (F2, L3)
    sensorInput.addDataToQueue(features[1], 0, 0)
    externalInput.addDataToQueue(locations[2], 0, 0)
    net.run(1)

    sensorInput.addDataToQueue(features[1], 0, 0)
    externalInput.addDataToQueue(locations[2], 0, 0)
    net.run(1)

    # check bursting (representation in L2 should be like in a random SP)
    self.assertEqual(len(self.getL4PredictedActiveCells(L4Column, -1)), 0)
    self.assertEqual(len(self.getL4BurstingCells(L4Column, -1)), 40)


  def generatePattern(self, max, size):
    """Generates a random feedback pattern."""
    return [random.randint(0, max-1) for _ in range(size)]


  def getL4PredictiveCells(self, column, step=None):
    """Returns the predictive cells at given timestep."""
    if step is None:
      return column._tm.mmGetTracePredictiveCells().data
    return column._tm.mmGetTracePredictiveCells().data[step]


  def getL4PredictedActiveCells(self, column, step=None):
    """Returns the predicted active cells at given timestep."""
    if step is None:
      return column._tm.mmGetTracePredictedActiveCells().data
    return column._tm.mmGetTracePredictedActiveCells().data[step]


  def getL4BurstingCells(self, column, step=None):
    """Returns the bursting cells at given timestep."""
    if step is None:
      return column._tm.mmGetTraceUnpredictedActiveColumns().data
    return column._tm.mmGetTraceUnpredictedActiveColumns().data[step]


  def getL4ActiveColumns(self, column, step=None):
    """Returns the active columns at given timestep."""
    if step is None:
      return column._tm.mmGetTraceActiveColumns().data
    return column._tm.mmGetTraceActiveColumns().data[step]


  def getCurrentL2Representation(self, column):
    return set(column._pooler.getActiveCells())


if __name__ == "__main__":
  unittest.main()
