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
    "initialPermanence": 0.51,
    "connectedPermanence": 0.6,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "minThresholdDistal": 10,
    "minThresholdProximal": 10,
    "predictedSegmentDecrement": 0.004,
    "activationThresholdDistal": 10,
    "maxNewProximalSynapseCount": 20,
    "maxNewDistalSynapseCount": 20,
    "numActiveColumnsPerInhArea": 40,
    "synPermProximalInc":  0.1,
    "synPermProximalDec": 0.001,
    "initialProximalPermanence": 0.6
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
    "seed": 42
  },
  "L2Params": {
    "columnCount": 1024,
    "inputWidth": 1024 * 8,
    "learningMode": 1,
    "inferenceMode": 1,
    "initialPermanence": 0.51,
    "connectedPermanence": 0.6,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "minThresholdDistal": 10,
    "minThresholdProximal": 10,
    "predictedSegmentDecrement": 0.004,
    "activationThresholdDistal": 10,
    "maxNewProximalSynapseCount": 20,
    "maxNewDistalSynapseCount": 20,
    "numActiveColumnsPerInhArea": 40,
    "synPermProximalInc":  0.1,
    "synPermProximalDec": 0.001,
    "initialProximalPermanence": 0.6
  }
}

networkConfig3 = {
  "networkType": "MultipleL4L2Columns",
  "numCorticalColumns": 2,
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
    "initialPermanence": 0.51,
    "connectedPermanence": 0.6,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "minThresholdDistal": 10,
    "minThresholdProximal": 10,
    "predictedSegmentDecrement": 0.004,
    "activationThresholdDistal": 10,
    "maxNewProximalSynapseCount": 20,
    "maxNewDistalSynapseCount": 20,
    "numActiveColumnsPerInhArea": 40,
    "synPermProximalInc":  0.1,
    "synPermProximalDec": 0.001,
    "initialProximalPermanence": 0.6
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


  def testMultipleL4L2ColumnsCreate(self):
    """
    In this simplistic test we create a network with 3 L4L2Columns, ensure it
    has the right number of regions and try to run some inputs through it
    without crashing.
    """

    net = createNetwork(networkConfig2)
    self.assertEqual(len(net.regions.keys()), 4*3,
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


  def testCustomParameters(self):
    """
    This test creates a network with custom parameters and tests that the
    network gets correctly constructed.
    """
    customConfig = {
      "networkType": "L4L2Column",
      "externalInputSize": 256,
      "sensorInputSize": 512,
      "L4Params": {
        "columnCount": 512,
        "cellsPerColumn": 16,
        "formInternalConnections": 1,
        "learningMode": 1,
        "inferenceMode": 1,
        "learnOnOneCell": 0,
        "initialPermanence": 0.23,
        "connectedPermanence": 0.75,
        "permanenceIncrement": 0.45,
        "permanenceDecrement": 0.1,
        "minThreshold": 15,
        "predictedSegmentDecrement": 0.21,
        "activationThreshold": 16,
        "maxNewSynapseCount": 24,

      },
      "L2Params": {
        "columnCount": 2048,
        "inputWidth": 512 * 16,
        "learningMode": 1,
        "inferenceMode": 1,
        "initialPermanence": 0.45,
        "connectedPermanence": 0.75,
        "permanenceIncrement": 0.23,
        "permanenceDecrement": 0.2,
        "minThresholdDistal": 12,
        "minThresholdProximal": 12,
        "predictedSegmentDecrement": 0.03,
        "activationThresholdDistal": 8,
        "maxNewProximalSynapseCount": 15,
        "activationThresholdDistal": 15,
        "numActiveColumnsPerInhArea": 35,
        "synPermProximalInc": 0.12,
        "synPermProximalDec": 0.1,
        "initialProximalPermanence": 0.56
      }
    }

    net = createNetwork(customConfig)

    self.assertEqual(
      len(net.regions.keys()), 4,
      "Incorrect number of regions"
    )

    # Get various regions
    externalInput = net.regions["externalInput_0"].getSelf()
    sensorInput = net.regions["sensorInput_0"].getSelf()
    L4Column = net.regions["L4Column_0"].getSelf()
    L2Column = net.regions["L2Column_0"].getSelf()

    # we need to do a first compute for the various elements to be constructed
    sensorInput.addDataToQueue([], 0, 0)
    externalInput.addDataToQueue([], 0, 0)
    net.run(1)

    # check that parameters are correct in L4
    for param, value in customConfig["L4Params"].iteritems():
      self.assertEqual(L4Column.getParameter(param), value)

    # check that parameters are correct in L2
    # some parameters are in the tm members
    for param, value in customConfig["L2Params"].iteritems():
      self.assertEqual(L2Column.getParameter(param), value)

    # check that parameters are correct in L2
    self.assertEqual(externalInput.outputWidth,
                     customConfig["externalInputSize"])
    self.assertEqual(sensorInput.outputWidth,
                     customConfig["sensorInputSize"])


  def testSingleColumnL4L2DataFlow(self):
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

    L4Representation00 = self.getL4PredictedActiveCells(L4Column)
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

    L4Representation02 = self.getL4PredictedActiveCells(L4Column)
    self.assertEqual(len(L4Representation02), 20)

    sensorInput.addDataToQueue(features[1], 0, 0)
    externalInput.addDataToQueue(locations[1], 0, 0)
    net.run(1)

    L4Representation11 = self.getL4PredictedActiveCells(L4Column)
    self.assertEqual(len(L4Representation11), 20)

    # send reset signal
    sensorInput.addResetToQueue(0)
    externalInput.addResetToQueue(0)
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
      self.getL4PredictedActiveCells(L4Column),
      L4Representation11
    )
    self.assertEqual(len(self.getL4BurstingCells(L4Column)), 0)
    sensorInput.addDataToQueue(features[1], 0, 0)
    externalInput.addDataToQueue(locations[2], 0, 0)
    net.run(1)

    # check bursting (representation in L2 should be like in a random SP)
    self.assertEqual(len(self.getL4PredictedActiveCells(L4Column)), 0)
    self.assertEqual(len(self.getL4BurstingCells(L4Column)), 20 * 8)


  def testTwoColumnsL4L2DataFlow(self):
    """
    This test trains a network with a few (feature, location) pairs and checks
    the data flows correctly, and that each intermediate representation is
    correct.

    Indices 0 and 1 in variable names refer to cortical column number.
    """

    # Create a simple network to test the sensor
    net = createNetwork(networkConfig3)

    self.assertEqual(
      len(net.regions.keys()), 4 * 2,
      "Incorrect number of regions"
    )

    # Get various regions
    externalInput0 = net.regions["externalInput_0"].getSelf()
    sensorInput0 = net.regions["sensorInput_0"].getSelf()
    L4Column0 = net.regions["L4Column_0"].getSelf()
    L2Column0 = net.regions["L2Column_0"].getSelf()

    externalInput1 = net.regions["externalInput_1"].getSelf()
    sensorInput1 = net.regions["sensorInput_1"].getSelf()
    L4Column1 = net.regions["L4Column_1"].getSelf()
    L2Column1 = net.regions["L2Column_1"].getSelf()

    # create a feature and location pool for column 0
    features0 = [self.generatePattern(1024, 20) for _ in xrange(2)]
    locations0 = [self.generatePattern(1024, 20) for _ in xrange(3)]

    # create a feature and location pool for column 1
    features1 = [self.generatePattern(1024, 20) for _ in xrange(2)]
    locations1 = [self.generatePattern(1024, 20) for _ in xrange(3)]

    # train with following pairs:
    # (F0, L0) (F1, L1) on object 1
    # (F0, L2) (F1, L1) on object 2

    # Object 1

    # start with an object A input to get L2 representations for object A
    sensorInput0.addDataToQueue(features0[0], 0, 0)
    externalInput0.addDataToQueue(locations0[0], 0, 0)
    sensorInput1.addDataToQueue(features1[0], 0, 0)
    externalInput1.addDataToQueue(locations1[0], 0, 0)
    net.run(1)

    # get L2 representation for object B
    L2RepresentationA0 = self.getCurrentL2Representation(L2Column0)
    L2RepresentationA1 = self.getCurrentL2Representation(L2Column1)
    self.assertEqual(len(L2RepresentationA0), 40)
    self.assertEqual(len(L2RepresentationA0), 40)

    for _ in xrange(3):
      sensorInput0.addDataToQueue(features0[0], 0, 0)
      externalInput0.addDataToQueue(locations0[0], 0, 0)
      sensorInput1.addDataToQueue(features1[0], 0, 0)
      externalInput1.addDataToQueue(locations1[0], 0, 0)
      net.run(1)

      # check L2
      self.assertEqual(
        self.getCurrentL2Representation(L2Column0),
        L2RepresentationA0
      )
      # check L2
      self.assertEqual(
        self.getCurrentL2Representation(L2Column1),
        L2RepresentationA1
      )
      sensorInput0.addDataToQueue(features0[1], 0, 0)
      externalInput0.addDataToQueue(locations0[1], 0, 0)
      sensorInput1.addDataToQueue(features1[1], 0, 0)
      externalInput1.addDataToQueue(locations1[1], 0, 0)
      net.run(1)

      # check L2
      self.assertEqual(
        self.getCurrentL2Representation(L2Column0),
        L2RepresentationA0
      )
      self.assertEqual(
        self.getCurrentL2Representation(L2Column1),
        L2RepresentationA1
      )

    # get L4 representations when they are stable
    sensorInput0.addDataToQueue(features0[0], 0, 0)
    externalInput0.addDataToQueue(locations0[0], 0, 0)
    sensorInput1.addDataToQueue(features1[0], 0, 0)
    externalInput1.addDataToQueue(locations1[0], 0, 0)
    net.run(1)

    L4Representation00_0 = self.getL4PredictedActiveCells(L4Column0)
    L4Representation00_1 = self.getL4PredictedActiveCells(L4Column1)
    self.assertEqual(len(L4Representation00_0), 20)
    self.assertEqual(len(L4Representation00_1), 20)

    # send reset signal
    sensorInput0.addResetToQueue(0)
    externalInput0.addResetToQueue(0)
    sensorInput1.addResetToQueue(0)
    externalInput1.addResetToQueue(0)
    net.run(1)

    # Object B

    # start with input to get L2 representations
    sensorInput0.addDataToQueue(features0[0], 0, 0)
    externalInput0.addDataToQueue(locations0[2], 0, 0)
    sensorInput1.addDataToQueue(features1[0], 0, 0)
    externalInput1.addDataToQueue(locations1[2], 0, 0)
    net.run(1)

    # get L2 representations for object B
    L2RepresentationB0 = self.getCurrentL2Representation(L2Column0)
    L2RepresentationB1 = self.getCurrentL2Representation(L2Column1)
    self.assertEqual(len(L2RepresentationB0), 40)
    self.assertEqual(len(L2RepresentationB1), 40)
    # check that it is very different from object A
    self.assertLessEqual(len(L2RepresentationA0 & L2RepresentationB0), 5)
    self.assertLessEqual(len(L2RepresentationA1 & L2RepresentationB1), 5)

    for _ in xrange(3):
      sensorInput0.addDataToQueue(features0[0], 0, 0)
      externalInput0.addDataToQueue(locations0[2], 0, 0)
      sensorInput1.addDataToQueue(features1[0], 0, 0)
      externalInput1.addDataToQueue(locations1[2], 0, 0)
      net.run(1)

      # check L2
      self.assertEqual(
        self.getCurrentL2Representation(L2Column0),
        L2RepresentationB0
      )
      # check L2
      self.assertEqual(
        self.getCurrentL2Representation(L2Column1),
        L2RepresentationB1
      )

      sensorInput0.addDataToQueue(features0[1], 0, 0)
      externalInput0.addDataToQueue(locations0[1], 0, 0)
      sensorInput1.addDataToQueue(features1[1], 0, 0)
      externalInput1.addDataToQueue(locations1[1], 0, 0)
      net.run(1)

      # check L2
      self.assertEqual(
        self.getCurrentL2Representation(L2Column0),
        L2RepresentationB0
      )
      # check L2
      self.assertEqual(
        self.getCurrentL2Representation(L2Column1),
        L2RepresentationB1
      )

    # get L4 representations when they are stable
    sensorInput0.addDataToQueue(features0[0], 0, 0)
    externalInput0.addDataToQueue(locations0[2], 0, 0)
    sensorInput1.addDataToQueue(features1[0], 0, 0)
    externalInput1.addDataToQueue(locations1[2], 0, 0)
    net.run(1)

    L4Representation02_0 = self.getL4PredictedActiveCells(L4Column0)
    L4Representation02_1 = self.getL4PredictedActiveCells(L4Column1)
    self.assertEqual(len(L4Representation02_0), 20)
    self.assertEqual(len(L4Representation02_1), 20)

    sensorInput0.addDataToQueue(features0[1], 0, 0)
    externalInput0.addDataToQueue(locations0[1], 0, 0)
    sensorInput1.addDataToQueue(features1[1], 0, 0)
    externalInput1.addDataToQueue(locations1[1], 0, 0)
    net.run(1)

    L4Representation11_0 = self.getL4PredictedActiveCells(L4Column0)
    L4Representation11_1 = self.getL4PredictedActiveCells(L4Column1)
    self.assertEqual(len(L4Representation11_0), 20)
    self.assertEqual(len(L4Representation11_1), 20)

    sensorInput0.addResetToQueue(0)
    externalInput0.addResetToQueue(0)
    sensorInput1.addResetToQueue(0)
    externalInput1.addResetToQueue(0)
    net.run(1)

    # check inference with each (feature, location) pair
    L2Column0.setParameter("learningMode", 0, 0)
    L4Column0.setParameter("learningMode", 0, 0)
    L2Column1.setParameter("learningMode", 0, 0)
    L4Column1.setParameter("learningMode", 0, 0)

    # (F0, L0)
    sensorInput0.addDataToQueue(features0[0], 0, 0)
    externalInput0.addDataToQueue(locations0[0], 0, 0)
    sensorInput1.addDataToQueue(features1[0], 0, 0)
    externalInput1.addDataToQueue(locations1[0], 0, 0)
    net.run(1)

    # check L2 representations, L4 representations, no bursting
    self.assertLessEqual(
      len(self.getCurrentL2Representation(L2Column0) - L2RepresentationA0),
      5
    )
    self.assertGreaterEqual(
      len(self.getCurrentL2Representation(L2Column0) & L2RepresentationA0),
      35
    )
    self.assertEqual(
      self.getL4PredictedActiveCells(L4Column0),
      L4Representation00_0
    )
    self.assertEqual(len(self.getL4BurstingCells(L4Column0)), 0)

    # be a little tolerant on this test
    self.assertLessEqual(
      len(self.getCurrentL2Representation(L2Column1) - L2RepresentationA1),
      5
    )
    self.assertGreaterEqual(
      len(self.getCurrentL2Representation(L2Column1) & L2RepresentationA1),
      35
    )
    self.assertEqual(
      self.getL4PredictedActiveCells(L4Column1),
      L4Representation00_1
    )
    self.assertEqual(len(self.getL4BurstingCells(L4Column1)), 0)

    # (F0, L2)
    # It is fed twice, for the ambiguous prediction test, because of the
    # one-off error in distal predictions
    # FIXME when this is changed in ColumnPooler
    sensorInput0.addDataToQueue(features0[0], 0, 0)
    externalInput0.addDataToQueue(locations0[2], 0, 0)
    sensorInput1.addDataToQueue(features1[0], 0, 0)
    externalInput1.addDataToQueue(locations1[2], 0, 0)

    sensorInput0.addDataToQueue(features0[0], 0, 0)
    externalInput0.addDataToQueue(locations0[2], 0, 0)
    sensorInput1.addDataToQueue(features1[0], 0, 0)
    externalInput1.addDataToQueue(locations1[2], 0, 0)
    net.run(2)

    # check L2 representation, L4 representation, no bursting
    self.assertEqual(
      self.getCurrentL2Representation(L2Column0),
      L2RepresentationB0
    )
    self.assertEqual(
      self.getL4PredictedActiveCells(L4Column0),
      L4Representation02_0
    )
    self.assertEqual(len(self.getL4BurstingCells(L4Column0)), 0)

    self.assertEqual(
      self.getCurrentL2Representation(L2Column1),
      L2RepresentationB1
    )
    self.assertEqual(
      self.getL4PredictedActiveCells(L4Column1),
      L4Representation02_1
    )
    self.assertEqual(len(self.getL4BurstingCells(L4Column1)), 0)

    # check predictions for next step...
    self.assertEqual(
      self.getCurrentL2PredictiveCells(L2Column0),
      L2RepresentationB0
    )
    self.assertEqual(
      self.getCurrentL2PredictiveCells(L2Column1),
      L2RepresentationB1
    )

    # ambiguous pattern: (F1, L1)
    sensorInput0.addDataToQueue(features0[1], 0, 0)
    externalInput0.addDataToQueue(locations0[1], 0, 0)
    sensorInput1.addDataToQueue(features1[1], 0, 0)
    externalInput1.addDataToQueue(locations1[1], 0, 0)
    net.run(1)

    # check L2 representation, L4 representation, no bursting
    # as opposed to the previous test, the representation is not ambiguous
    self.assertEqual(
      self.getCurrentL2Representation(L2Column0),
      L2RepresentationB0
    )
    self.assertEqual(
      self.getL4PredictedActiveCells(L4Column0),
      L4Representation11_0
    )
    self.assertEqual(len(self.getL4BurstingCells(L4Column0)), 0)

    self.assertEqual(
      self.getCurrentL2Representation(L2Column1),
      L2RepresentationB1
    )
    self.assertEqual(
      self.getL4PredictedActiveCells(L4Column1),
      L4Representation11_1
    )
    self.assertEqual(len(self.getL4BurstingCells(L4Column1)), 0)

    # unknown signal
    sensorInput0.addDataToQueue(features0[1], 0, 0)
    externalInput0.addDataToQueue(locations0[2], 0, 0)
    sensorInput1.addDataToQueue(features1[1], 0, 0)
    externalInput1.addDataToQueue(locations1[2], 0, 0)
    net.run(1)

    # check bursting (representation in L2 should be like in a random SP)
    self.assertLessEqual(len(self.getL4PredictedActiveCells(L4Column0)), 3)
    self.assertGreaterEqual(len(self.getL4BurstingCells(L4Column0)), 20 * 7)
    self.assertLessEqual(len(self.getL4PredictedActiveCells(L4Column1)), 3)
    self.assertGreaterEqual(len(self.getL4BurstingCells(L4Column1)), 20 * 7)


  def generatePattern(self, max, size):
    """Generates a random feedback pattern."""
    cellsIndices = range(max)
    random.shuffle(cellsIndices)
    return cellsIndices[:size]


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
    """Returns the current active representation in a given L2 column."""
    return set(column._pooler.activeCells)


  def getCurrentL2PredictiveCells(self, column):
    """Returns the current predictive cells in a given L2 column."""
    return set(column._pooler.tm.getPredictiveCells())



if __name__ == "__main__":
  unittest.main()
