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
import copy

from htmresearch.frameworks.layers.laminar_network import createNetwork


networkConfig = {
  "networkType": "L2456Columns",
  "numCorticalColumns": 1,
  "randomSeedBase": 42,

  "sensorParams": {
    "outputWidth": 2048,
  },

  "coarseSensorParams": {
    "outputWidth": 2048,
  },

  "locationParams": {
    "activeBits": 41,
    "outputWidth": 2048,
    "radius": 2,
    "verbosity": 0,
  },

  "L4RegionType": "py.ExtendedTMRegion",

  "L4Params": {
    "columnCount": 2048,
    "cellsPerColumn": 8,
    "learn": True,
    "learnOnOneCell": False,
    "initialPermanence": 0.51,
    "connectedPermanence": 0.6,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "minThreshold": 10,
    "predictedSegmentDecrement": 0.004,
    "activationThreshold": 13,
    "sampleSize": 20,
  },

  "L6Params": {
    "columnCount": 2048,
    "cellsPerColumn": 8,
    "learn": True,
    "learnOnOneCell": False,
    "initialPermanence": 0.51,
    "connectedPermanence": 0.6,
    "permanenceIncrement": 0.1,
    "permanenceDecrement": 0.02,
    "minThreshold": 10,
    "predictedSegmentDecrement": 0.004,
    "activationThreshold": 13,
    "sampleSize": 20,
  },

  "L5Params": {
    "inputWidth": 2048 * 8,
    "cellCount": 2048,
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
    "distalSegmentInhibitionFactor": 1.5,
    "learningMode": True,
  },

  "L2Params": {
    "inputWidth": 2048 * 8,
    "cellCount": 2048,
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
    "distalSegmentInhibitionFactor": 1.5,
    "learningMode": True,
  },
}


class L2456NetworkTest(unittest.TestCase):
  """ Super simple test of laminar network factory"""

  @classmethod
  def setUpClass(cls):
    random.seed(42)


  def _checkPhases(self, net):
    """
    Given a L2456 network, check regions have the correct phases and prefixes
    """
    # Regions whose names begin with the key, should have the assigned phase
    phaseList = {
      "locationInput": 0, "coarseSensorInput": 0, "sensorInput": 0,
      "L6Column": 2, "L5Column": 3, "L4Column": 4, "L2Column": 5
    }
    for region in net.regions.values():
      name = region.name.split("_")[0]
      self.assertTrue(phaseList.has_key(name),
                      "Region name prefix incorrect: " + region.name)
      self.assertEqual(net.getPhases(region.name)[0], phaseList[name],
                       "Incorrect phase for " + region.name)


  def _runNetwork(self, net, numColumns):
    """
    Check we can run the network. Insert unique SDRs into each sensor in each
    column. Run the network for 3 iterations.
    """
    for i in range(numColumns):
      # Add some input vectors to the queue
      suffix = "_" + str(i)
      locationInput = net.regions["locationInput" + suffix].getSelf()
      coarseSensorInput = net.regions["coarseSensorInput" + suffix].getSelf()
      sensorInput = net.regions["sensorInput" + suffix].getSelf()

      # Add 3 input vectors to each column, making sure they are all unique
      for k in range(3):
        locationInput.addDataToQueue(
          [2 + i + k * 10, 42 + i + k * 10, 100 + i + k * 10], 0, 9)
        coarseSensorInput.addDataToQueue(
          [12 + i + k * 10, 52 + i + k * 10, 110 + i + k * 10], 0, 9)
        sensorInput.addDataToQueue(
          [22 + i + k * 10, 62 + i + k * 10, 120 + i + k * 10], 0, 9)

    # Now run the network
    for k in range(3):
      net.run(1)

      # Check L6 and L4 regions are getting the right sensor input
      # We won't verify location output other than to ensure there is input
      for i in range(numColumns):
        # Add some input vectors to the queue
        suffix = "_" + str(i)
        L6Column = net.regions["L6Column" + suffix]
        L4Column = net.regions["L4Column" + suffix]

        self.assertEqual(
          L6Column.getInputData("activeColumns").nonzero()[0].sum(),
          sum([12 + i + k * 10, 52 + i + k * 10, 110 + i + k * 10]),
          "Feedforward input to L6Column is incorrect"
        )

        self.assertGreaterEqual(
          L6Column.getInputData("basalInput").nonzero()[0].sum(),
          40, "External input to L6Column is incorrect"
        )

        self.assertEqual(
          L4Column.getInputData("activeColumns").nonzero()[0].sum(),
          sum([22 + i + k * 10, 62 + i + k * 10, 120 + i + k * 10]),
          "Feedforward input to L4Column is incorrect"
        )


  def testCreatingSingleColumn(self):
    """
    In this simplistic test we just create a network, ensure it has the
    right number of regions and try to run some inputs through it without
    crashing.
    """

    # Create a simple network to test the sensor
    net = createNetwork(networkConfig)

    # Does it have the correct number of regions?
    self.assertEqual(len(net.regions.keys()),7,
                     "Incorrect number of regions")

    # Do regions have the correct phases?
    self._checkPhases(net)

    # Check we can run the network
    self._runNetwork(net, 1)


  def testCreatingMultipleColumns(self):
    """
    We create a network with 5 columns
    """
    config = copy.deepcopy(networkConfig)
    config["numCorticalColumns"] = 6

    # Create a simple network to test the sensor
    net = createNetwork(config)

    self.assertEqual(len(net.regions.keys()),7*config["numCorticalColumns"],
                     "Incorrect number of regions")

    # Do regions have the correct phases?
    self._checkPhases(net)

    # Check we can run the network
    self._runNetwork(net, config["numCorticalColumns"])


  def testIncorrectWidths(self):
    """
    We create a network with sensor and coarse sensor widths that don't match
    column counts. We expect assertion errors in this case
    """
    config = copy.deepcopy(networkConfig)
    config["numCorticalColumns"] = 2
    config["L4Params"]["columnCount"] = 42
    with self.assertRaises(AssertionError):
      createNetwork(config)

    config = copy.deepcopy(networkConfig)
    config["numCorticalColumns"] = 2
    config["L6Params"]["columnCount"] = 42
    with self.assertRaises(AssertionError):
      createNetwork(config)


  @unittest.skip("TODO: depends on NUP-2309")
  def testLinks(self):
    """
    We create a network with 5 columns and check all the links are correct
    """
    config = copy.deepcopy(networkConfig)
    config["numCorticalColumns"] = 5

    # Create a simple network to test the sensor
    net = createNetwork(config)

    self.assertTrue(False)


if __name__ == "__main__":
  unittest.main()
