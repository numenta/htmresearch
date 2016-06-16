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

from htmresearch.support.register_regions import registerAllResearchRegions
from htmresearch.frameworks.layers.laminar_network import createNetwork


networkConfig1 = {
  "networkType": "L4L2Column",
  "externalInputSize": 1024,
  "sensorInputSize": 1024,
  "L4Params": {
    "columnCount": 1024,
    "cellsPerColumn": 8,
    "temporalImp": "tmCPP",
  },
  "L2Params": {
    "columnCount": 1024,
  }
}

networkConfig2 = {
  "networkType": "MultipleL4L2Columns",
  "numColumns": 3,
  "externalInputSize": 1024,
  "sensorInputSize": 1024,
  "L4Params": {
    "columnCount": 1024,
    "cellsPerColumn": 8,
    "temporalImp": "tmCPP",
  },
  "L2Params": {
    "columnCount": 1024,
  }
}


class L2ColumnTest(unittest.TestCase):
  """ Super simple test of RawSensor """

  @classmethod
  def setUpClass(cls):
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
    externalInput = net.regions["externalInput"].getSelf()
    sensorInput = net.regions["sensorInput"].getSelf()

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
    In this simplistic test we create a network with 3 L4L2Columns.
    """

    net = createNetwork(networkConfig2)
    self.assertEqual(len(net.regions.keys()),4*3,
                     "Incorrect number of regions")


if __name__ == "__main__":
  unittest.main()

