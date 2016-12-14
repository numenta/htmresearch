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

import json
import numpy
import os
import shutil
import tempfile
import unittest

from nupic.encoders.coordinate import CoordinateEncoder
from nupic.engine import Network
from htmresearch.support.register_regions import registerAllResearchRegions


class CoordinateSensorRegionTest(unittest.TestCase):
  """ CoordinateSensorRegion simple test based on RawSensor test """

  @classmethod
  def setUpClass(cls):
    registerAllResearchRegions()
    cls.tmpDir = tempfile.mkdtemp()
    cls.encoder = CoordinateEncoder(n=1029, w=21, verbosity=0)

  @classmethod
  def tearDownClass(cls):
    if os.path.exists(cls.tmpDir):
      shutil.rmtree(cls.tmpDir)

  def testSensor(self):
    # Create a simple network to test the sensor
    params = {
      "activeBits": self.encoder.w,
      "outputWidth": self.encoder.n,
      "radius": 2,
      "verbosity": self.encoder.verbosity,
    }
    net = Network()
    region = net.addRegion("coordinate", "py.CoordinateSensorRegion",
                           json.dumps(params))
    vfe = net.addRegion("output", "VectorFileEffector", "")
    net.link("coordinate", "output", "UniformLink", "")

    self.assertEqual(region.getParameter("outputWidth"),
                     self.encoder.n, "Incorrect outputWidth parameter")

    # Add vectors to the queue using two different add methods. Later we
    # will check to ensure these are actually output properly.
    region.executeCommand(["addDataToQueue", "[2, 4, 6]", "0", "42"])
    regionPy = region.getSelf()
    regionPy.addDataToQueue([2, 42, 1023], 1, 43)
    regionPy.addDataToQueue([18, 19, 20], 0, 44)

    # Set an output file before we run anything
    vfe.setParameter("outputFile", os.path.join(self.tmpDir, "temp.csv"))

    # Run the network and check outputs are as expected
    net.run(1)
    expected = self.encoder.encode((numpy.array([2, 4, 6]), params["radius"]))
    actual = region.getOutputData("dataOut")
    self.assertEqual(actual.sum(), expected.sum(), "Value of dataOut incorrect")
    self.assertEqual(region.getOutputData("resetOut"), 0,
                     "Value of resetOut incorrect")
    self.assertEqual(region.getOutputData("sequenceIdOut"), 42,
                     "Value of sequenceIdOut incorrect")

    net.run(1)
    expected = self.encoder.encode((numpy.array([2, 42, 1023]),
                                    params["radius"]))
    actual = region.getOutputData("dataOut")
    self.assertEqual(actual.sum(), expected.sum(), "Value of dataOut incorrect")
    self.assertEqual(region.getOutputData("resetOut"), 1,
                     "Value of resetOut incorrect")
    self.assertEqual(region.getOutputData("sequenceIdOut"), 43,
                     "Value of sequenceIdOut incorrect")

    # Make sure we can save and load the network after running
    net.save(os.path.join(self.tmpDir, "coordinateNetwork.nta"))
    net2 = Network(os.path.join(self.tmpDir, "coordinateNetwork.nta"))
    region2 = net2.regions.get("coordinate")
    vfe2 = net2.regions.get("output")

    # Ensure parameters are preserved
    self.assertEqual(region2.getParameter("outputWidth"), self.encoder.n,
                     "Incorrect outputWidth parameter")

    # Ensure the queue is preserved through save/load
    vfe2.setParameter("outputFile", os.path.join(self.tmpDir, "temp.csv"))
    net2.run(1)
    expected = self.encoder.encode((numpy.array([18, 19, 20]),
                                    params["radius"]))
    actual = region2.getOutputData("dataOut")
    self.assertEqual(actual.sum(), expected.sum(), "Value of dataOut incorrect")
    self.assertEqual(region2.getOutputData("resetOut"), 0,
                     "Value of resetOut incorrect")

    self.assertEqual(region2.getOutputData("sequenceIdOut"), 44,
                     "Value of sequenceIdOut incorrect")

  if __name__ == "__main__":
    unittest.main()
