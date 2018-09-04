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
import os
import shutil
import tempfile
import unittest

from nupic.engine import Network
from htmresearch.support.register_regions import registerAllResearchRegions



class RawSensorTest(unittest.TestCase):
  """ Super simple test of RawSensor """

  @classmethod
  def setUpClass(cls):
    registerAllResearchRegions()
    cls.tmpDir = tempfile.mkdtemp()


  @classmethod
  def tearDownClass(cls):
    if os.path.exists(cls.tmpDir):
      shutil.rmtree(cls.tmpDir)


  def testSensor(self):

    # Create a simple network to test the sensor
    rawParams = {"outputWidth": 1029}
    net = Network()
    rawSensor = net.addRegion("raw","py.RawSensor", json.dumps(rawParams))
    vfe = net.addRegion("output","VectorFileEffector","")
    net.link("raw", "output", "UniformLink", "", "dataOut", "sparseDataIn")

    self.assertEqual(rawSensor.getParameter("outputWidth"),1029,
                     "Incorrect outputWidth parameter")

    # Add vectors to the queue using two different add methods. Later we
    # will check to ensure these are actually output properly.
    rawSensor.executeCommand(["addDataToQueue", "[2, 4, 6]", "0", "42"])
    rawSensorPy = rawSensor.getSelf()
    rawSensorPy.addDataToQueue([2, 42, 1023], 1, 43)
    rawSensorPy.addDataToQueue([18, 19, 20], 0, 44)

    # Set an output file before we run anything
    vfe.setParameter("outputFile",os.path.join(self.tmpDir,"temp.csv"))

    # Run the network and check outputs are as expected
    net.run(1)
    self.assertEqual(sum(rawSensor.getOutputData("dataOut")),
                     sum([2, 4, 6]), "Value of dataOut incorrect")
    self.assertEqual(rawSensor.getOutputData("resetOut").sum(),0,
                     "Value of resetOut incorrect")
    self.assertEqual( rawSensor.getOutputData("sequenceIdOut").sum(),42,
                      "Value of sequenceIdOut incorrect")

    net.run(1)
    self.assertEqual(sum(rawSensor.getOutputData("dataOut")),
                     sum([2, 42, 1023]), "Value of dataOut incorrect")
    self.assertEqual(rawSensor.getOutputData("resetOut").sum(),1,
                     "Value of resetOut incorrect")
    self.assertEqual( rawSensor.getOutputData("sequenceIdOut").sum(),43,
                      "Value of sequenceIdOut incorrect")

    # Make sure we can save and load the network after running
    net.save(os.path.join(self.tmpDir,"rawNetwork.nta"))
    net2 = Network(os.path.join(self.tmpDir,"rawNetwork.nta"))
    rawSensor2 = net2.regions.get("raw")
    vfe2 = net2.regions.get("output")

    # Ensure parameters are preserved
    self.assertEqual(rawSensor2.getParameter("outputWidth"),1029,
                     "Incorrect outputWidth parameter")

    # Ensure the queue is preserved through save/load
    vfe2.setParameter("outputFile",os.path.join(self.tmpDir,"temp.csv"))
    net2.run(1)
    self.assertEqual(sum(rawSensor2.getOutputData("dataOut")),
                     sum([18, 19, 20]), "Value of dataOut incorrect")
    self.assertEqual(rawSensor2.getOutputData("resetOut").sum(),0,
                     "Value of resetOut incorrect")
    self.assertEqual( rawSensor2.getOutputData("sequenceIdOut").sum(),44,
                      "Value of sequenceIdOut incorrect")


if __name__ == "__main__":
  unittest.main()

