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
import unittest

from htmresearch.support.register_regions import registerAllResearchRegions
from nupic.engine import Network

class RawSensorTest(unittest.TestCase):

  def testSensor(self):

    # Setup
    registerAllResearchRegions()
    if os.path.exists("temp.csv"):
      os.unlink("temp.csv")

    # Create a simple network to test the sensor
    rawParams = {"outputWidth": 1024}
    net = Network()
    raw = net.addRegion("raw","py.RawSensor", json.dumps(rawParams))
    vfe = net.addRegion("output","VectorFileEffector","")
    net.link("raw", "output", "UniformLink", "")

    # Set an output file before we run anything
    vfe.setParameter("outputFile","temp.csv")

    # Add vectors using two different methods
    raw.executeCommand(["addDataToQueue", "[2, 4, 6]", "0", "42"])
    sensor = raw.getSelf()
    sensor.addDataToQueue([2, 42, 1023], 0, 42)

    # Run the network
    net.run(2)

    net.save("rawNetwork.nta")


if __name__ == "__main__":
  unittest.main()
