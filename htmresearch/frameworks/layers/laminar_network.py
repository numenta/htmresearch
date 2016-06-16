#!/usr/bin/env python
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
"""
The methods here contain factories to create networks of multiple layers
and for experimenting with different laminar structures.

The first network type supported, "L4L2Column", is a single cortical column
containing and L4 and an L2 layer. L4 gets two inputs and feeds into L2. The L2
column feeds back to L4.

             L2Column  <------|
               ^  |           |
               |  |           |
               |  v           |
        --->  L4Column <------|
        |          ^          |
        |          |        reset
        |          |          |
externalInput  sensorInput -->|

Regions will be named as shown above. The reset signal from sensorInput is
sent to the other regions.

How do you like my ascii art?
"""
import json

from nupic.engine import Network
from htmresearch.support.register_regions import registerAllResearchRegions


def createL4L2Column(networkConfig):
  """
  Create a network consisting of a single column containing one L4 and one L2.

  networkConfig must be of the following format:

    {
      "networkType": "L4L2Column",
      "externalInputSize": 1024,
      "sensorInputSize": 1024,
      "L4Params": {
        <constructor parameters for GeneralTemporalMemoryRegion
      },
      "L2Params": {
        <constructor parameters for L2Column>
      }
    }
  """
  assert networkConfig["networkType"] == "L4L2Column"

  network = Network()

  # Create the two sensors
  extInput = network.addRegion(
    "externalInput", "py.RawSensor",
    json.dumps({"outputWidth": networkConfig["externalInputSize"]}))
  sensorInput = network.addRegion(
    "sensorInput", "py.RawSensor",
    json.dumps({"outputWidth": networkConfig["sensorInputSize"]}))

  # We use TMRegion now as a placeholder until we have a
  # GeneralTemporalMemoryRegion
  L4Column = network.addRegion("L4Column", "py.TMRegion",
                               json.dumps(networkConfig["L4Params"]))

  L2Column = network.addRegion("L2Column", "py.L2Column",
                               json.dumps(networkConfig["L2Params"]))


  # Link sensors to L4
  network.link("externalInput", "L4Column", "UniformLink", "",
               srcOutput="dataOut", destInput="externalInput")
  network.link("sensorInput", "L4Column", "UniformLink", "",
               srcOutput="dataOut", destInput="bottomUpIn")

  # Link L4 to L2, and L2's feedback to L4
  network.link("L4Column", "L2Column", "UniformLink", "")
  network.link("L2Column", "L4Column", "UniformLink", "",
               srcOutput="feedForwardOutput", destInput="topDownIn")

  # Link reset output to L4 and L2
  network.link("sensorInput", "L4Column", "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")
  network.link("sensorInput", "L2Column", "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")

  return network



def createNetwork(networkConfig):
  """
  Create and initialize the specified network instance.

  @param networkConfig: (dict) the configuration of this network.
  @return network: (Network) The actual network
  """

  registerAllResearchRegions()

  if networkConfig["networkType"] == "L4L2Column":
    return createL4L2Column(networkConfig)

