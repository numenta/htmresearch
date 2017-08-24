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
Functions to create a network consisting of multiple L2456 columns.

NETWORK STRUCTURE
=================

Each column contains one L2,L4,L5, and L6 column, and looks like this:

                      +--------+
                      |L2Column|
                      +--+---+-+
                         |   ^
                         |   |
                         |   |
                      +--v---+-+
               +----> |L4Column|
               |      +-------++
               |              ^
               |              +-----------+
               |                          |
               |      +--------+          |
               |      |L5Column|          |
               |      +--+----++          |
               |         |    ^           |
               |         |    |           |
               |         |    |           |
               |      +--v----++          |
               +----- |L6Column|          |
            +-------> +----+---+          |
            |              ^              |
            |              |              |
            |              +              +
    LocationInput     CoarseSensor      Sensor


In addition:
All the L2 columns are fully connected to each other through their lateral
inputs.  Same is true for L5 columns. All regions get reset signal from Sensor.


REGION NAMES AND TYPES
======================

L2Column and L5Column are ColumnPooler's. L4Column and L6Column are Extended
Temporal Memories. LocationInput is a CoordinateSensor. CoarseSensor and Sensor
are raw sensors that are supposed to output the same information at different
levels of coarseness.

Region names are locationInput, sensorInput, coarseSensorInput, L2Column,
L4Column, L5Column, and L6Column. Region names have a column number appended as
in locationInput_0, locationInput_1, etc.


NETWORK CONFIGURATION
=====================

networkConfig must be a dict with the following format:

  {
    "networkType": "L2456Columns",
    "numCorticalColumns": 3,
    "randomSeedBase": 42,

    "sensorParams": {
      <constructor parameters for RawSensor>
    },
    "coarseSensorParams": {
      <constructor parameters for RawSensor>
    },
    "locationParams": {
      <constructor parameters for CoordinateSensorRegion>
    },
    "L2Params": {
      <constructor parameters for ColumnPoolerRegion>
    },
    "L4Params": {
      <constructor parameters for ApicalTMPairRegion
    },
    "L5Params": {
      <constructor parameters for ColumnPoolerRegion>
    },
    "L6Params": {
      <constructor parameters for ApicalTMPairRegion
    }
  }

Each region gets its seed set to randomSeedBase + colNumber


ACKNOWLEDGEMENTS
================

ASCII art made easy by http://asciiflow.com - thank you!

"""
import copy
import json

def enableProfiling(network):
  """Enable profiling for all regions in the network."""
  for region in network.regions.values():
    region.enableProfiling()


def _createL2456Column(network, networkConfig, suffix=""):
  """
  Create a single L2456 column with appropriate suffix on the name.
  """

  locationInputName = "locationInput" + suffix
  sensorInputName = "sensorInput" + suffix
  coarseSensorInputName = "coarseSensorInput" + suffix
  L2ColumnName = "L2Column" + suffix
  L4ColumnName = "L4Column" + suffix
  L5ColumnName = "L5Column" + suffix
  L6ColumnName = "L6Column" + suffix

  # TODO: Convert locationInput to a coordinate sensor region once its ready
  # Add the three sensors to network.
  network.addRegion(
    locationInputName, "py.CoordinateSensorRegion",
    json.dumps(networkConfig["locationParams"]))
  network.addRegion(
    coarseSensorInputName, "py.RawSensor",
    json.dumps(networkConfig["coarseSensorParams"]))
  network.addRegion(
    sensorInputName, "py.RawSensor",
    json.dumps(networkConfig["sensorParams"]))

  # Add L2/L5 column pooler regions
  network.addRegion(
    L2ColumnName, "py.ColumnPoolerRegion",
    json.dumps(networkConfig["L2Params"]))
  network.addRegion(
    L5ColumnName, "py.ColumnPoolerRegion",
    json.dumps(networkConfig["L5Params"]))

  # Add L4/L6 extended temporal memory regions
  L6Params = copy.deepcopy(networkConfig["L6Params"])
  L6Params["basalInputWidth"] = networkConfig["locationParams"]["outputWidth"]
  L6Params["apicalInputWidth"] = networkConfig["L5Params"]["cellCount"]
  network.addRegion(
    L6ColumnName, "py.ApicalTMPairRegion",
    json.dumps(L6Params))

  L4Params = copy.deepcopy(networkConfig["L4Params"])
  L4Params["basalInputWidth"] = (
    L6Params["columnCount"] * L6Params["cellsPerColumn"] )
  L4Params["apicalInputWidth"] = networkConfig["L2Params"]["cellCount"]
  network.addRegion(
    L4ColumnName, "py.ApicalTMPairRegion",
    json.dumps(L4Params))

  # Once regions are created, ensure inputs match column counts
  assert(network.regions[L6ColumnName].getParameter("columnCount") ==
         network.regions[coarseSensorInputName].getParameter("outputWidth")), \
         "L6 column count must equal coarse sensor width"

  assert(network.regions[L4ColumnName].getParameter("columnCount") ==
         network.regions[sensorInputName].getParameter("outputWidth")), \
         "L4 column count must equal sensor width"

  # Link up the sensors
  network.link(locationInputName, L6ColumnName, "UniformLink", "",
               srcOutput="dataOut", destInput="basalInput")
  network.link(coarseSensorInputName, L6ColumnName, "UniformLink", "",
               srcOutput="dataOut", destInput="activeColumns")
  network.link(sensorInputName, L4ColumnName, "UniformLink", "",
               srcOutput="dataOut", destInput="activeColumns")

  # Link L6 to L4
  network.link(L6ColumnName, L4ColumnName, "UniformLink", "",
               srcOutput="activeCells", destInput="basalInput")

  # Link L4 to L2, L6 to L5
  network.link(L4ColumnName, L2ColumnName, "UniformLink", "",
               srcOutput="activeCells", destInput="feedforwardInput")
  network.link(L4ColumnName, L2ColumnName, "UniformLink", "",
               srcOutput="predictedActiveCells",
               destInput="feedforwardGrowthCandidates")
  network.link(L6ColumnName, L5ColumnName, "UniformLink", "",
               srcOutput="activeCells", destInput="feedforwardInput")
  network.link(L6ColumnName, L5ColumnName, "UniformLink", "",
               srcOutput="predictedActiveCells",
               destInput="feedforwardGrowthCandidates")

  # Link L2 feedback to L4, L5 to L6
  network.link(L2ColumnName, L4ColumnName, "UniformLink", "",
               srcOutput="feedForwardOutput", destInput="apicalInput",
               propagationDelay=1)
  network.link(L5ColumnName, L6ColumnName, "UniformLink", "",
               srcOutput="feedForwardOutput", destInput="apicalInput",
               propagationDelay=1)

  # Link reset outputs to L5 and L2. For L6 and L4, an empty input is sufficient
  # for a reset.
  network.link(sensorInputName, L5ColumnName, "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")
  network.link(sensorInputName, L2ColumnName, "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")


  # Set phases appropriately so regions are executed in the proper sequence.
  # This is required particularly when we create multiple columns - the order of
  # execution is not the same as the order of region creation.
  # All sensors have phase 0
  # All L6's have phase 2
  # All L5's have phase 3
  # All L4's have phase 4
  # All L2's have phase 5
  # Note: we skip phase 1 in case we add spatial poolers on top of the sensors.

  network.setPhases(locationInputName,[0])
  network.setPhases(sensorInputName,[0])
  network.setPhases(coarseSensorInputName,[0])
  network.setPhases(L6ColumnName,[2])
  network.setPhases(L5ColumnName,[3])
  network.setPhases(L4ColumnName, [4])
  network.setPhases(L2ColumnName, [5])

  return network


def createL2456Columns(network, networkConfig):
  """
  Create a network consisting of multiple L2456 columns as described in
  the file comments above.
  """

  # Create each column
  numCorticalColumns = networkConfig["numCorticalColumns"]
  for i in xrange(numCorticalColumns):
    networkConfigCopy = copy.deepcopy(networkConfig)
    randomSeedBase = networkConfigCopy["randomSeedBase"]
    networkConfigCopy["L2Params"]["seed"] = randomSeedBase + i
    networkConfigCopy["L4Params"]["seed"] = randomSeedBase + i
    networkConfigCopy["L5Params"]["seed"] = randomSeedBase + i
    networkConfigCopy["L6Params"]["seed"] = randomSeedBase + i
    networkConfigCopy["L2Params"][
      "numOtherCorticalColumns"] = numCorticalColumns - 1
    networkConfigCopy["L5Params"][
      "numOtherCorticalColumns"] = numCorticalColumns - 1

    suffix = "_" + str(i)
    network = _createL2456Column(network, networkConfigCopy, suffix)

  # Now connect the L2 columns laterally to every other L2 column, and
  # the same for L5 columns.
  for i in range(networkConfig["numCorticalColumns"]):
    suffixSrc = "_" + str(i)
    for j in range(networkConfig["numCorticalColumns"]):
      if i != j:
        suffixDest = "_" + str(j)
        network.link(
            "L2Column" + suffixSrc, "L2Column" + suffixDest,
            "UniformLink", "",
            srcOutput="feedForwardOutput", destInput="lateralInput")
        network.link(
            "L5Column" + suffixSrc, "L5Column" + suffixDest,
            "UniformLink", "",
            srcOutput="feedForwardOutput", destInput="lateralInput")

  enableProfiling(network)

  return network
