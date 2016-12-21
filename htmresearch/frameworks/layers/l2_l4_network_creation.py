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
column feeds back to L4. There is an option to include a SpatialPooler
in between the inputs and L4:

             L2Column  <------|
               ^  |           |
               |  |           |
               |  v           |
        --->  L4Column <------|
        |          ^          |
     + - - +       |       + - - +
        SP       reset        SP
     + - - +       |       + - - +
        |          |          |
externalInput  sensorInput -->|

The second network type supported, "MultipleL4L2Columns", allows you to create
N L4L2 columns with the above structure. In this case the L2 columns will also
be laterally connected to one another (each one receives input from all other
columns.)

                ----------------------------------
                |                                |
                v                                v
             L2Column  <------|               L2Column  <------|
               ^  |           |                 ^  |           |
               |  |           |                 |  |           |
               |  v           |                 |  v           |
        --->  L4Column <------|          --->  L4Column <------|
        |          ^          |          |          ^          |
        |          |          |          |          |          |
     + - - +    + - - +       |       + - - +    + - - +       |
        SP         SP       reset        SP         SP       reset
     + - - +    + - - +       |       + - - +    + - - +       |
        |          |          |          |          |          |
externalInput  sensorInput -->|  externalInput  sensorInput -->|

For both network types, regions will be named as shown above plus a suffix
indicating column number, such as "externalInput_0", "L2Column_3", etc. The
reset signal from sensorInput is sent to the other regions.

Also, how do you like my ascii art?

"""
import copy
import json

def enableProfiling(network):
  """Enable profiling for all regions in the network."""
  for region in network.regions.values():
    region.enableProfiling()


def _addLateralSPRegion(network, networkConfig, suffix=""):
  spParams = networkConfig.get("lateralSPParams", {})

  if not spParams:
    return # User has not specified SpatialPooler parameters and we can safely
           # skip this part

  spParams["inputWidth"] = networkConfig["externalInputSize"]

  network.addRegion("lateralSPRegion", "py.SPRegion", json.dumps(spParams))


def _addFeedForwardSPRegion(network, networkConfig, suffix=""):
  spParams = networkConfig.get("feedForwardSPParams", {})

  if not spParams:
    return # User has not specified SpatialPooler parameters and we can safely
           # skip this part

  spParams["inputWidth"] = networkConfig["sensorInputSize"]

  network.addRegion("feedForwardSPRegion", "py.SPRegion", json.dumps(spParams))


def _linkLateralSPRegion(network, networkConfig, externalInputName, L4ColumnName):
  spParams = networkConfig.get("lateralSPParams", {})

  if not spParams:
    # Link sensors to L4, ignoring SP
    network.link(externalInputName, L4ColumnName, "UniformLink", "",
                 srcOutput="dataOut", destInput="externalBasalInput")
    return

  # Link lateral input to SP input, SP output to L4 lateral input
  network.link(externalInputName, "lateralSPRegion", "UniformLink", "",
               srcOutput="dataOut", destInput="bottomUpIn")
  network.link("lateralSPRegion", L4ColumnName, "UniformLink", "",
               srcOutput="bottomUpOut", destInput="externalBasalInput")


def _linkFeedForwardSPRegion(network, networkConfig, sensorInputName, L4ColumnName):
  spParams = networkConfig.get("feedForwardSPParams", {})

  if not spParams:
    # Link sensors to L4, ignoring SP
    network.link(sensorInputName, L4ColumnName, "UniformLink", "",
                 srcOutput="dataOut", destInput="feedForwardInput")
    return

  # Link lateral input to SP input, SP output to L4 lateral input
  network.link(sensorInputName, "feedForwardSPRegion", "UniformLink", "",
               srcOutput="dataOut", destInput="bottomUpIn")
  network.link("feedForwardSPRegion", L4ColumnName, "UniformLink", "",
               srcOutput="bottomUpOut", destInput="feedForwardInput")


def _setLateralSPPhases(network, networkConfig):
  spParams = networkConfig.get("lateralSPParams", {})

  if not spParams:
    return  # User has not specified SpatialPooler parameters and we can safely
    # skip this part

  network.setPhases("lateralSPRegion", [1])


def _setFeedForwardSPPhases(network, networkConfig):
  spParams = networkConfig.get("feedForwardSPParams", {})

  if not spParams:
    return  # User has not specified SpatialPooler parameters and we can safely
    # skip this part

  network.setPhases("feedForwardSPRegion", [1])


def createL4L2Column(network, networkConfig, suffix=""):
  """
  Create a a single column containing one L4 and one L2.

  networkConfig is a dict that must contain the following keys (additional keys
  ok):

    {
      "externalInputSize": 1024,
      "sensorInputSize": 1024,
      "L4Params": {
        <constructor parameters for ExtendedTMRegion
      },
      "L2Params": {
        <constructor parameters for ColumnPoolerRegion>
      },
      "lateralSPParams": {
        <constructor parameters for optional SPRegion>
      },
      "feedForwardSPParams": {
        <constructor parameters for optional SPRegion>
      }
    }

  Region names are externalInput, sensorInput, L4Column, and ColumnPoolerRegion.
  Each name has an optional string suffix appended to it.

  Configuration options:

    "lateralSPParams" and "feedForwardSPParams" are optional. If included
    appropriate spatial pooler regions will be added to the network.

    If externalInputSize is 0, the externalInput sensor (and SP if appropriate)
    will NOT be created. In this case it is expected that L4 will have
    formInternalBasalConnections set to True.
  """

  externalInputName = "externalInput" + suffix
  sensorInputName = "sensorInput" + suffix
  L4ColumnName = "L4Column" + suffix
  L2ColumnName = "L2Column" + suffix

  L4Params = copy.deepcopy(networkConfig["L4Params"])
  L4Params["basalInputWidth"] = networkConfig["externalInputSize"]
  L4Params["apicalInputWidth"] = networkConfig["L2Params"]["cellCount"]

  if networkConfig["externalInputSize"] > 0:
    network.addRegion(
      externalInputName, "py.RawSensor",
      json.dumps({"outputWidth": networkConfig["externalInputSize"]}))
  network.addRegion(
    sensorInputName, "py.RawSensor",
    json.dumps({"outputWidth": networkConfig["sensorInputSize"]}))

  # Fixup network to include SP, if defined in networkConfig
  if networkConfig["externalInputSize"] > 0:
    _addLateralSPRegion(network, networkConfig, suffix)
  _addFeedForwardSPRegion(network, networkConfig, suffix)

  network.addRegion(
    L4ColumnName, "py.ExtendedTMRegion",
    json.dumps(L4Params))
  network.addRegion(
    L2ColumnName, "py.ColumnPoolerRegion",
    json.dumps(networkConfig["L2Params"]))

  # Set phases appropriately so regions are executed in the proper sequence
  # This is required when we create multiple columns - the order of execution
  # is not the same as the order of region creation.
  if networkConfig["externalInputSize"] > 0:
    network.setPhases(externalInputName,[0])
  network.setPhases(sensorInputName,[0])

  _setLateralSPPhases(network, networkConfig)
  _setFeedForwardSPPhases(network, networkConfig)

  # L4 and L2 regions always have phases 2 and 3, respectively
  network.setPhases(L4ColumnName,[2])
  network.setPhases(L2ColumnName,[3])

  # Link SP region(s), if applicable
  if networkConfig["externalInputSize"] > 0:
    _linkLateralSPRegion(network, networkConfig, externalInputName, L4ColumnName)
  _linkFeedForwardSPRegion(network, networkConfig, sensorInputName, L4ColumnName)

  # Link L4 to L2, and L2's feedback to L4
  network.link(L4ColumnName, L2ColumnName, "UniformLink", "",
               srcOutput="feedForwardOutput", destInput="feedforwardInput")
  network.link(L2ColumnName, L4ColumnName, "UniformLink", "",
               srcOutput="feedForwardOutput", destInput="externalApicalInput")

  # Link reset output to L4 and L2
  network.link(sensorInputName, L4ColumnName, "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")
  network.link(sensorInputName, L2ColumnName, "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")

  enableProfiling(network)

  return network


def createMultipleL4L2Columns(network, networkConfig):
  """
  Create a network consisting of multiple columns.  Each column contains one L4
  and one L2, is identical in structure to the network created by
  createL4L2Column. In addition all the L2 columns are fully connected to each
  other through their lateral inputs.

  Region names have a column number appended as in externalInput_0,
  externalInput_1, etc.

  networkConfig must be of the following format (see createL4L2Column for
  further documentation):

    {
      "networkType": "MultipleL4L2Columns",
      "numCorticalColumns": 3,
      "externalInputSize": 1024,
      "sensorInputSize": 1024,
      "L4Params": {
        <constructor parameters for ExtendedTMRegion
      },
      "L2Params": {
        <constructor parameters for ColumnPoolerRegion>
      },
      "lateralSPParams": {
        <constructor parameters for optional SPRegion>
      },
      "feedForwardSPParams": {
        <constructor parameters for optional SPRegion>
      }
    }
  """

  # Create each column
  numCorticalColumns = networkConfig["numCorticalColumns"]
  for i in xrange(numCorticalColumns):
    networkConfigCopy = copy.deepcopy(networkConfig)
    layerConfig = networkConfigCopy["L2Params"]
    layerConfig["seed"] = layerConfig.get("seed", 42) + i

    layerConfig["numOtherCorticalColumns"] = numCorticalColumns - 1

    suffix = "_" + str(i)
    network = createL4L2Column(network, networkConfigCopy, suffix)

  # Now connect the L2 columns laterally
  for i in range(networkConfig["numCorticalColumns"]):
    suffixSrc = "_" + str(i)
    for j in range(networkConfig["numCorticalColumns"]):
      if i != j:
        suffixDest = "_" + str(j)
        network.link(
            "L2Column" + suffixSrc, "L2Column" + suffixDest,
            "UniformLink", "",
            srcOutput="feedForwardOutput", destInput="lateralInput")

  enableProfiling(network)

  return network