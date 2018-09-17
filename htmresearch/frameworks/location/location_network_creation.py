# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
The methods here contain factories to create networks of multiple layers for
experimenting with grid cell location layer (L6a)
"""
import copy
import json

from htmresearch.frameworks.location.path_integration_union_narrowing import (
  computeRatModuleParametersFromReadoutResolution,
  computeRatModuleParametersFromCellCount)



def createL4L6aLocationColumn(network, config, suffix=""):
  """
  Create a single column network containing L4 and L6a layers. L4 layer
  processes sensor inputs while L6a processes motor commands using grid cell
  modules. Sensory input is represented by the feature's active columns and
  motor input is represented by the displacement vector [dx, dy].

  The grid cell modules used by this network are based on
  :class:`ThresholdedGaussian2DLocationModule` where the firing rate is computed
  from on one or more Gaussian activity bumps. The cells are distributed
  uniformly through the rhombus, packed in the optimal hexagonal arrangement.
  ::

                                  +-------+
                      +---------->|       |<------------+
                      |     +---->|  L4   |--winner---+ |
                      |     |     |       |           | |
                      |     |     +-------+           | |
                      |     |       |   ^             | |
                      |     |       |   |             | |
                      |     |       |   |             | |
                      |     |       v   |             | |
                      |     |     +-------+           | |
                      |     |     |       |           | |
                      |     +---->|  L6a  |<----------+ |
                      |     |     |       |--learnable--+
                      |     |     +-------+
                      |     |         ^
                 feature  reset       |
                      |     |         |
                      |     |         |
                   [sensorInput] [motorInput]


  .. note::
    Region names are "motorInput", "sensorInput", "L4", and "L6a".
    Each name has an optional string suffix appended to it.

  :param network: network to add the column
  :type network: Network
  :param config: Configuration parameters:
                {
                  "sensorInputSize": int,
                  "inverseReadoutResolution": int or float (optional)
                  "L4Params": {
                    constructor parameters for :class:`ApicalTMPairRegion`
                  },
                  "L6aParams": {
                    constructor parameters for :class:`Guassian2DLocationRegion`
                  }
                }

  :type config: dict
  :param suffix: optional string suffix appended to region name. Useful when
                 creating multicolumn networks.
  :type suffix: str

  :return: Reference to the given network
  :rtype: Network
  """
  L6aConfig = copy.deepcopy(config["L6aParams"])
  if "inverseReadoutResolution" in config:
    # Configure L6a based on 'resolution'
    resolution = config.pop("inverseReadoutResolution")
    params = computeRatModuleParametersFromReadoutResolution(resolution)
    L6aConfig.update(params)
  else:
    baselineCellsPerAxis = L6aConfig.get("baselineCellsPerAxis", 6)
    params = computeRatModuleParametersFromCellCount(L6aConfig["cellsPerAxis"],
                                                     baselineCellsPerAxis)
    L6aConfig.update(params)

  # Configure L4 'basalInputSize' to be compatible L6a output
  moduleCount = L6aConfig["moduleCount"]
  cellsPerAxis = L6aConfig["cellsPerAxis"]
  L4Config = copy.deepcopy(config["L4Params"])
  L4Config["basalInputWidth"] = moduleCount * cellsPerAxis * cellsPerAxis

  # Add regions to network
  motorInputName = "motorInput" + suffix
  sensorInputName = "sensorInput" + suffix
  L4Name = "L4" + suffix
  L6aName = "L6a" + suffix

  network.addRegion(sensorInputName, "py.RawSensor",
                    json.dumps({"outputWidth": config["sensorInputSize"]}))
  network.addRegion(motorInputName, "py.RawValues",
                    json.dumps({"outputWidth": 2}))
  network.addRegion(L4Name, "py.ApicalTMPairRegion", json.dumps(L4Config))
  network.addRegion(L6aName, "py.Guassian2DLocationRegion",
                    json.dumps(L6aConfig))

  # Link sensory input to L4
  network.link(sensorInputName, L4Name, "UniformLink", "",
               srcOutput="dataOut", destInput="activeColumns")

  # Link motor input to L6a
  network.link(motorInputName, L6aName, "UniformLink", "",
               srcOutput="dataOut", destInput="displacement")

  # Link L6a to L4
  network.link(L6aName, L4Name, "UniformLink", "",
               srcOutput="activeCells", destInput="basalInput")
  network.link(L6aName, L4Name, "UniformLink", "",
               srcOutput="learnableCells", destInput="basalGrowthCandidates")

  # Link L4 feedback to L6a
  network.link(L4Name, L6aName, "UniformLink", "",
               srcOutput="activeCells", destInput="anchorInput",
               propagationDelay=1)
  network.link(L4Name, L6aName, "UniformLink", "",
               srcOutput="winnerCells", destInput="anchorGrowthCandidates",
               propagationDelay=1)

  # Link reset signal to L4 and L6a
  network.link(sensorInputName, L4Name, "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")
  network.link(sensorInputName, L6aName, "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")

  # Set phases appropriately
  network.setPhases(sensorInputName, [0])
  network.setPhases(motorInputName, [0])
  network.setPhases(L4Name, [1])
  network.setPhases(L6aName, [1])

  return network



def createL246aLocationColumn(network, config, suffix=""):
  """
  Create a single column network composed of L2, L4 and L6a layers.
  L2 layer computes the object representation using :class:`ColumnPoolerRegion`,
  L4 layer processes sensors input while L6a processes motor commands using grid
  cell modules. Sensory input is represented by the feature's active columns and
  motor input is represented by the displacement vector [dx, dy].

  The grid cell modules used by this network are based on
  :class:`ThresholdedGaussian2DLocationModule` where the firing rate is computed
  from on one or more Gaussian activity bumps. The cells are distributed
  uniformly through the rhombus, packed in the optimal hexagonal arrangement.
  ::

                               +-------+
                        reset  |       |
                        +----->|  L2   |<------------------+
                        |      |       |                   |
                        |      +-------+                   |
                        |        |   ^                     |
                        |        |   |                     |
                        |        |   |                     |
                        |        v   |                     |
                        |      +-------+                   |
                  +----------->|       |--predictedActive--+
                  |     |      |  L4   |<------------+
                  |     +----->|       |--winner---+ |
                  |     |      +-------+           | |
                  |     |        |   ^             | |
                  |     |        |   |             | |
                  |     |        |   |             | |
                  |     |        v   |             | |
                  |     |      +-------+           | |
                  |     |      |       |           | |
                  |     +----->|  L6a  |<----------+ |
                  |     |      |       |--learnable--+
                  |     |      +-------+
             feature  reset        ^
                  |     |          |
                  |     |          |
               [sensorInput]  [motorInput]


  .. note::
    Region names are "motorInput", "sensorInput". "L2", "L4", and "L6a".
    Each name has an optional string suffix appended to it.

  :param network: network to add the column
  :type network: Network
  :param config: Configuration parameters:
                {
                  "sensorInputSize": int,
                  "enableFeedback": True,
                  "inverseReadoutResolution": int or float (optional)
                  "L2Params": {
                    constructor parameters for :class:`ColumnPoolerRegion`
                  },
                  "L4Params": {
                    constructor parameters for :class:`ApicalTMPairRegion`
                  },
                  "L6aParams": {
                    constructor parameters for :class:`Guassian2DLocationRegion`
                  }
                }

  :type config: dict
  :param suffix: optional string suffix appended to region name. Useful when
                 creating multicolumn networks.
  :type suffix: str

  :return: Reference to the given network
  :rtype: Network
  """

  # Add L4 - L6a location layers
  network = createL4L6aLocationColumn(network, config, suffix)
  L4Name = "L4" + suffix
  sensorInputName = "sensorInput" + suffix

  # Add L2 - L4 object layers
  L2Name = "L2" + suffix
  L2Params = copy.deepcopy(config["L2Params"])
  network.addRegion(L2Name, "py.ColumnPoolerRegion", json.dumps(L2Params))

  # Link L4 to L2
  network.link(L4Name, L2Name, "UniformLink", "",
               srcOutput="activeCells", destInput="feedforwardInput")
  network.link(L4Name, L2Name, "UniformLink", "",
               srcOutput="predictedActiveCells",
               destInput="feedforwardGrowthCandidates")

  # Link L2 feedback to L4
  if config.get("enableFeedback", True):
    network.link(L2Name, L4Name, "UniformLink", "",
                 srcOutput="feedForwardOutput", destInput="apicalInput",
                 propagationDelay=1)

  # Link reset output to L2
  network.link(sensorInputName, L2Name, "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")

  # Set L2 phase to be after L4
  network.setPhases(L2Name, [3])

  return network
