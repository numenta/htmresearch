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
from nupic.engine import Network



def createL4L6aLocationColumn(network, L4Params, L6aParams,
                              inverseReadoutResolution=None,
                              baselineCellsPerAxis=6, suffix=""):
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

    Phase
    -----                    +-------+
                 +---------->|       |<------------+
     [2]         |     +---->|  L4   |--winner---+ |
                 |     |     |       |           | |
                 |     |     +-------+           | |
                 |     |       |   ^             | |
                 |     |       |   |             | |
                 |     |       |   |             | |
                 |     |       v   |             | |
                 |     |     +-------+           | |
                 |     |     |       |           | |
     [1,3]       |     +---->|  L6a  |<----------+ |
                 |     |     |       |--learnable--+
                 |     |     +-------+
                 |     |         ^
            feature  reset       |
                 |     |         |
                 |     |         |
     [0]      [sensorInput] [motorInput]


  .. note::
    Region names are "motorInput", "sensorInput", "L4", and "L6a".
    Each name has an optional string suffix appended to it.

  :param network: network to add the column
  :type network: Network
  :param L4Params:  constructor parameters for :class:`ApicalTMPairRegion`
  :type L4Params: dict
  :param L6aParams:  constructor parameters for :class:`GridCellLocationRegion`
  :type L6aParams: dict
  :param inverseReadoutResolution: Optional readout resolution.
    The readout resolution specifies the diameter of the circle of phases in the
    rhombus encoded by a bump. See `createRatModuleFromReadoutResolution.
  :type inverseReadoutResolution: int
  :param baselineCellsPerAxis: The baselineCellsPerAxis implies the readout
    resolution of a grid cell module. If baselineCellsPerAxis=6, that implies
    that the readout resolution is approximately 1/3. If baselineCellsPerAxis=8,
    the readout resolution is approximately 1/4
  :type baselineCellsPerAxis: int or float
  :param suffix: optional string suffix appended to region name. Useful when
                 creating multicolumn networks.
  :type suffix: str

  :return: Reference to the given network
  :rtype: Network
  """
  L6aParams = copy.deepcopy(L6aParams)
  if inverseReadoutResolution is not None:
    # Configure L6a based on 'resolution'
    params = computeRatModuleParametersFromReadoutResolution(inverseReadoutResolution)
    L6aParams.update(params)
  else:
    params = computeRatModuleParametersFromCellCount(L6aParams["cellsPerAxis"],
                                                     baselineCellsPerAxis)
    L6aParams.update(params)

  numOfcols = L4Params["columnCount"]
  cellsPerCol = L4Params["cellsPerColumn"]
  L6aParams["anchorInputSize"] = numOfcols * cellsPerCol

  # Configure L4 'basalInputSize' to be compatible L6a output
  moduleCount = L6aParams["moduleCount"]
  cellsPerAxis = L6aParams["cellsPerAxis"]

  L4Params = copy.deepcopy(L4Params)
  L4Params["basalInputWidth"] = moduleCount * cellsPerAxis * cellsPerAxis

  # Configure sensor output to be compatible with L4 params
  columnCount = L4Params["columnCount"]

  # Add regions to network
  motorInputName = "motorInput" + suffix
  sensorInputName = "sensorInput" + suffix
  L4Name = "L4" + suffix
  L6aName = "L6a" + suffix

  network.addRegion(sensorInputName, "py.RawSensor",
                    json.dumps({"outputWidth": columnCount}))
  network.addRegion(motorInputName, "py.RawValues",
                    json.dumps({"outputWidth": 2}))
  network.addRegion(L4Name, "py.ApicalTMPairRegion", json.dumps(L4Params))
  network.addRegion(L6aName, "py.GridCellLocationRegion",
                    json.dumps(L6aParams))

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
               srcOutput="activeCells", destInput="anchorInput")
  network.link(L4Name, L6aName, "UniformLink", "",
               srcOutput="winnerCells", destInput="anchorGrowthCandidates")

  # Link reset signal to L4 and L6a
  network.link(sensorInputName, L4Name, "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")
  network.link(sensorInputName, L6aName, "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")

  # Set phases appropriately
  network.setPhases(motorInputName, [0])
  network.setPhases(sensorInputName, [0])
  network.setPhases(L4Name, [2])
  network.setPhases(L6aName, [1, 3])

  return network



def createL246aLocationColumn(network, L2Params, L4Params, L6aParams,
                              baselineCellsPerAxis=6,
                              inverseReadoutResolution=None, suffix=""):
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

   Phase                       +-------+
   -----                reset  |       |
                        +----->|  L2   |<------------------+
   [3]                  |      |       |                   |
                        |      +-------+                   |
                        |        |   ^                     |
                        |        |   |                     |
                        |     +1 |   |                     |
                        |        v   |                     |
                        |      +-------+                   |
                  +----------->|       |--winnerCells------+
   [2]            |     |      |  L4   |<------------+
                  |     +----->|       |--winner---+ |
                  |     |      +-------+           | |
                  |     |        |   ^             | |
                  |     |        |   |             | |
                  |     |        |   |             | |
                  |     |        v   |             | |
                  |     |      +-------+           | |
                  |     |      |       |           | |
    [1,3]         |     +----->|  L6a  |<----------+ |
                  |     |      |       |--learnable--+
                  |     |      +-------+
             feature  reset        ^
                  |     |          |
                  |     |          |
    [0]        [sensorInput]  [motorInput]


  .. note::
    Region names are "motorInput", "sensorInput". "L2", "L4", and "L6a".
    Each name has an optional string suffix appended to it.

  :param network: network to add the column
  :type network: Network
  :param L2Params:  constructor parameters for :class:`ColumnPoolerRegion`
  :type L2Params: dict
  :param L4Params:  constructor parameters for :class:`ApicalTMPairRegion`
  :type L4Params: dict
  :param L6aParams:  constructor parameters for :class:`GridCellLocationRegion`
  :type L6aParams: dict
  :param inverseReadoutResolution: Optional readout resolution.
    The readout resolution specifies the diameter of the circle of phases in the
    rhombus encoded by a bump. See `createRatModuleFromReadoutResolution.
  :type inverseReadoutResolution: int
  :param baselineCellsPerAxis: The baselineCellsPerAxis implies the readout
    resolution of a grid cell module. If baselineCellsPerAxis=6, that implies
    that the readout resolution is approximately 1/3. If baselineCellsPerAxis=8,
    the readout resolution is approximately 1/4
  :type baselineCellsPerAxis: int or float
  :param suffix: optional string suffix appended to region name. Useful when
                 creating multicolumn networks.
  :type suffix: str
  :return: Reference to the given network
  :rtype: Network
  """

  # Configure L2 'inputWidth' to be compatible with L4
  numOfcols = L4Params["columnCount"]
  cellsPerCol = L4Params["cellsPerColumn"]
  L2Params = copy.deepcopy(L2Params)
  L2Params["inputWidth"] = numOfcols * cellsPerCol

  # Configure L4 'apicalInputWidth' to be compatible L2 output
  L4Params = copy.deepcopy(L4Params)
  L4Params["apicalInputWidth"] = L2Params["cellCount"]

  # Add L4 - L6a location layers
  network = createL4L6aLocationColumn(network=network,
                                      L4Params=L4Params,
                                      L6aParams=L6aParams,
                                      inverseReadoutResolution=inverseReadoutResolution,
                                      baselineCellsPerAxis=baselineCellsPerAxis,
                                      suffix=suffix)
  L4Name = "L4" + suffix
  sensorInputName = "sensorInput" + suffix

  # Add L2 - L4 object layers
  L2Name = "L2" + suffix
  network.addRegion(L2Name, "py.ColumnPoolerRegion", json.dumps(L2Params))

  # Link L4 to L2
  network.link(L4Name, L2Name, "UniformLink", "",
               srcOutput="activeCells", destInput="feedforwardInput")
  network.link(L4Name, L2Name, "UniformLink", "",
               srcOutput="winnerCells",
               destInput="feedforwardGrowthCandidates")

  # Link L2 feedback to L4
  network.link(L2Name, L4Name, "UniformLink", "",
               srcOutput="feedForwardOutput", destInput="apicalInput",
               propagationDelay=1)

  # Link reset output to L2
  network.link(sensorInputName, L2Name, "UniformLink", "",
               srcOutput="resetOut", destInput="resetIn")

  # Set L2 phase to be after L4
  network.setPhases(L2Name, [3])

  return network



def createMultipleL246aLocationColumn(network, numberOfColumns, L2Params,
                                      L4Params, L6aParams,
                                      inverseReadoutResolution=None,
                                      baselineCellsPerAxis=6):
  """
  Create a network consisting of multiple columns. Each column contains one L2,
  one L4 and one L6a layers identical in structure to the network created by
  :func:`createL246aLocationColumn`. In addition all the L2 columns are fully
  connected to each other through their lateral inputs.
  ::

                            +----lateralInput--+
                            | +--------------+ |
                            | |       +1     | |
 Phase                      v |              v |
 -----                   +-------+         +-------+
                  reset  |       |         |       | reset
 [3]              +----->|  L2   |         |  L2   |<----+
                  |      |       |         |       |     |
                  |      +-------+         +-------+     |
                  |        |   ^             |   ^       |
                  |     +1 |   |          +1 |   |       |
                  |        |   |             |   |       |
                  |        v   |             v   |       |
                  |      +-------+         +-------+     |
 [2]        +----------->|       |         |       |<----------+
            |     |      |  L4   |         |  L4   |     |     |
            |     +----->|       |         |       |<----+     |
            |     |      +-------+         +-------+     |     |
            |     |        |   ^             |   ^       |     |
            |     |        |   |             |   |       |     |
            |     |        |   |             |   |       |     |
            |     |        v   |             v   |       |     |
            |     |      +-------+         +-------+     |     |
            |     |      |       |         |       |     |     |
 [1,3]      |     +----->|  L6a  |         |  L6a  |<----+     |
            |     |      |       |         |       |     |     |
            |     |      +-------+         +-------+     |     |
       feature  reset        ^                 ^      reset  feature
            |     |          |                 |         |     |
            |     |          |                 |         |     |
 [0]     [sensorInput]  [motorInput]      [motorInput] [sensorInput]

  .. note::
    Region names are "motorInput", "sensorInput". "L2", "L4", and "L6a".
    Each name has column number appended to it.
    For example: "sensorInput_0", "L2_1", "L6a_0" etc.

  :param network: network to add the column
  :type network: Network
  :param numberOfColumns: Number of columns to create
  :type numberOfColumns: int
  :param L2Params:  constructor parameters for :class:`ColumnPoolerRegion`
  :type L2Params: dict
  :param L4Params:  constructor parameters for :class:`ApicalTMPairRegion`
  :type L4Params: dict
  :param L6aParams:  constructor parameters for :class:`GridCellLocationRegion`
  :type L6aParams: dict
  :param inverseReadoutResolution: Optional readout resolution.
    The readout resolution specifies the diameter of the circle of phases in the
    rhombus encoded by a bump. See `createRatModuleFromReadoutResolution.
  :type inverseReadoutResolution: int
  :param baselineCellsPerAxis: The baselineCellsPerAxis implies the readout
    resolution of a grid cell module. If baselineCellsPerAxis=6, that implies
    that the readout resolution is approximately 1/3. If baselineCellsPerAxis=8,
    the readout resolution is approximately 1/4
  :type baselineCellsPerAxis: int or float
  :return: Reference to the given network
  :rtype: Network
  """
  L2Params = copy.deepcopy(L2Params)
  L4Params = copy.deepcopy(L4Params)
  L6aParams = copy.deepcopy(L6aParams)

  # Update L2 numOtherCorticalColumns parameter
  L2Params["numOtherCorticalColumns"] = numberOfColumns - 1

  for i in xrange(numberOfColumns):
    # Make sure random seed is different for each column
    L2Params["seed"] = L2Params.get("seed", 42) + i
    L4Params["seed"] = L4Params.get("seed", 42) + i
    L6aParams["seed"] = L6aParams.get("seed", 42) + i

    # Create column
    network = createL246aLocationColumn(network=network,
                                        L2Params=L2Params,
                                        L4Params=L4Params,
                                        L6aParams=L6aParams,
                                        inverseReadoutResolution=inverseReadoutResolution,
                                        baselineCellsPerAxis=baselineCellsPerAxis,
                                        suffix="_" + str(i))

  # Now connect the L2 columns laterally
  if numberOfColumns > 1:
    for i in xrange(numberOfColumns):
      src = str(i)
      for j in xrange(numberOfColumns):
        if i != j:
          dest = str(j)
          network.link(
            "L2_" + src, "L2_" + dest,
            "UniformLink", "",
            srcOutput="feedForwardOutput", destInput="lateralInput",
            propagationDelay=1)

  return network
