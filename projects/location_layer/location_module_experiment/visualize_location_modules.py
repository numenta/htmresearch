# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
Visualize an array of location modules while moving a sensor along simple paths.
"""

from __future__ import print_function
import csv
import json
import math
import os
import random

from htmresearch.algorithms.location_modules import Superficial2DLocationModule


def go():
  if not os.path.exists("logs"):
    os.makedirs("logs")

  filename = "logs/isolated-modules.log"
  with open(filename, "w") as fileOut:
    print ("Logging to", filename)

    worldWidth = 100.0
    worldHeight = 100.0

    stepsPerScale = 5

    locationConfigs = []
    for i in xrange(9):
      scale = 10.0 * (math.sqrt(2) ** i)

      for _ in xrange(2):
        orientation = random.gauss(7.5, 7.5) * math.pi / 180.0
        orientation = random.choice([orientation, -orientation])

        locationConfigs.append({
          "cellDimensions": (5, 5),
          "moduleMapDimensions": (scale, scale),
          "orientation": orientation,
          "cellCoordinateOffsets": (0.5,),
        })

    print(json.dumps({"width": worldWidth,
                      "height": worldHeight}), file=fileOut)
    print(json.dumps(locationConfigs), file=fileOut)

    modules = [Superficial2DLocationModule(anchorInputSize=0, **config)
               for config in locationConfigs]

    location = [5.0, 5.0]

    for module in modules:
      module.activateRandomLocation()

    for deltaLocation in 30*[[2.0, 2.5]]:
      location[0] += deltaLocation[0]
      location[1] += deltaLocation[1]

      print("move", file=fileOut)
      print(json.dumps(list(deltaLocation)), file=fileOut)
      print("locationInWorld", file=fileOut)
      print(json.dumps(location), file=fileOut)

      for module in modules:
        module.movementCompute(deltaLocation)

      print("shift", file=fileOut)
      cellsByModule = [module.getActiveCells().tolist()
                       for module in modules]
      print(json.dumps(cellsByModule), file=fileOut)
      pointsByModule = []
      for module in modules:
        pointsByModule.append((module.activePhases * module.cellDimensions).tolist())
      print(json.dumps(pointsByModule), file=fileOut)


if __name__ == "__main__":
  go()
  print("Visualize this CSV file at:")
  print("http://numenta.github.io/htmresearch/visualizations/location-layer/location-modules.html")
  print("or in a Jupyter notebook with the htmresearchviz0 package and the "
        "printLocationModulesRecording function.")
