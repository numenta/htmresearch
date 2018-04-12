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
Mimic the single_layer_2d_experiment, but this time use location modules.
"""

import math
import os
import random

import numpy as np

from grid_2d_location_experiment import Grid2DLocationExperiment
from three_layer_tracing import Grid2DLocationExperimentVisualizer as trace


DISCRETE_OBJECTS = {
  "Object 1": {(0,0): "A",
               (0,1): "B",
               (0,2): "A",
               (1,0): "A",
               (1,2): "A"},
  "Object 2": {(0,1): "A",
               (1,0): "B",
               (1,1): "B",
               (1,2): "B",
               (2,1): "A"},
  "Object 3": {(0,1): "A",
               (1,0): "A",
               (1,1): "B",
               (1,2): "A",
               (2,0): "B",
               (2,1): "A",
               (2,2): "B"},
  "Object 4": {(0,0): "A",
               (0,1): "A",
               (0,2): "A",
               (1,0): "A",
               (1,2): "B",
               (2,0): "B",
               (2,1): "B",
               (2,2): "B"},
}

DISCRETE_OBJECT_PLACEMENTS = {
  "Object 1": (2, 3),
  "Object 2": (6, 2),
  "Object 3": (3, 7),
  "Object 4": (7, 6)
}

CM_PER_UNIT = 100.0 / 12.0


def doExperiment(cellDimensions, cellCoordinateOffsets):
  if not os.path.exists("logs"):
    os.makedirs("logs")

  objects = dict(
    (objectName, [{"top": location[0] * CM_PER_UNIT,
                   "left": location[1] * CM_PER_UNIT,
                   "width": CM_PER_UNIT,
                   "height": CM_PER_UNIT,
                   "name": featureName}
                  for location, featureName in objectDict.iteritems()])
    for objectName, objectDict in DISCRETE_OBJECTS.iteritems())

  objectPlacements = dict(
    (objectName, [placement[0] * CM_PER_UNIT,
                  placement[1] * CM_PER_UNIT])
    for objectName, placement in DISCRETE_OBJECT_PLACEMENTS.iteritems())

  locationConfigs = []
  for i in xrange(9):
    scale = 10.0 * (math.sqrt(2) ** i)

    for _ in xrange(2):
      orientation = random.gauss(7.5, 7.5) * math.pi / 180.0
      orientation = random.choice([orientation, -orientation])

      locationConfigs.append({
        "cellDimensions": cellDimensions,
        "moduleMapDimensions": (scale, scale),
        "orientation": orientation,
        "cellCoordinateOffsets": cellCoordinateOffsets,
      })

  exp = Grid2DLocationExperiment(
    featureNames=("A", "B"),
    objects=objects,
    objectPlacements=objectPlacements,
    locationConfigs=locationConfigs,
    worldDimensions=(100, 100))

  exp.learnObjects()

  filename = "logs/{}-points-{}-cells.log".format(
    len(cellCoordinateOffsets)**2, np.prod(cellDimensions))
  synapseFilename = "logs/{}-points-{}-cells-synapses.log".format(
    len(cellCoordinateOffsets)**2, np.prod(cellDimensions))

  with open(filename, "w") as fileOut, \
       open(synapseFilename, "w") as synapseFileOut:
    with trace(exp, fileOut, includeSynapses=False), \
         trace(exp, synapseFileOut, includeSynapses=True):
      print "Logging to", filename
      print "Logging to", synapseFilename
      exp.inferObjectsWithRandomMovements()



if __name__ == "__main__":
  doExperiment(cellDimensions=(5, 5),
               cellCoordinateOffsets=(0.5,))

  doExperiment(cellDimensions=(5, 5),
               cellCoordinateOffsets=(0.05, 0.5, 0.95))

  doExperiment(cellDimensions=(10, 10),
               cellCoordinateOffsets=(0.05, 0.5, 0.95))

  print "Visualize these logs at:"
  print "http://numenta.github.io/htmresearch/visualizations/location-layer/location-module-inference.html"
  print ("or in a Jupyter notebook with the htmresearchviz0 package and the "
         "printLocationModuleInference function.")
