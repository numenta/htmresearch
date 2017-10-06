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
Mimic the location_module_experiment, but this time use multiple cortical
columns and egocentric locations.
"""

import math
import os
import random

import numpy as np

from grid_multi_column_experiment import MultiColumn2DExperiment
from tracing import MultiColumn2DExperimentVisualizer as trace


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


def doExperiment():
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

  cellDimensions = (10, 10)

  locationConfigs = []
  for i in xrange(4):
    scale = 10.0 * (math.sqrt(2) ** i)

    for _ in xrange(4):
      orientation = random.gauss(7.5, 7.5) * math.pi / 180.0
      orientation = random.choice([orientation, -orientation])

      locationConfigs.append({
        "cellDimensions": cellDimensions,
        "moduleMapDimensions": (scale, scale),
        "orientation": orientation,
      })

  print("Initializing experiment")
  exp = MultiColumn2DExperiment(
    featureNames=("A", "B"),
    objects=objects,
    objectPlacements=objectPlacements,
    locationConfigs=locationConfigs,
    numCorticalColumns=3,
    worldDimensions=(100, 100))

  print("Learning objects")
  filename = "logs/{}-cells-learn.log".format(np.prod(cellDimensions))
  with open(filename, "w") as fileOut:
    with trace(exp, fileOut, includeSynapses=True):
      print "Logging to", filename
      bodyPlacement = [6. * CM_PER_UNIT, 1. * CM_PER_UNIT]
      exp.learnObjects(bodyPlacement)

  filename = "logs/{}-cells-infer.log".format(np.prod(cellDimensions))
  with open(filename, "w") as fileOut:
    with trace(exp, fileOut, includeSynapses=True):
      print "Logging to", filename

      bodyPlacement = [6. * CM_PER_UNIT, 11. * CM_PER_UNIT]
      exp.inferObjectsWithTwoTouches(bodyPlacement)


if __name__ == "__main__":
  doExperiment()

  print "Visualize these logs at:"
  print "http://numenta.github.io/htmresearch/visualizations/location-layer/multi-column-inference.html"
  print ("or in a Jupyter notebook with the htmresearchviz0 package and the "
         "printMultiColumnInferenceRecording function.")
