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
Mimic L2L4Inference capacity test but reuse the multi-column experiment
framework.
"""

import argparse
import math
import os
import random

import numpy as np

from grid_multi_column_experiment import MultiColumn2DExperiment
from tracing import MultiColumn2DExperimentVisualizer as trace

CM_PER_UNIT = 100.0 / 12.0


def doExperiment(numCorticalColumns):
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
    for objectName, placement in OBJECT_PLACEMENTS_LEARN.iteritems())

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

  exp = MultiColumn2DExperiment(
    featureNames=(FEATURES),
    objects=objects,
    objectPlacements=objectPlacements,
    locationConfigs=locationConfigs,
    numCorticalColumns=numCorticalColumns,
    worldDimensions=(100, 100),
    featureW=10,
    cellsPerColumn=16,
  )

  bodyPlacement = [6. * CM_PER_UNIT, 1. * CM_PER_UNIT]
  exp.learnObjects(bodyPlacement)

  bodyPlacement = [6. * CM_PER_UNIT, 11. * CM_PER_UNIT]
  numTouchesRequired = exp.inferObjects(bodyPlacement, maxTouches=10)
  for touches, count in sorted(numTouchesRequired.iteritems()):
    print "{}: {}".format(touches, count)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--columns", default=1, type=int, help="number of cortical columns")
  parser.add_argument("--objects", default=4, type=int, help="number of objects")
  parser.add_argument("--features", default=4, type=int, help="number of features")
  args = parser.parse_args()

  OBJ_MAX_DIM = 10
  POSSIBLE_FEATURE_LOCS = []
  for i in xrange(OBJ_MAX_DIM):
    for j in xrange(OBJ_MAX_DIM):
      POSSIBLE_FEATURE_LOCS.append((i, j))

  FEATURES = ["{}".format(i) for i in xrange(args.features)]

  NUM_OBJECTS = args.objects
  POINTS_PER_OBJ = 10
  DISCRETE_OBJECTS = {}
  OBJECT_PLACEMENTS_LEARN = {}
  for i in xrange(NUM_OBJECTS):
    np.random.shuffle(POSSIBLE_FEATURE_LOCS)
    locs = POSSIBLE_FEATURE_LOCS[0:POINTS_PER_OBJ]
    objName = "Object {}".format(i)
    feats = [np.random.choice(FEATURES) for _ in xrange(POINTS_PER_OBJ)]
    DISCRETE_OBJECTS[objName] = dict(zip(locs, feats))

    objLoc = (np.random.randint(OBJ_MAX_DIM),
              np.random.randint(OBJ_MAX_DIM))
    OBJECT_PLACEMENTS_LEARN[objName] = objLoc

  print "Columns: {} Features: {} Objects: {}".format(args.columns, args.features, args.objects)
  doExperiment(args.columns)
