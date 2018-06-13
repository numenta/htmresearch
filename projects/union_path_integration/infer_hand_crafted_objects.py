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
Use path integration of unions to recognize hand-crafted objects. Output a
visualization of the experiment.
"""

import math
import io
import os
import random

import numpy as np

from htmresearch.frameworks.location.path_integration_union_narrowing import (
  PIUNCorticalColumn, PIUNExperiment)
from two_layer_tracing import PIUNVisualizer as trace


OBJECTS = [
  {"name": "Object 1",
   "features": [{"top": 0, "left": 0, "width": 10, "height": 10, "name": "A"},
                {"top": 0, "left": 10, "width": 10, "height": 10, "name": "B"},
                {"top": 0, "left": 20, "width": 10, "height": 10, "name": "A"},
                {"top": 10, "left": 0, "width": 10, "height": 10, "name": "A"},
                {"top": 10, "left": 20, "width": 10, "height": 10, "name": "A"}]},
  {"name": "Object 2",
   "features": [{"top": 0, "left": 10, "width": 10, "height": 10, "name": "A"},
                {"top": 10, "left": 0, "width": 10, "height": 10, "name": "B"},
                {"top": 10, "left": 10, "width": 10, "height": 10, "name": "B"},
                {"top": 10, "left": 20, "width": 10, "height": 10, "name": "B"},
                {"top": 20, "left": 10, "width": 10, "height": 10, "name": "A"}]},
  {"name": "Object 3",
   "features": [{"top": 0, "left": 10, "width": 10, "height": 10, "name": "A"},
                {"top": 10, "left": 0, "width": 10, "height": 10, "name": "A"},
                {"top": 10, "left": 10, "width": 10, "height": 10, "name": "B"},
                {"top": 10, "left": 20, "width": 10, "height": 10, "name": "A"},
                {"top": 20, "left": 0, "width": 10, "height": 10, "name": "B"},
                {"top": 20, "left": 10, "width": 10, "height": 10, "name": "A"},
                {"top": 20, "left": 20, "width": 10, "height": 10, "name": "B"}]},
  {"name": "Object 4",
   "features": [{"top": 0, "left": 0, "width": 10, "height": 10, "name": "A"},
                {"top": 0, "left": 10, "width": 10, "height": 10, "name": "A"},
                {"top": 0, "left": 20, "width": 10, "height": 10, "name": "A"},
                {"top": 10, "left": 0, "width": 10, "height": 10, "name": "A"},
                {"top": 10, "left": 20, "width": 10, "height": 10, "name": "B"},
                {"top": 20, "left": 0, "width": 10, "height": 10, "name": "B"},
                {"top": 20, "left": 10, "width": 10, "height": 10, "name": "B"},
                {"top": 20, "left": 20, "width": 10, "height": 10, "name": "B"}]},
]


def doExperiment(cellDimensions, cellCoordinateOffsets):
  """
  Learn a set of objects. Then try to recognize each object. Output an
  interactive visualization.

  @param cellDimensions (pair)
  The cell dimensions of each module

  @param cellCoordinateOffsets (sequence)
  The "cellCoordinateOffsets" parameter for each module
  """
  if not os.path.exists("traces"):
    os.makedirs("traces")

  locationConfigs = []
  for i in xrange(5):
    scale = 10.0 * (math.sqrt(2) ** i)

    for _ in xrange(4):
      orientation = np.radians(random.gauss(7.5, 7.5))
      orientation = random.choice([orientation, -orientation])

      locationConfigs.append({
        "cellDimensions": cellDimensions,
        "moduleMapDimensions": (scale, scale),
        "orientation": orientation,
        "cellCoordinateOffsets": cellCoordinateOffsets,
      })

  column = PIUNCorticalColumn(locationConfigs)
  exp = PIUNExperiment(column, featureNames=("A", "B"))

  for objectDescription in OBJECTS:
    exp.learnObject(objectDescription)

  filename = "traces/{}-points-{}-cells.html".format(
    len(cellCoordinateOffsets)**2, np.prod(cellDimensions))

  with io.open(filename, "w", encoding="utf8") as fileOut:
    with trace(fileOut, exp, includeSynapses=True):
      print "Logging to", filename
      for objectDescription in OBJECTS:
        succeeded = exp.inferObjectWithRandomMovements(objectDescription)
        if not succeeded:
          print 'Failed to infer object "{}"'.format(objectDescription["name"])



if __name__ == "__main__":
  doExperiment(cellDimensions=(5, 5),
               cellCoordinateOffsets=(0.5,))

  doExperiment(cellDimensions=(5, 5),
               cellCoordinateOffsets=(0.05, 0.5, 0.95))

  doExperiment(cellDimensions=(10, 10),
               cellCoordinateOffsets=(0.05, 0.5, 0.95))
