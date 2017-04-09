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
Learn a set of 2D objects in one location, then infer the objects when they're
in other locations.
"""

import csv
import os

from runner import SingleLayerLocation2DExperiment
from logging import SingleLayer2DExperimentVisualizer as trace



if __name__ == "__main__":
  if not os.path.exists("logs"):
    os.makedirs("logs")

  exp = SingleLayerLocation2DExperiment(
    diameter=12,
    featureNames=["A", "B"],
    objects={
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
                   (2,2): "B"}
    })

  # Learn how motor commands correspond to changes in location.
  exp.learnTransitions()

  objectPlacements = {
    "Object 1": (2, 3),
    "Object 2": (6, 2),
    "Object 3": (3, 7),
    "Object 4": (7, 6)
  }

  # Learn objects in egocentric space.
  exp.learnObjects(objectPlacements)

  # Infer the objects without any location input.
  filename = "logs/infer-no-location.csv"
  with open(filename, "w") as fileOut:
    print "Logging to", filename
    with trace(exp, csv.writer(fileOut)):
      exp.inferObjectsWithRandomMovements(objectPlacements)

  # Shuffle the objects. Infer them without any location input.
  filename = "logs/infer-shuffled-location.csv"
  with open(filename, "w") as fileOut:
    print "Logging to", filename
    with trace(exp, csv.writer(fileOut)):
      exp.inferObjectsWithRandomMovements({
        "Object 1": (7, 6),
        "Object 2": (2, 7),
        "Object 3": (7, 2),
        "Object 4": (3, 3)
      })

  print ("Visualize these CSV files at "
         "http://numenta.github.io/nupic.research/visualizations/location-layer/single-layer-2d.html "
         "or in a Jupyter notebook with the htmresearchviz0 package and the "
         "printSingleLayer2DExperiment function.")
