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
When learning objects, reuse locations. Show how sharing locations can cause
inference to take longer.
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
    featureNames=["A", "B", "C"],
    objects={
      "Object 1": {(0,0): "C",
                   (0,1): "B",
                   (1,1): "C",
                   (2,0): "A",
                   (2,1): "C"},
      "Object 2": {(0,0): "A",
                   (0,1): "C",
                   (1,1): "C",
                   (2,0): "C",
                   (2,1): "C"},
      "Object 3": {(2,0): "A",
                   (2,1): "B",
                   (2,2): "C"}
    })

  # Learn how motor commands correspond to changes in location.
  exp.learnTransitions()

  objectPlacements = {
    "Object 1": (4, 4),
    "Object 2": (4, 4),
    "Object 3": (4, 4),
  }

  # Learn objects in egocentric space.
  exp.learnObjects(objectPlacements)

  newObjectPlacements = {
    "Object 1": (3, 3),
    "Object 2": (5, 6),
    "Object 3": (7, 3),
  }

  # Infer the objects without any location input.
  filename = "logs/infer-not-perfect.csv"
  with open(filename, "w") as fileOut:
    print "Logging to", filename
    with trace(exp, csv.writer(fileOut)):
      exp.inferObject(newObjectPlacements, "Object 3", (2,0), [(0,1), (0,1)])


  filename = "logs/infer-recovery.csv"
  with open("logs/infer-recovery.csv", "w") as fileOut:
    print "Logging to", filename
    with trace(exp, csv.writer(fileOut)):
      exp.inferObject(newObjectPlacements, "Object 3", (2,0), [(0,1), (0,-1), (0,1)])


  print "Visualize these CSV files at:"
  print "http://numenta.github.io/htmresearch/visualizations/location-layer/single-layer-2d.html"
  print ("or in a Jupyter notebook with the htmresearchviz0 package and the "
         "printSingleLayer2DExperiment function.")
