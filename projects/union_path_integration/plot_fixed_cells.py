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

"""Plot fixed cell experiments."""

import collections
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

CWD = os.path.dirname(os.path.realpath(__file__))
CHART_DIR = os.path.join(CWD, "charts")


def computeCapacity(results, threshold):
  """Returns largest number of objects with accuracy above threshold."""
  closestBelow = None
  closestAbove = None
  for numObjects, accuracy in sorted(results):
    if accuracy >= threshold:
      if closestAbove is None or closestAbove[0] < numObjects:
        closestAbove = (numObjects, accuracy)
        closestBelow = None
    else:
      if closestBelow is None:
        closestBelow = (numObjects, accuracy)
  if closestBelow is None or closestAbove is None:
    print closestBelow, threshold, closestAbove
    raise ValueError(
        "Results must include a value above and below threshold of {}".format(threshold))

  print "  Capacity threshold is between {} and {}".format(closestAbove[0], closestBelow[0])

  return closestAbove[0]


def chart():
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

  # 5000 cells max
  #
  # Generated with:
  # TODO

  for totalCells in (5000, 10000):
    accuracies = collections.defaultdict(list)
    capacities = {}
    for modules in (1, 2, 3, 4, 5, 6, 7, 8):
      numCells = (int(math.sqrt(int(totalCells / modules))) ** 2) * modules
      with open("results/fixed_cells_{}_cells_{}_modules.json".format(str(totalCells), str(modules)), "r") as f:
        data = json.load(f)
      for exp in data:
        modSize = np.prod(exp[0]["cellDimensions"])
        numFeatures = exp[0]["numFeatures"]
        k = (modSize, numFeatures)
        numObjects = exp[0]["numObjects"]

        failed = exp[1]["convergence"].get("null", 0)
        accuracy = (float(numObjects) - float(failed)) / float(numObjects)

        accuracies[modules].append((numObjects, accuracy))

      moduleCapacity = computeCapacity(accuracies[modules], 0.9)
      objsPerCell = float(moduleCapacity) / float(numCells)
      capacities[modules] = objsPerCell

    x = []
    y = []
    for i, j in sorted(capacities.iteritems()):
      x.append(i)
      y.append(j)

    plt.plot(
      x, y, "o-", label="{} Total Cells".format(str(totalCells)),
    )

  plt.xlabel("Number of Modules")
  plt.ylabel("Capacity (objects per cell)")
  plt.legend(loc="upper right")

  plt.tight_layout()

  plt.savefig(os.path.join(CHART_DIR, "fixed_cells.pdf"))

  plt.clf()


if __name__ == "__main__":
  chart()
