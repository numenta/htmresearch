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

"""Plot location module representations during narrowing."""

import collections
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

CWD = os.path.dirname(os.path.realpath(__file__))
TRACE_DIR = os.path.join(CWD, "traces")
CHART_DIR = os.path.join(CWD, "charts")


def chart():
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

  # Convergence vs. number of objects, comparing # unique features
  #
  # Generated with:
  #   python convergence_simulation.py --numObjects 100 --numUniqueFeatures 40 --locationModuleWidth 10 --numSensations 10 --seed1 100 --seed2 357627 --useRawTrace

  data = {}
  with open("traces/4-points-100-cells-100-objects-40-feats.trace", "r") as f:
    obj = None
    # List of steps. Steps are lists of location modules. Location moduels are lists of active cells.
    locReprs = []

    while True:
      try:
        line = f.next().strip()
      except StopIteration:
        break
      if line == "currentObject":
        # Close out previous object
        if obj is not None:
          data[int(obj["name"])] = locReprs

        # Start new object
        locReprs = []
        obj = json.loads(f.next().strip())
      elif line == "locationLayer":
        locationCells = json.loads(f.next().strip())
        locReprs.append([module[0] for module in locationCells])
  if obj is not None and len(locReprs) > 0:
    data[int(obj["name"])] = locReprs

  numSteps = 10
  numModules = 4
  numCells = 100
  numObjs = 3
  width = 15

  fig, axes = plt.subplots(numModules, numObjs)
  finishingSteps = [2, 1, 4]
  #for i, obj in enumerate((4, 9, 10)):
  #for i, obj in enumerate((9, 29, 42)):
  for i, obj in enumerate((15, 11, 12)):
  #for i, obj in enumerate(xrange(10, 20)):
    plotData = np.ones((numCells * numModules, numSteps*width, 3), dtype=np.float32)
    for step, modules in enumerate(data[obj]):
      if step >= numSteps:
        continue
      for module in xrange(numModules):
        cells = [idx + (module * numCells) for idx in modules[module]]
        stepStart = step * width
        stepStop = (step + 1) * width
        plotData[cells, stepStart:stepStop, :] = [0, 0, 0]

    for m in xrange(numModules):
      axes[m, i].add_patch(matplotlib.patches.Rectangle((finishingSteps[i] * width, -1), width, numModules * numCells + 2, color="red", fill=False))
      axes[m, i].set_yticks([])
      if m == 0:
        axes[m, i].set_title("Object {}".format(i + 1))
      if m == 3:
        axes[m, i].set_xticks(np.arange(10) * width + (width / 2))
        axes[m, i].set_xticklabels([str(v+1) for v in np.arange(10)])
        axes[m, i].set(xlabel="Sensations")
      else:
        axes[m, i].set_xticks([])
        axes[m, i].set_xticklabels([])
      if i == 0:
        axes[m, i].set(ylabel="Module {}".format(m))
      #axesm, ii].set_ylim((0, 501))

      axes[m, i].imshow(plotData[m*numCells:(m+1)*numCells], interpolation="none")

  #plt.set(xlabel="Steps", ylabel="Cells")
  #plt.xlabel("Steps")
  #plt.ylabel("Cells")

  filename = os.path.join(CHART_DIR, "location_narrowing.pdf")
  print "Saving", filename
  plt.savefig(filename)


if __name__ == "__main__":
  chart()
