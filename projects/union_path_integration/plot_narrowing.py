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

import argparse
import collections
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

CWD = os.path.dirname(os.path.realpath(__file__))
TRACE_DIR = os.path.join(CWD, "traces")
CHART_DIR = os.path.join(CWD, "charts")


def examplesChart(inFilename, outFilename, objectCount, objectNumbers):
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

  with open(inFilename, "r") as f:
    experiments = json.load(f)
  for exp in experiments:
    numObjects = exp[0]["numObjects"]
    if numObjects == objectCount:
      locationLayerTimelineByObject = dict(
        (int(k), v)
        for k, v in exp[1]["locationLayerTimelineByObject"].iteritems())
      inferredStepByObject = dict(
        (int(k), v)
        for k, v in exp[1]["inferredStepByObject"].iteritems())
      break

  numSteps = 9
  numModules = 4
  numCells = 100
  numObjs = len(objectNumbers)
  width = 15

  fig, axes = plt.subplots(numModules, numObjs, figsize=(2*numObjs, 5))
  for i, obj in enumerate(objectNumbers):
    plotData = np.ones((numCells * numModules, numSteps*width, 3), dtype=np.float32)
    for step, modules in enumerate(locationLayerTimelineByObject[obj]):
      if step >= numSteps:
        continue
      for module in xrange(numModules):
        cells = [idx + (module * numCells) for idx in modules[module]]
        stepStart = step * width
        stepStop = (step + 1) * width
        plotData[cells, stepStart:stepStop, :] = [0, 0, 0]

    for m in xrange(numModules):
      if inferredStepByObject[obj] is not None:
        axes[m, i].add_patch(matplotlib.patches.Rectangle(
          ((inferredStepByObject[obj] - 1) * width, -1), width,
          numModules * numCells + 2, color="red", fill=False))
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

  filename = os.path.join(CHART_DIR, outFilename)
  print "Saving", filename
  plt.savefig(filename)


def aggregateChart(inFilename, outFilename, objectCounts):
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

  markers = ["*", "x", "o", "P"]

  plt.figure(figsize=(5,4.5))

  resultsByNumObjects = {}
  with open(inFilename, "r") as f:
    experiments = json.load(f)
  for exp in experiments:
    numObjects = exp[0]["numObjects"]
    resultsByNumObjects[numObjects] = exp[1]["locationLayerTimelineByObject"]

  for numObjects, marker in zip(objectCounts, markers):
    results = resultsByNumObjects[numObjects]

    timestepsByObject = [
      timesteps
      for k, timesteps in results.iteritems()]

    numCells = 100
    numSteps = 9

    x = np.arange(1, numSteps + 1)
    y = np.zeros((9), dtype="float")
    for iTimestep, timestepByObject in enumerate(zip(*timestepsByObject)):
      totalActive = 0
      potentialActive = 0
      for activeCellsByModule in timestepByObject:
        for activeCells in activeCellsByModule:
          totalActive += len(activeCells)
          potentialActive += numCells

      if iTimestep < numSteps:
        y[iTimestep] = totalActive / float(potentialActive)

    plt.plot(x, y, "{}-".format(marker), label="{} learned objects".format(numObjects))

  plt.xlabel("Number of sensations")
  plt.ylabel("Cell activation density")
  plt.ylim(0.0, 0.3)
  plt.legend()

  plt.tight_layout()

  filename = os.path.join(CHART_DIR, outFilename)
  print "Saving", filename
  plt.savefig(filename)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--inFile", type=str, required=True)
  parser.add_argument("--outFile1", type=str, required=True)
  parser.add_argument("--outFile2", type=str, required=True)
  parser.add_argument("--exampleObjectCount", type=int, default=100)
  parser.add_argument("--aggregateObjectCounts", type=int, nargs="+", default=[50, 75, 100, 125])
  parser.add_argument("--exampleObjectNumbers", type=int, nargs="+", default=-1)
  args = parser.parse_args()

  exampleObjectNumbers = (args.exampleObjectNumbers
                          if args.exampleObjectNumbers != -1 else
                          range(args.exampleObjectCount))

  examplesChart(args.inFile, args.outFile1, args.exampleObjectCount, exampleObjectNumbers)
  aggregateChart(args.inFile, args.outFile2, args.aggregateObjectCounts)
