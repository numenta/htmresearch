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

"""Plot capacity trend charts."""

import argparse
from collections import defaultdict
import json
import math
import os
import itertools

import matplotlib.pyplot as plt
import matplotlib.lines
import numpy as np

import ambiguity_index

CWD = os.path.dirname(os.path.realpath(__file__))
CHART_DIR = os.path.join(CWD, "charts")



def createChart(inFilename, outFilename):
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)


  capacitiesByParams = defaultdict(list)
  moduleCounts = set()
  allCellCounts = set()
  allFeatureCounts = set()
  with open(inFilename, "r") as f:
    experiments = json.load(f)
  for exp in experiments:
    numModules = exp[0]["numModules"]
    thresholds = exp[0]["thresholds"]
    locationModuleWidth = exp[0]["locationModuleWidth"]
    numUniqueFeatures = exp[0]["numFeatures"]

    cellsPerModule = locationModuleWidth*locationModuleWidth

    moduleCounts.add(numModules)
    allCellCounts.add(cellsPerModule)
    allFeatureCounts.add(numUniqueFeatures)

    params = (numModules, cellsPerModule, thresholds, numUniqueFeatures)
    capacitiesByParams[params].append(exp[1]["numObjects"])

  moduleCounts = sorted(moduleCounts)
  allCellCounts = sorted(allCellCounts)
  allFeatureCounts = sorted(allFeatureCounts)

  meanCapacityByParams = {}
  for params, capacities in capacitiesByParams.iteritems():
    meanCapacityByParams[params] = sum(capacities) / float(len(capacities))


  fig, (ax1, ax2, ax3) = plt.subplots(figsize=(6,2.2), ncols=3)

  #
  # NUMBER OF MODULES
  #
  cellsPerModule = 100
  numUniqueFeatures = 100
  markers = ["o", "D"]
  markerSizes = [4.0, 4.0]
  for thresholds, marker, markerSize in zip([-1, 0], markers, markerSizes):
    ax1.plot(moduleCounts, [meanCapacityByParams[(numModules,
                                                  cellsPerModule,
                                                  thresholds,
                                                  numUniqueFeatures)]
                            for numModules in moduleCounts],
             "{}-".format(marker), color="C0", markersize=markerSize)

  ax1.text(1, 685, "Threshold:")
  ax1.text(32, 590, "$ n $")
  ax1.text(23, 200, "$ \\lceil n * 0.8 \\rceil $")

  ax1.set_xlabel("Number of\nModules", fontsize=12)
  ax1.set_ylabel("Capacity", fontsize=12)
  ax1.set_xlim(0, ax1.get_xlim()[1])
  ax1.set_ylim(0, ax1.get_ylim()[1])
  xticks = [0] + moduleCounts
  ax1.set_xticks(xticks)
  ax1.set_xticklabels([(x if x % 10 == 0 else "")
                       for x in xticks])

  #
  # CELLS PER MODULE
  #
  numModules = 10
  thresholds = -1
  numUniqueFeatures = 100
  ax2.plot(allCellCounts, [meanCapacityByParams[(numModules,
                                                 cellsPerModule,
                                                 thresholds,
                                                 numUniqueFeatures)]
                           for cellsPerModule in allCellCounts],
           "o-", color="C0", markersize=4.0)

  ax2.set_xlabel("Cells Per Module", fontsize=12)
  ax2.set_xlim(0, ax2.get_xlim()[1])
  ax2.set_ylim(0, ax2.get_ylim()[1])


  #
  # NUMBER OF UNIQUE FEATURES
  #
  numModules = 10
  cellsPerModule = 100
  thresholds = -1
  ax3.plot(allFeatureCounts, [meanCapacityByParams[(numModules,
                                                    cellsPerModule,
                                                    thresholds,
                                                    numUniqueFeatures)]
                              for numUniqueFeatures in allFeatureCounts],
           "o-", color="C1", markersize=4.0)

  ax3.set_xlabel("Number of\nUnique Features", fontsize=12)
  ax3.set_xlim(0, ax3.get_xlim()[1])
  ax3.set_ylim(0, ax3.get_ylim()[1])


  plt.tight_layout()

  filePath = os.path.join(CHART_DIR, outFilename)
  print "Saving", filePath
  plt.savefig(filePath)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--inFile", type=str, required=True)
  parser.add_argument("--outFile", type=str, required=True)
  args = parser.parse_args()

  createChart(args.inFile, args.outFile)
