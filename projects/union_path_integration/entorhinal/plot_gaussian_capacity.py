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

"""Plot capacity charts."""

from collections import defaultdict
import json
import math
import os
import itertools

import matplotlib.pyplot as plt
import matplotlib.lines
import numpy as np

from htmresearch.frameworks.location import ambiguity_index

CWD = os.path.dirname(os.path.realpath(__file__))
CHART_DIR = os.path.join(CWD, "charts")


def varyResolution_varyNumModules(inFilenames, outFilename,
                                  resolutions=(2, 3, 4),
                                  moduleCounts=(6, 12, 18,),
                                  maxNumObjectsByResolution={}):
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax2 = ax1.twiny()

  colors = ("C0", "C1", "C2")
  markers = ("o", "o", "o")
  markerSizes = (2, 4, 6)

  allResults = defaultdict(lambda: defaultdict(list))

  for inFilename in inFilenames:
    with open(inFilename, "r") as f:
      experiments = json.load(f)
    for exp in experiments:
      numModules = exp[0]["numModules"]
      numObjects = exp[0]["numObjects"]
      resolution = exp[0]["inverseReadoutResolution"]

      failed = exp[1].get("null", 0)
      allResults[(numModules, resolution)][numObjects].append(
        1.0 - (float(failed) / float(numObjects)))

  for resolution, color in zip(resolutions, colors):
    for numModules, marker, markerSize in zip(moduleCounts, markers, markerSizes):
      resultsByNumObjects = allResults[(numModules, resolution)]
      expResults = [(numObjects, sum(results) / len(results))
                    for numObjects, results in resultsByNumObjects.iteritems()
                    if resolution not in maxNumObjectsByResolution
                    or numObjects <= maxNumObjectsByResolution[resolution]]

      x = []
      y = []
      for i, j in sorted(expResults):
        x.append(i)
        y.append(j)

      ax1.plot(
        x, y, "{}-".format(marker), color=color, linewidth=1, markersize=markerSize
      )


  leg = ax1.legend(loc="upper right", title=" Readout bins per axis:",
                   frameon=False,
                   handles=[matplotlib.lines.Line2D([], [], color=color)
                            for color in colors],
                   labels=resolutions)
  ax1.add_artist(leg)

  leg = ax1.legend(loc="center right", title="Number of modules:",
                   bbox_to_anchor=(0.99, 0.6),
                   frameon=False,
                   handles=[matplotlib.lines.Line2D([], [],
                                                    marker=marker,
                                                    markersize=markerSize,
                                                    color="black")
                            for marker, markerSize in zip(markers, markerSizes)],
                   labels=moduleCounts)

  ax1.set_xlabel("# learned objects")
  ax1.set_ylabel("Recognition accuracy after many sensations")
  ax2.set_xlabel("Sensory ambiguity index", labelpad=8)

  locs, labels = ambiguity_index.getTotalExpectedOccurrencesTicks_2_5(
      ambiguity_index.numOtherOccurrencesOfMostUniqueFeature_lowerBound80_100features_10locationsPerObject)
  ax2.set_xticks(locs)
  ax2.set_xticklabels(labels)
  ax2.set_xlim(ax1.get_xlim())
  ax2_color = 'gray'
  ax2.xaxis.label.set_color(ax2_color)
  ax2.tick_params(axis='x', colors=ax2_color)

  plt.tight_layout()

  filePath = os.path.join(CHART_DIR, outFilename)
  print "Saving", filePath
  plt.savefig(filePath)



def varyModuleSize_varyResolution(inFilenames, outFilename,
                                  scalingFactors=[1, 2], resolutions=(2, 3, 4),
                                  maxNumObjectsByParams={}):
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

  allResults = defaultdict(lambda: defaultdict(list))

  for inFilename in inFilenames:
    with open(inFilename, "r") as f:
      experiments = json.load(f)
    for exp in experiments:
      enlargeModuleFactor = exp[0]["enlargeModuleFactor"]
      numObjects = exp[0]["numObjects"]
      resolution = exp[0]["inverseReadoutResolution"]

      failed = exp[1].get("null", 0)
      allResults[(enlargeModuleFactor, resolution)][numObjects].append(
        1.0 - (float(failed) / float(numObjects)))

  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax2 = ax1.twiny()

  colors = ("C0", "C1", "C2")
  markers = ("o", "o", "o")
  markerSizes = (2, 4, 6)
  for scalingFactor, color in zip(scalingFactors, colors):
    for resolution, marker, markerSize in zip(resolutions, markers, markerSizes):
      resultsByNumObjects = allResults[(scalingFactor, resolution)]
      expResults = [(numObjects, sum(results) / len(results))
                     for numObjects, results in resultsByNumObjects.iteritems()
                    if (scalingFactor, resolution) not in maxNumObjectsByParams
                    or numObjects <= maxNumObjectsByParams[(scalingFactor, resolution)]]

      x = []
      y = []
      for i, j in sorted(expResults):
        x.append(i)
        y.append(j)

      ax1.plot(
        x, y, "{}-".format(marker), color=color, linewidth=1, markersize=markerSize
      )


  # Carefully use whitespace in title to shift the entries in the legend to
  # align with the next legend.
  leg = ax1.legend(loc="upper right", title="Bump size:       ",
                   # bbox_to_anchor=(0.98, 1.0),
                   frameon=False,
                   handles=[matplotlib.lines.Line2D([], [], color=color)
                            for color in colors],
                   labels=["$ \\sigma = \\sigma_{rat} $",
                           "$ \\sigma = \\sigma_{rat} / 2.0 $"])

  ax1.add_artist(leg)


  leg = ax1.legend(loc="center right", title="Readout bins per axis:",
                   bbox_to_anchor=(1.0, 0.6),
                   frameon=False,
                   handles=[matplotlib.lines.Line2D([], [],
                                                    marker=marker,
                                                    markersize=markerSize,
                                                    color="black")
                            for marker, markerSize in zip(markers, markerSizes)],
                   labels=["$ \\frac{\\sigma_{rat}}{\\sigma} * 2 $ ",
                           "$ \\frac{\\sigma_{rat}}{\\sigma} * 3 $ ",
                           "$ \\frac{\\sigma_{rat}}{\\sigma} * 4 $ "])

  ax1.set_xlim(ax1.get_xlim()[0], 400)
  print ax1.get_xlim()

  ax1.set_xlabel("# learned objects")
  ax1.set_ylabel("Recognition accuracy after many sensations")
  ax2.set_xlabel("Sensory ambiguity index", labelpad=8)

  locs, labels = ambiguity_index.getTotalExpectedOccurrencesTicks_2_5(
      ambiguity_index.numOtherOccurrencesOfMostUniqueFeature_lowerBound80_100features_10locationsPerObject)
  ax2.set_xticks(locs)
  ax2.set_xticklabels(labels)
  ax2.set_xlim(ax1.get_xlim())
  ax2_color = 'gray'
  ax2.xaxis.label.set_color(ax2_color)
  ax2.tick_params(axis='x', colors=ax2_color)

  plt.tight_layout()

  filePath = os.path.join(CHART_DIR, outFilename)
  print "Saving", filePath
  plt.savefig(filePath)



def varyModuleSize_varyNumModules(inFilenames, outFilename,
                                  scalingFactors=[1, 2, 3],
                                  moduleCounts=(6,12,18),
                                  maxNumObjectsByParams={}):
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

  allResults = defaultdict(lambda: defaultdict(list))

  for inFilename in inFilenames:
    with open(inFilename, "r") as f:
      experiments = json.load(f)
    for exp in experiments:
      enlargeModuleFactor = exp[0]["enlargeModuleFactor"]
      numObjects = exp[0]["numObjects"]
      numModules = exp[0]["numModules"]

      failed = exp[1].get("null", 0)
      allResults[(enlargeModuleFactor, numModules)][numObjects].append(
        1.0 - (float(failed) / float(numObjects)))

  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax2 = ax1.twiny()
  # Optional: swap axes
  # ax1.xaxis.tick_top()
  # ax1.xaxis.set_label_position('top')
  # ax2.xaxis.tick_bottom()
  # ax2.xaxis.set_label_position('bottom')


  colors = ("C0", "C1", "C2")
  markers = ("o", "o", "o")
  markerSizes = (2, 4, 6)
  for scalingFactor, color in zip(scalingFactors, colors):
    for numModules, marker, markerSize in zip(moduleCounts, markers, markerSizes):
      resultsByNumObjects = allResults[(scalingFactor, numModules)]
      expResults = [(numObjects, sum(results) / len(results))
                     for numObjects, results in resultsByNumObjects.iteritems()
                    if (scalingFactor, numModules) not in maxNumObjectsByParams
                    or numObjects <= maxNumObjectsByParams[(scalingFactor, numModules)]]

      x = []
      y = []
      for i, j in sorted(expResults):
        x.append(i)
        y.append(j)

      ax1.plot(
        x, y, "{}-".format(marker), color=color, linewidth=1, markersize=markerSize
      )

  # Carefully use whitespace in title to shift the entries in the legend to
  # align with the next legend.
  leg = ax1.legend(loc="upper right", title="Module size:       ",
                   frameon=False,
                   handles=[matplotlib.lines.Line2D([], [], color=color)
                            for color in colors],
                   labels=["rat", "rat * 2", "rat * 3"])
  ax1.add_artist(leg)


  leg = ax1.legend(loc="center right", title="Number of modules:",
                   bbox_to_anchor=(1.0, 0.6),
                   frameon=False,
                   handles=[matplotlib.lines.Line2D([], [],
                                                    marker=marker,
                                                    markersize=markerSize,
                                                    color="black")
                            for marker, markerSize in zip(markers, markerSizes)],
                   labels=moduleCounts)

  ax1.set_xlabel("# learned objects")
  ax1.set_ylabel("Recognition accuracy after many sensations")
  ax2.set_xlabel("Sensory ambiguity index", labelpad=8)

  locs, labels = ambiguity_index.getTotalExpectedOccurrencesTicks_2_5(
      ambiguity_index.numOtherOccurrencesOfMostUniqueFeature_lowerBound80_100features_10locationsPerObject)

  ax2.set_xticks(locs)
  ax2.set_xticklabels(labels)
  ax2.set_xlim(ax1.get_xlim())
  ax2_color = 'gray'
  ax2.xaxis.label.set_color(ax2_color)
  ax2.tick_params(axis='x', colors=ax2_color)

  plt.tight_layout()

  filePath = os.path.join(CHART_DIR, outFilename)
  print "Saving", filePath
  plt.savefig(filePath)



if __name__ == "__main__":
  varyResolution_varyNumModules(
    ["results/gaussian_varyNumModules_100_feats_2_resolution.json",
     "results/gaussian_varyNumModules_100_feats_3_resolution.json",
     "results/gaussian_varyNumModules_100_feats_4_resolution.json"],
    "capacity100_gaussian_varyResolution_varyNumModules.pdf",
    maxNumObjectsByResolution={
      4: 160,
      3: 130,
    }
  )

  # varyModuleSize_varyResolution(
  #   [],
  #   "capacity100_gaussian_varyModuleSize_varyResolution.pdf",
  #   maxNumObjectsByParams={
  #     (1.0, 4): 150,
  #   }
  # )

  varyModuleSize_varyNumModules(
    ["results/varyModuleSize_100_feats_1_enlarge.json",
     "results/varyModuleSize_100_feats_2_enlarge.json",
     "results/varyModuleSize_100_feats_3_enlarge.json"],
    "capacity100_gaussian_varyModuleSize_varyNumModules.pdf",
    maxNumObjectsByParams={
      # (1.0, 4): 150,
    }
  )
