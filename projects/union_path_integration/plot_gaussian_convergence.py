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

"""Plot recognition time charts."""

from collections import defaultdict
import json
import os

import ambiguity_index

import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np

CWD = os.path.dirname(os.path.realpath(__file__))
CHART_DIR = os.path.join(CWD, "charts")



def varyResolution_varyNumModules(inFilenames, outFilename,
                                  resolutions=(2, 3, 4),
                                  moduleCounts=(6, 12, 18,),
                                  xlim=None):
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

  allResults = defaultdict(lambda: defaultdict(list))

  for inFilename in inFilenames:
    with open(inFilename, "r") as f:
      experiments = json.load(f)
    for exp in experiments:
      numModules = exp[0]["numModules"]
      numObjects = exp[0]["numObjects"]
      resolution = exp[0]["inverseReadoutResolution"]

      results = []
      for numSensationsStr, numOccurrences in exp[1].items():
        if numSensationsStr == "null":
          results += [np.inf] * numOccurrences
        else:
          results += [int(numSensationsStr)] * numOccurrences

      allResults[(numModules, resolution)][numObjects] += results

  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax2 = ax1.twiny()

  colors = ("C0", "C1", "C2")
  markers = ("o", "o", "o")
  markerSizes = (2, 4, 6)
  for resolution, color in zip(resolutions, colors):
    for numModules, marker, markerSize in zip(moduleCounts, markers, markerSizes):
      resultsByNumObjects = allResults[(numModules, resolution)]

      expResults = sorted((numObjects, np.median(results))
                          for numObjects, results in resultsByNumObjects.iteritems())

      # Results up to the final non-infinite median.
      lineResults = [(numObjects, median)
                     for numObjects, median in expResults
                     if median != np.inf]

      # Results excluding the final non-infinite median.
      numCircleMarkers = len(lineResults)
      if len(lineResults) < len(expResults):
        numCircleMarkers -= 1

      # Results including only the final non-infinite median.
      lineEndResults = ([lineResults[-1]] if len(lineResults) < len(expResults)
                        else [])

      ax1.plot([numObjects for numObjects, median in lineResults],
               [median for numObjects, median in lineResults],
               "{}-".format(marker), markevery=xrange(numCircleMarkers),
               color=color, linewidth=1, markersize=markerSize)
      if len(lineResults) < len(expResults):
        endNumObjects, endMedian = lineEndResults[-1]
        ax1.plot([endNumObjects], [endMedian], "x", color=color,
                 markeredgewidth=markerSize/2, markersize=markerSize*1.5)

  if xlim is not None:
    ax1.set_xlim(xlim[0], xlim[1])
  ax1.set_ylim(0, ax1.get_ylim()[1])
  ax1.set_xlabel("# learned objects")
  ax1.set_ylabel("Median # sensations before recognition")
  ax2.set_xlabel("Sensory ambiguity index", labelpad=8)

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
                                  enlargeModuleFactors=[1.0, 2.0],
                                  resolutions=(2, 3, 4),
                                  xlim=None):
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

      results = []
      for numSensationsStr, numOccurrences in exp[1].items():
        if numSensationsStr == "null":
          results += [np.inf] * numOccurrences
        else:
          results += [int(numSensationsStr)] * numOccurrences

      allResults[(enlargeModuleFactor, resolution)][numObjects] += results

  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax2 = ax1.twiny()

  colors = ("C0", "C1", "C2")
  markers = ("o", "o", "o")
  markerSizes = (2, 4, 6)
  for resolution, marker, markerSize in zip(resolutions, markers, markerSizes):
    for enlargeModuleFactor, color in zip(enlargeModuleFactors, colors):
      resultsByNumObjects = allResults[(enlargeModuleFactor, resolution)]

      expResults = sorted((numObjects, np.median(results))
                          for numObjects, results in resultsByNumObjects.iteritems())

      # Results up to the final non-infinite median.
      lineResults = [(numObjects, median)
                     for numObjects, median in expResults
                     if median != np.inf]

      # Results excluding the final non-infinite median.
      numCircleMarkers = len(lineResults)
      if len(lineResults) < len(expResults):
        numCircleMarkers -= 1

      # Results including only the final non-infinite median.
      lineEndResults = ([lineResults[-1]] if len(lineResults) < len(expResults)
                        else [])

      ax1.plot([numObjects for numObjects, median in lineResults],
               [median for numObjects, median in lineResults],
               "{}-".format(marker), markevery=xrange(numCircleMarkers),
               color=color, linewidth=1, markersize=markerSize)
      if len(lineResults) < len(expResults):
        endNumObjects, endMedian = lineEndResults[-1]
        ax1.plot([endNumObjects], [endMedian], "x", color=color,
                 markeredgewidth=markerSize/2, markersize=markerSize*1.5)

  if xlim is not None:
    ax1.set_xlim(xlim[0], xlim[1])
  ax1.set_ylim(0, ax1.get_ylim()[1])
  ax1.set_xlabel("# learned objects")
  ax1.set_ylabel("Median # sensations before recognition")
  ax2.set_xlabel("Sensory ambiguity index", labelpad=8)

  # Carefully use whitespace in title to shift the entries in the legend to
  # align with the previous legend.
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
                                  enlargeModuleFactors=[1.0, 2.0, 3.0],
                                  moduleCounts=(6, 12, 18),
                                  xlim=None):
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

      results = []
      for numSensationsStr, numOccurrences in exp[1].items():
        if numSensationsStr == "null":
          results += [np.inf] * numOccurrences
        else:
          results += [int(numSensationsStr)] * numOccurrences

      allResults[(enlargeModuleFactor, numModules)][numObjects] += results

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
  for numModules, marker, markerSize in zip(moduleCounts, markers, markerSizes):
    for enlargeModuleFactor, color in zip(enlargeModuleFactors, colors):
      resultsByNumObjects = allResults[(enlargeModuleFactor, numModules)]

      expResults = sorted((numObjects, np.median(results))
                          for numObjects, results in resultsByNumObjects.iteritems())

      # Results up to the final non-infinite median.
      lineResults = [(numObjects, median)
                     for numObjects, median in expResults
                     if median != np.inf]

      # Results excluding the final non-infinite median.
      numCircleMarkers = len(lineResults)
      if len(lineResults) < len(expResults):
        numCircleMarkers -= 1

      # Results including only the final non-infinite median.
      lineEndResults = ([lineResults[-1]] if len(lineResults) < len(expResults)
                        else [])

      ax1.plot([numObjects for numObjects, median in lineResults],
               [median for numObjects, median in lineResults],
               "{}-".format(marker), markevery=xrange(numCircleMarkers),
               color=color, linewidth=1, markersize=markerSize)
      if len(lineResults) < len(expResults):
        endNumObjects, endMedian = lineEndResults[-1]
        ax1.plot([endNumObjects], [endMedian], "x", color=color,
                 markeredgewidth=markerSize/2, markersize=markerSize*1.5)

  if xlim is not None:
    ax1.set_xlim(xlim[0], xlim[1])
  ax1.set_ylim(0, ax1.get_ylim()[1])
  ax1.set_xlabel("# learned objects")
  ax1.set_ylabel("Median # sensations before recognition")
  ax2.set_xlabel("Sensory ambiguity index", labelpad=8)

  # Carefully use whitespace in title to shift the entries in the legend to
  # align with the previous legend.
  leg = ax1.legend(loc="upper right", title="Module size:       ",
                   # bbox_to_anchor=(0.98, 1.0),
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
    "convergence100_gaussian_varyResolution_varyNumModules.pdf",
    xlim=(2.5, 167.5))

  # varyModuleSize_varyResolution(
  #   [],
  #   "recognition_time_varyModuleSize.pdf",
  #   xlim=(-6.0, 400.0)
  # )

  varyModuleSize_varyNumModules(
    ["results/varyModuleSize_100_feats_1_enlarge.json",
     "results/varyModuleSize_100_feats_2_enlarge.json",
     "results/varyModuleSize_100_feats_3_enlarge.json"],
    "convergence100_gaussian_varyModuleSize_varyNumModules.pdf",
    xlim=(0, 577.0)
  )
