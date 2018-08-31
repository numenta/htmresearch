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

import scipy.special

CWD = os.path.dirname(os.path.realpath(__file__))
CHART_DIR = os.path.join(CWD, "charts")


DETAILED_LABELS = False



def createTheChart(inFilename, outFilename1, outFilename2, xlim2):
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)


  convergenceResultsByParams = defaultdict(lambda: defaultdict(list))
  capacityResults = defaultdict(lambda: defaultdict(list))

  with open(inFilename, "r") as f:
    experiments = json.load(f)
  for exp in experiments:
    featureDistribution = exp[0]["featureDistribution"]
    featuresPerObject = exp[0]["featuresPerObject"]
    numObjects = exp[0]["numObjects"]
    numUniqueFeatures = exp[0]["numFeatures"]

    failed = exp[1]["convergence"].get("null", 0)
    accuracy = 1.0 - (float(failed) / float(numObjects))
    capacityResults[(featureDistribution, numUniqueFeatures, featuresPerObject)][numObjects].append(accuracy)

    for occurrences, result in exp[1]["occurrencesConvergenceLog"]:
      # k = np.median(occurrences)
      k = min(occurrences)
      convergenceResultsByParams[
        (featureDistribution, numUniqueFeatures, featuresPerObject)][
          k].append(result)


  resultsByParams = {}
  for params, convergenceResultsByMin in convergenceResultsByParams.iteritems():
    resultsByParams[params] = sorted(
      (sampleMinimum, float(sum(1 if r != None else 0
                                for r in results)) / len(results))
      for sampleMinimum, results in convergenceResultsByMin.iteritems())

  fig1, ax1 = plt.subplots(figsize=(6.0, 4.8))
  fig2, ax2 = plt.subplots(figsize=(6.0, 4.8))

  objectSets = [
      ("AllFeaturesEqual_Replacement", 100, 10, "o", 3),
      ("AllFeaturesEqual_Replacement", 40, 10, "o", 3),
      ("AllFeaturesEqual_Replacement", 100, 5, "o", 3),
      ("AllFeaturesEqual_NoReplacement", 100, 10, "^", 4),
      ("TwoPools_Replacement", 100, 10, "^", 4),
      ("TwoPools_Structured", 100, 10, "^", 4),
  ]

  for i, (featureDistribution, numUniqueFeatures, featuresPerObject, marker, markerSize) in enumerate(objectSets):
    if DETAILED_LABELS:
      if featureDistribution == "AllFeaturesEqual_Replacement":
        label = "{}, {} features, {} per object".format(featureDistribution, numUniqueFeatures, featuresPerObject)
      else:
        label = featureDistribution
    else:
      label = "Object set {}".format(i + 1)

    resultsByNumObjects = capacityResults[(featureDistribution, numUniqueFeatures, featuresPerObject)]
    expResults = sorted((numObjects, sum(results) / len(results))
                        for numObjects, results in resultsByNumObjects.iteritems())
    x, y = zip(*expResults)
    ax1.plot(x, y, "{}-".format(marker), label=label, markersize=markerSize)

    results = resultsByParams[(featureDistribution, numUniqueFeatures, featuresPerObject)]
    x, y = zip(*results)
    ax2.plot(x, y, "{}-".format(marker), label=label, markersize=markerSize)

  ax1.set_xlabel("Number of Learned Objects", fontsize=12)
  ax1.set_ylabel("Recognition Accuracy\nAfter Many Sensations", fontsize=12)
  ax2.set_xlabel("Number of Locations Recalled by\nObject's Rarest Feature", fontsize=12)
  ax2.set_ylabel("Recognition Accuracy\nAfter Many Sensations", fontsize=12)
  if xlim2 is not None:
    ax2.set_xlim([0, xlim2])

  fig1.tight_layout()
  fig2.tight_layout()
  ax1.legend(loc="upper right", fontsize=10)
  ax2.legend(loc="upper right", fontsize=10)

  filePath1 = os.path.join(CHART_DIR, outFilename1)
  print "Saving", filePath1
  fig1.savefig(filePath1)

  filePath2 = os.path.join(CHART_DIR, outFilename2)
  print "Saving", filePath2
  fig2.savefig(filePath2)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--inFile", type=str, required=True)
  parser.add_argument("--outFile1", type=str, required=True)
  parser.add_argument("--outFile2", type=str, required=True)
  parser.add_argument("--xlim2", type=float, default=None)
  args = parser.parse_args()

  createTheChart(args.inFile, args.outFile1, args.outFile2, args.xlim2)
