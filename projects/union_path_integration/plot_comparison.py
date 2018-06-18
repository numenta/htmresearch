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

"""Plot comparison chart."""

import collections
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

CWD = os.path.dirname(os.path.realpath(__file__))
CHART_DIR = os.path.join(CWD, "charts")


def computeCapacity(results, threshold):
  """Returns largest number of objects with accuracy above threshold."""
  closestBelow = None
  closestAbove = None
  for numObjects, accuracy in results:
    if accuracy >= threshold:
      if closestAbove is None or accuracy < closestAbove[1]:
        closestAbove = (numObjects, accuracy)
    else:
      if closestBelow is None or accuracy > closestBelow[1]:
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

  # Convergence vs. number of objects, comparing # unique features

  #plt.style.use("ggplot")

  # Generate with:
  # Todo: narrow down the actual number of objects so these can more efficiently be run.
  # python convergence_simulation.py --numObjects 4000 4500 5000 --numUniqueFeatures 200 --locationModuleWidth 14 --resultName results/convergence_vs_num_objs_196_cpm_200_feats_2.json
  #python convergence_simulation.py --numObjects 3000 3500 4000 --numUniqueFeatures 200 --locationModuleWidth 12 --resultName results/convergence_vs_num_objs_144_cpm_200_feats_2.json
  # python convergence_simulation.py --numObjects 2500 3000 3500 --numUniqueFeatures 200 --locationModuleWidth 10 --resultName results/convergence_vs_num_objs_100_cpm_200_feats_2.json
  # python convergence_simulation.py --numObjects 1500 2000 2500 --numUniqueFeatures 200 --locationModuleWidth 7 --resultName results/convergence_vs_num_objs_49_cpm_200_feats_2.json
  fnames = (
      "results/convergence_vs_num_objs_196_cpm_200_feats.json",
      "results/convergence_vs_num_objs_196_cpm_200_feats_2.json",
      "results/convergence_vs_num_objs_144_cpm_200_feats.json",
      "results/convergence_vs_num_objs_144_cpm_200_feats_2.json",
      "results/convergence_vs_num_objs_100_cpm_200_feats.json",
      "results/convergence_vs_num_objs_100_cpm_200_feats_2.json",
      "results/convergence_vs_num_objs_49_cpm_200_feats.json",
      "results/convergence_vs_num_objs_49_cpm_200_feats_2.json",

      "results/convergence_vs_num_objs_196_cpm_150_feats.json",
      "results/convergence_vs_num_objs_144_cpm_150_feats.json",
      "results/convergence_vs_num_objs_100_cpm_150_feats.json",
      "results/convergence_vs_num_objs_49_cpm_150_feats.json",
      "results/convergence_vs_num_objs_49_cpm_150_feats_2.json",

      "results/convergence_vs_num_objs_196_cpm_100_feats.json",
      "results/convergence_vs_num_objs_144_cpm_100_feats.json",
      "results/convergence_vs_num_objs_100_cpm_100_feats.json",
      "results/convergence_vs_num_objs_100_cpm_100_feats_2.json",
      "results/convergence_vs_num_objs_100_cpm_100_feats_3.json",
      "results/convergence_vs_num_objs_49_cpm_100_feats.json",
      "results/convergence_vs_num_objs_49_cpm_100_feats_2.json",

      "results/convergence_vs_num_objs_196_cpm_50_feats.json",
      "results/convergence_vs_num_objs_144_cpm_50_feats.json",
      "results/convergence_vs_num_objs_144_cpm_50_feats_3.json",
      "results/convergence_vs_num_objs_100_cpm_50_feats.json",
      "results/convergence_vs_num_objs_100_cpm_50_feats_2.json",
      "results/convergence_vs_num_objs_49_cpm_50_feats.json",

      "results/convergence_vs_num_objs_196_cpm_175_feats.json",
      "results/convergence_vs_num_objs_196_cpm_125_feats.json",
      "results/convergence_vs_num_objs_196_cpm_75_feats.json",
      "results/convergence_vs_num_objs_196_cpm_25_feats.json",
      "results/convergence_vs_num_objs_196_cpm_25_feats_2.json",

      "results/convergence_vs_num_objs_144_cpm_175_feats.json",
      "results/convergence_vs_num_objs_144_cpm_125_feats.json",
      "results/convergence_vs_num_objs_144_cpm_75_feats.json",
      "results/convergence_vs_num_objs_144_cpm_25_feats.json",
      "results/convergence_vs_num_objs_144_cpm_125_feats_2.json",

      "results/convergence_vs_num_objs_100_cpm_175_feats.json",
      "results/convergence_vs_num_objs_100_cpm_175_feats_2.json",
      "results/convergence_vs_num_objs_100_cpm_125_feats.json",
      "results/convergence_vs_num_objs_100_cpm_75_feats.json",
      "results/convergence_vs_num_objs_100_cpm_25_feats.json",

      "results/convergence_vs_num_objs_49_cpm_175_feats.json",
      "results/convergence_vs_num_objs_49_cpm_125_feats.json",
      "results/convergence_vs_num_objs_49_cpm_75_feats.json",
      "results/convergence_vs_num_objs_49_cpm_25_feats.json",
  )

  # (modSize, numFeats) -> [(numObjs, accuracy), ...]
  capacities = collections.defaultdict(list)
  for fname in fnames:
    with open(fname, "r") as f:
      data = json.load(f)

    for exp in data:
      modSize = np.prod(exp[0]["cellDimensions"])
      numFeatures = exp[0]["numFeatures"]
      k = (modSize, numFeatures)
      numObjects = exp[0]["numObjects"]

      failed = exp[1].get("null", 0)
      accuracy = (float(numObjects) - float(failed)) / float(numObjects)
      capacities[k].append((numObjects, accuracy))

  plotData = np.zeros((4, 8))
  modSizeMap = (49, 100, 144, 196)
  numFeaturesMap = (25, 50, 75, 100, 125, 150, 175, 200)
  #numFeaturesMap = (50, 100, 150, 200)
  scatterX = []
  scatterY = []
  scatterLabels = []
  for modSizeI in xrange(4):
    for numFeaturesI in xrange(8):
      modSize = modSizeMap[modSizeI]
      numFeatures = numFeaturesMap[numFeaturesI]
      print modSize, numFeatures
      capacity90 = computeCapacity(capacities[(modSize, numFeatures)], 0.9)
      plotData[modSizeI][numFeaturesI] = capacity90

      scatterX.append(modSize * numFeatures)
      scatterY.append(capacity90)
      scatterLabels.append("({}, {})".format(modSize, numFeatures))

  # Fit data
  def func(x, a, b):
    return (x * a) + b
  popt, pcov = scipy.optimize.curve_fit(func, np.array(scatterX), scatterY)
  a, b = popt
  perr = np.sqrt(np.diag(pcov))
  print "Fitted data to linear model"
  print "  Best fit values:", popt
  print "  Covariance matrix:", pcov
  print "  1 stdev:", perr

  xlabels = [str(v) for v in numFeaturesMap]
  ylabels = [str(v) for v in modSizeMap]

  fig, ax = plt.subplots()
  plt.imshow(plotData)

  ax.invert_yaxis()
  ax.set_xticks(np.arange(len(xlabels)))
  ax.set_yticks(np.arange(len(ylabels)))
  ax.set_xticklabels(xlabels)
  ax.set_yticklabels(ylabels)
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor")

  for i in xrange(len(ylabels)):
    for j in xrange(len(xlabels)):
      text = ax.text(j, i, str(int(plotData[i, j])), ha="center", va="center", color="w")

  plt.xlabel("Number of Unique Features")
  plt.ylabel("Cells Per Module")

  fig.tight_layout()

  plt.savefig(os.path.join(CHART_DIR, "comparison_modsize_vs_feats.pdf"))

  plt.clf()

  # Module size vs. # features comparison scatter plot

  plt.plot([0, 40000], [b, (a * 40000) + b])
  aBelow = a - perr[0]
  aAbove = a + perr[0]
  bBelow = b - perr[1]
  bAbove = b + perr[1]
  yBelow = [bBelow, (aBelow * 40000) + bBelow]
  yAbove = [bAbove, (aAbove * 40000) + bAbove]
  plt.fill_between([0, 40000], yBelow, yAbove, alpha=0.3)
  plt.scatter(scatterX, scatterY)

  #plt.figure(figsize=(4, 3))

  plt.xlabel("Cells Per Module x Number of Unique Features")
  plt.ylabel("Capacity")
  fig.tight_layout()

  plt.savefig(os.path.join(CHART_DIR, "comparison_with_fit.pdf"))

  plt.clf()


if __name__ == "__main__":
  chart()
