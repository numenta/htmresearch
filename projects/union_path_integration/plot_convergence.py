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

"""Plot convergence chart."""

import collections
import json
import os

import matplotlib.pyplot as plt
import numpy as np

CWD = os.path.dirname(os.path.realpath(__file__))
CHART_DIR = os.path.join(CWD, "charts")


def chart():
  if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

  # Convergence vs. number of objects, comparing # unique features

  plt.style.use("ggplot")

  for feats in (50, 100, 5000):
    with open("results/convergence_vs_num_objs_{}_feats.json".format(feats), "r") as f:
      convVsObjects = json.load(f)

    yData = collections.defaultdict(list)
    for exp in convVsObjects:
      numObjects = int(str(exp[0]["numObjects"]))
      if "null" in exp[1].keys():
        continue
      results = exp[1].items()
      total = 0
      count = 0
      for i, j in results:
        total += (int(str(i)) * j)
        count += j
      y = float(total) / float(count)
      yData[numObjects].append(y)

    x = list(sorted(yData.keys()))
    yData = sorted(yData.iteritems())
    y = [float(sum(pair[1])) / float(len(pair[1]))
         if None not in pair[1] else None
         for pair in yData]
    std = [np.std(pair[1])
           for pair in yData]
    yBelow = [yi - stdi
              for yi, stdi in zip(y, std)]
    yAbove = [yi + stdi
              for yi, stdi in zip(y, std)]
    xError = x[:len(yBelow)]

    plt.plot(
        x, y, "-", label="{} unique features".format(feats),
    )
    plt.fill_between(xError, yBelow, yAbove, alpha=0.3)

  plt.xlabel("Number of Objects")
  plt.ylabel("Number of Sensations")
  plt.legend(loc="center right")

  plt.savefig(os.path.join(CHART_DIR, "convergence_vs_objects_w_feats.pdf"))

  plt.clf()

  # Convergence vs. number of objects, varying module size

  plt.style.use("ggplot")

  for cpm in (25, 100, 400):
    with open("results/convergence_vs_num_objs_{}_cpm.json".format(cpm), "r") as f:
      convVsObjs = json.load(f)

    yData = collections.defaultdict(list)
    for exp in convVsObjs:
      results = exp[1].items()
      total = 0
      count = 0
      for i, j in results:
        total += (int(str(i)) * j)
        count += j
      y = float(total) / float(count)
      numObjects = int(str(exp[0]["numObjects"]))
      yData[numObjects].append(y)

    x = list(sorted(yData.keys()))
    yData = sorted(yData.iteritems())
    y = [float(sum(pair[1])) / float(len(pair[1])) for pair in yData]
    std = [np.std(pair[1]) for pair in yData]
    yBelow = [yi - stdi for yi, stdi in zip(y, std)]
    yAbove = [yi + stdi for yi, stdi in zip(y, std)]

    plt.plot(
        x, y, "-", label="{} cells per module".format(cpm),
    )
    plt.fill_between(x, yBelow, yAbove, alpha=0.3)

  plt.xlabel("Number of Objects")
  plt.ylabel("Number of Sensations")
  plt.legend(loc="upper left")

  plt.savefig(os.path.join(CHART_DIR, "convergence_vs_objects.pdf"))

  plt.clf()

  # Convergence vs. number of modules

  plt.style.use("ggplot")

  for cpm in (25, 100, 400):
    with open("results/convergence_vs_num_modules_100_feats_{}_cpm.json".format(cpm), "r") as f:
      convVsMods100 = json.load(f)

    yData = collections.defaultdict(list)
    for exp in convVsMods100:
      results = exp[1].items()
      total = 0
      count = 0
      for i, j in results:
        if str(i) == "null":
          total = 50 * j
        else:
          total += (int(str(i)) * j)
        count += j
      y = float(total) / float(count)
      numModules = int(str(exp[0]["numModules"]))
      yData[numModules].append(y)

    yData = sorted(yData.iteritems())
    x = [i+1 for i in xrange(10)]
    y = [float(sum(pair[1])) / float(len(pair[1])) for pair in yData]
    std = [np.std(pair[1]) for pair in yData]
    yBelow = [yi - stdi for yi, stdi in zip(y, std)]
    yAbove = [yi + stdi for yi, stdi in zip(y, std)]

    plt.plot(
        x, y, "-", label="{} cells per module".format(cpm),
    )
    plt.fill_between(x, yBelow, yAbove, alpha=0.3)

  plt.xlabel("Number of Modules")
  plt.ylabel("Convergence")
  plt.legend(loc="upper right")

  plt.savefig(os.path.join(CHART_DIR, "convergence_vs_modules_100_feats.pdf"))

  plt.clf()


if __name__ == "__main__":
  chart()
