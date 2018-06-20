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
  #
  # Generated with:
  #   python convergence_simulation.py --numObjects 200 400 600 800 1000 1200 1400 1600 1800 2000 --numUniqueFeatures 50 --locationModuleWidth 20 --resultName results/convergence_vs_num_objs_50_feats.json
  #   python convergence_simulation.py --numObjects 200 400 600 800 1000 1200 1400 1600 1800 2000 --numUniqueFeatures 100 --locationModuleWidth 20 --resultName results/convergence_vs_num_objs_100_feats.json
  #   python convergence_simulation.py --numObjects 200 400 600 800 1000 1200 1400 1600 1800 2000 --numUniqueFeatures 5000 --locationModuleWidth 20 --resultName results/convergence_vs_num_objs_5000_feats.json

  #plt.style.use("ggplot")

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
  plt.ylabel("Average Number of Sensations")
  plt.legend(loc="center right")

  plt.savefig(os.path.join(CHART_DIR, "convergence_vs_objects_w_feats.pdf"))

  plt.clf()

  # Convergence vs. number of objects, varying module size
  # NOT USED in Columns Plus
  #
  # Generated with:
  # TODO

  #plt.style.use("ggplot")

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
        x, y, "o-", label="{} cells per module".format(cpm),
    )
    plt.fill_between(x, yBelow, yAbove, alpha=0.3)

  plt.xlabel("Number of Objects")
  plt.ylabel("Average Number of Sensations")
  plt.legend(loc="upper left")

  plt.savefig(os.path.join(CHART_DIR, "convergence_with_modsize.pdf"))

  plt.clf()

  # Convergence vs. number of modules
  #
  # Generated with:
  #   python convergence_simulation.py --numObjects 100 --numUniqueFeatures 100 --locationModuleWidth 5 --numModules 1 2 3 4 5 6 7 8 9 10 --resultName results/convergence_vs_num_modules_100_feats_25_cpm.json --repeat 10
  #   python convergence_simulation.py --numObjects 100 --numUniqueFeatures 100 --locationModuleWidth 10 --numModules 1 2 3 4 5 6 7 8 9 10 --resultName results/convergence_vs_num_modules_100_feats_100_cpm.json --repeat 10
  #   python convergence_simulation.py --numObjects 100 --numUniqueFeatures 100 --locationModuleWidth 20 --numModules 1 2 3 4 5 6 7 8 9 10 --resultName results/convergence_vs_num_modules_100_feats_400_cpm.json --repeat 10

  #plt.style.use("ggplot")

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

    x = [i+1 for i in xrange(10)]
    #y = [float(sum(pair[1])) / float(len(pair[1])) for pair in yData]
    y = [float(sum(yData[step])) / float(len(yData[step])) for step in x]
    #yData20 = yData[19][1]
    #y20 = float(sum(yData20)) / float(len(yData20))

    yData = sorted(yData.iteritems())
    std = [np.std(pair[1]) for pair in yData]
    yBelow = [yi - stdi for yi, stdi in zip(y, std)]
    yAbove = [yi + stdi for yi, stdi in zip(y, std)]

    plt.plot(
        x, y, "o-", label="{} cells per module".format(cpm),
    )
    plt.fill_between(x, yBelow, yAbove, alpha=0.3)

  plt.plot([1, 10], [2.7, 2.7], "--")

  plt.xlabel("Number of Modules")
  plt.ylabel("Average Number of Sensations")
  plt.legend(loc="upper right")

  plt.savefig(os.path.join(CHART_DIR, "convergence_vs_modules_100_feats.pdf"))

  plt.clf()

  # Cumulative convergence
  #
  # Generated with:
  #   python convergence_simulation.py --numObjects 100 --numUniqueFeatures 10 --locationModuleWidth 20 --resultName results/cumulative_convergence_400_cpm_10_feats_100_objs.json --repeat 10
  #   python convergence_simulation.py --numObjects 100 --numUniqueFeatures 10 --locationModuleWidth 40 --resultName results/cumulative_convergence_1600_cpm_10_feats_100_objs.json --repeat 10
  #   python ideal_sim.py
  #   python bof_sim.py

  # 400 CPM

  yData = collections.defaultdict(list)

  with open("results/cumulative_convergence_400_cpm_10_feats_100_objs.json", "r") as f:
    experiments = json.load(f)
  for exp in experiments:
    cum = 0
    for i in xrange(40):
      step = i + 1
      count = exp[1].get(str(step), 0)
      yData[step].append(count)

  x = [i+1 for i in xrange(20)]
  y = []
  tot = float(sum([sum(counts) for counts in yData.values()]))
  cum = 0.0
  for step in x:
    counts = yData[step]
    cum += float(sum(counts))
    y.append(100.0 * cum / tot)
  std = [np.std(yData[step]) for step in x]
  yBelow = [yi - stdi for yi, stdi in zip(y, std)]
  yAbove = [yi + stdi for yi, stdi in zip(y, std)]

  plt.plot(
      x, y, "o-", label="400 Cells per Module",
  )
  plt.fill_between(x, yBelow, yAbove, alpha=0.3)

  # 1600 CPM

  yData = collections.defaultdict(list)

  with open("results/cumulative_convergence_100_cpm_10_feats_100_objs.json", "r") as f:
    experiments = json.load(f)
  for exp in experiments:
    cum = 0
    for i in xrange(40):
      step = i + 1
      count = exp[1].get(str(step), 0)
      yData[step].append(count)

  x = [i+1 for i in xrange(20)]
  y = []
  tot = float(sum([sum(counts) for counts in yData.values()]))
  cum = 0.0
  for step in x:
    counts = yData[step]
    cum += float(sum(counts))
    y.append(100.0 * cum / tot)
  std = [np.std(yData[step]) for step in x]
  yBelow = [yi - stdi for yi, stdi in zip(y, std)]
  yAbove = [yi + stdi for yi, stdi in zip(y, std)]

  plt.plot(
      x, y, "o-", label="100 Cells Per Module",
  )
  plt.fill_between(x, yBelow, yAbove, alpha=0.3)

  # Ideal
  with open("results/ideal.json", "r") as f:
    idealResults = json.load(f)
  y = []
  std = [np.std(idealResults.get(str(steps), [0])) for steps in x]
  tot = float(sum([sum(counts) for counts in idealResults.values()]))
  cum = 0.0
  for steps in x:
    counts = idealResults.get(str(steps), [])
    if len(counts) > 0:
      cum += float(sum(counts))
    y.append(100.0 * cum / tot)
  yBelow = [yi - stdi for yi, stdi in zip(y, std)]
  yAbove = [yi + stdi for yi, stdi in zip(y, std)]

  plt.plot(
      x, y, "o-", label="Ideal Observor",
  )
  plt.fill_between(x, yBelow, yAbove, alpha=0.3)

  # BOF
  with open("results/bof.json", "r") as f:
    bofResults = json.load(f)
  y = []
  std = [np.std(bofResults.get(str(steps), [0])) for steps in x]
  tot = float(sum([sum(counts) for counts in bofResults.values()]))
  cum = 0.0
  for steps in x:
    counts = bofResults.get(str(steps), [])
    if len(counts) > 0:
      cum += float(sum(counts))
    y.append(100.0 * cum / tot)
  yBelow = [yi - stdi for yi, stdi in zip(y, std)]
  yAbove = [yi + stdi for yi, stdi in zip(y, std)]

  plt.plot(
      x, y, "o-", label="Bag of Features",
  )
  plt.fill_between(x, yBelow, yAbove, alpha=0.3)

  # Formatting
  plt.xlabel("Number of Sensations")
  plt.ylabel("Cumulative Accuracy")
  plt.legend(loc="center right")
  plt.xticks([(i+1)*2 for i in xrange(10)])

  plt.tight_layout()
  plt.savefig(os.path.join(CHART_DIR, "cumulative_accuracy.pdf"))

  plt.clf()


if __name__ == "__main__":
  chart()
