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

"""Plot capacity chart."""

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

  # Capacity vs num objects for 100 unique features
  #
  # Generated with:
  #   python convergence_simulation.py --numObjects 200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 --numUniqueFeatures 100 --locationModuleWidth 5 --resultName results/capacity_100_feats_25_cpm.json
  #   python convergence_simulation.py --numObjects 200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 --numUniqueFeatures 100 --locationModuleWidth 10 --resultName results/capacity_100_feats_100_cpm.json
  #   python convergence_simulation.py --numObjects 200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 --numUniqueFeatures 100 --locationModuleWidth 20 --resultName results/capacity_100_feats_400_cpm.json

  #plt.style.use("ggplot")

  markers = ("s", "o", "^")
  for cpm, marker in zip((100, 196, 400), markers):
    with open("results/capacity_1_500_feats_{}_cpm.json".format(str(cpm)), "r") as f:
      experiments = json.load(f)
    expResults = []
    for exp in experiments:
      numObjects = exp[0]["numObjects"]
      #if numObjects > 2601:
      #  continue
      failed = exp[1].get("null", 0)
      expResults.append((
        numObjects,
        1.0 - (float(failed) / float(numObjects))
      ))

    x = []
    y = []
    for i, j in sorted(expResults):
      x.append(i)
      y.append(j)

    plt.plot(
      x, y, "{}-".format(marker), label="{} Cells Per Module".format(str(cpm)),
    )

  plt.xlabel("Number of Objects")
  plt.ylabel("Accuracy")
  plt.legend(loc="center left")

  plt.tight_layout()

  plt.savefig(os.path.join(CHART_DIR, "capacity100.pdf"))

  plt.clf()

  # Capacity vs num objects with different # of unique features
  #
  # Generated with:
  #   python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 --numUniqueFeatures 50 --locationModuleWidth 20 --thresholds 18 --resultName results/capacity_50_feats_400_cpm.json
  #   python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 --numUniqueFeatures 100 --locationModuleWidth 20 --thresholds 18 --resultName results/capacity_100_feats_400_cpm.json
  #   python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 --numUniqueFeatures 500 --locationModuleWidth 20 --thresholds 18 --resultName results/capacity_500_feats_400_cpm.json

  #plt.style.use("ggplot")

  for feats, marker in zip((100, 200, 500), markers):
    with open("results/capacity_{}_feats_400_cpm.json".format(str(feats)), "r") as f:
      experiments = json.load(f)
    expResults = []
    for exp in experiments:
      numObjects = exp[0]["numObjects"]
      failed = exp[1].get("null", 0)
      expResults.append((
        numObjects,
        1.0 - (float(failed) / float(numObjects))
      ))

    x = []
    y = []
    for i, j in sorted(expResults):
      x.append(i)
      y.append(j)

    plt.plot(
      x, y, "{}-".format(marker), label="{} Unique Features".format(str(feats)),
    )

  plt.xlabel("Number of Objects")
  plt.ylabel("Accuracy")
  plt.legend(loc="center left")

  plt.tight_layout()

  plt.savefig(os.path.join(CHART_DIR, "capacity_with_features.pdf"))

  plt.clf()

  ## Capacity vs num objects for 5000 unique features

  #plt.style.use("ggplot")

  #plt.plot(
  #    X5k, Y5k25, "-", label="25 cells per module",
  #)
  #plt.plot(
  #    X5k, Y5k100, "-", label="100 cells per module",
  #)
  #plt.plot(
  #    X5k, Y5k400, "-", label="400 cells per module",
  #)

  #plt.xlabel("Number of Objects")
  #plt.ylabel("Accuracy")
  #plt.legend(loc="lower left")
  #plt.ylim(-0.01, 1.01)

  #plt.savefig(os.path.join(CHART_DIR, "capacity5000.pdf"))

  #plt.clf()


if __name__ == "__main__":
  chart()
