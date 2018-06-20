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

X = [
  0,
  200,
  400,
  600,
  800,
  1000,
  1200,
  1400,
  1600,
  1800,
  2000,
  2200,
  2400,
  2600,
]
Y25 = [
  1,
  0.365,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
]
Y100 = [
  1,
  1,
  1,
  0.08833333333,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
]
Y400 = [
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  0.9992857143,
  0.481875,
  0,
  0,
  0,
  0,
  0,
]
X5k = [
  0,
  500,
  1000,
  1500,
  2000,
  2500,
  3000,
  3500,
  4000,
  4500,
  5000,
]
Y5k25 = [
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  0.9991428571,
  0.992,
  0.9415555556,
  0.8178,
]
Y5k100 = [
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
]
Y5k400 = [
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
]


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

  plt.plot(
      X, Y25, "o-", label="25 cells per module",
  )
  plt.plot(
      X, Y100, "o-", label="100 cells per module",
  )
  plt.plot(
      X, Y400, "o-", label="400 cells per module",
  )

  plt.xlabel("Number of Objects")
  plt.ylabel("Accuracy")
  plt.legend(loc="upper right")

  plt.savefig(os.path.join(CHART_DIR, "capacity100.pdf"))

  plt.clf()

  # Capacity vs num objects with different # of unique features
  #
  # Generated with:
  #   python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 --numUniqueFeatures 50 --locationModuleWidth 20 --resultName results/capacity_50_feats_400_cpm.json
  #   python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 --numUniqueFeatures 100 --locationModuleWidth 20 --resultName results/capacity_100_feats_400_cpm.json
  #   python convergence_simulation.py --numObjects 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 --numUniqueFeatures 500 --locationModuleWidth 20 --resultName results/capacity_500_feats_400_cpm.json

  #plt.style.use("ggplot")

  # TODO
  #plt.plot(
  #    X5k, Y5k25, "-", label="25 cells per module",
  #)

  plt.xlabel("Number of Objects")
  plt.ylabel("Accuracy")
  plt.legend(loc="lower left")
  plt.ylim(-0.01, 1.01)

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
