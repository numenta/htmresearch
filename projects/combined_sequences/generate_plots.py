# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

"""
This file plots the behavior of L4-L2-TM network as you train it on sequences.
"""

import os
import cPickle
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})


def plotMultipleInferenceRun(stats,
                       fields,
                       basename,
                       plotDir="plots"):
  """
  Plots individual inference runs.
  """
  if not os.path.exists(plotDir):
    os.makedirs(plotDir)

  plt.figure()

  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']

  # plot request stats
  for i, field in enumerate(fields):
    fieldKey = field[0] + " C0"
    trace = []
    for s in stats:
      trace += s[fieldKey]
    plt.plot(trace, label=field[1], color=colorList[i])

  # format
  plt.legend(loc="upper right")
  plt.xlabel("Input number")
  plt.xticks(range(0, len(stats)*stats[0]["numSteps"]+1,5))
  plt.ylabel("Number of cells")
  plt.ylim(-5, 55)
  plt.title("Inferring combined sensorimotor and temporal sequence stream")

  # save
  relPath = "{}_exp_combined.pdf".format(basename)
  path = os.path.join(plotDir, relPath)
  plt.savefig(path)
  plt.close()



if __name__ == "__main__":

  dirName = os.path.dirname(os.path.realpath(__file__))

  # Generate plots for the section "Simulations with Combined Sequences"
  if True:
    resultsFilename = os.path.join(dirName, "combined_results.pkl")
    with open(resultsFilename, "rb") as f:
      results = cPickle.load(f)

    plotMultipleInferenceRun(
      results["statistics"][0:10],
      fields=[
        ("L4 PredictedActive", "Predicted active cells in sensorimotor layer"),
        ("TM PredictedActive",
         "Predicted active cells in temporal sequence layer"),
      ],
      basename=results["name"],
      plotDir=os.path.join(dirName, "plots")
    )

