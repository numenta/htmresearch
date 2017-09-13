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
This file plots the results obtained from combined_sequences.py.
"""

import os
import cPickle
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})


def plotOneInferenceRun(stats,
                       fields,
                       basename,
                       itemType="",
                       plotDir="plots",
                       experimentID=0):
  """
  Plots individual inference runs.
  """
  if not os.path.exists(plotDir):
    os.makedirs(plotDir)

  plt.figure()

  # plot request stats
  for field in fields:
    fieldKey = field[0] + " C0"
    plt.plot(stats[fieldKey], marker='+', label=field[1])

  # format
  plt.legend(loc="upper right")
  plt.xlabel("Input number")
  plt.xticks(range(stats["numSteps"]))
  plt.ylabel("Number of cells")
  plt.ylim(-5, 100)
  # plt.ylim(plt.ylim()[0] - 5, plt.ylim()[1] + 5)
  plt.title("Activity while inferring {}".format(itemType))

  # save
  relPath = "{}_exp_{}.pdf".format(basename, experimentID)
  path = os.path.join(plotDir, relPath)
  plt.savefig(path)
  plt.close()


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

  # Generate the first plot for the section "Simulations with Pure
  # Temporal Sequences"
  if False:
    resultsFilename = os.path.join(dirName, "pure_sequences_example.pkl")
    with open(resultsFilename, "rb") as f:
      results = cPickle.load(f)

    for objectId,stat in results["statistics"].itervalues():
      plotOneInferenceRun(
        stat,
        itemType="a single sequence",
        fields=[
          # ("L4 Predicted", "Predicted sensorimotor cells"),
          # ("L2 Representation", "L2 Representation"),
          # ("L4 Representation", "Active sensorimotor cells"),
          ("L4 PredictedActive", "Predicted active cells in sensorimotor layer"),
          ("TM NextPredicted", "Predicted cells in temporal sequence layer"),
          ("TM PredictedActive",
           "Predicted active cells in temporal sequence layer"),
        ],
        basename=exp.name,
        experimentID=objectId,
        plotDir=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "detailed_plots")
      )


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

