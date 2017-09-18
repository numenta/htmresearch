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

import cPickle
import matplotlib.pyplot as plt
from optparse import OptionParser
import os
import sys

import numpy

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})


def plotOneInferenceRun(stats,
                        fields,
                        basename,
                        itemType="",
                        plotDir="plots",
                        ymax=100,
                        trialNumber=0):
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
  plt.ylim(-5, ymax)
  plt.title("Activity while inferring {}".format(itemType))

  # save
  relPath = "{}_exp_{}.pdf".format(basename, trialNumber)
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


def plotAccuracyDuringSensorimotorInference(results, featureRange, objectRange,
                         title="", yaxis=""):
  """
  Plot accuracy vs number of features
  """

  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # accuracy[o,f] = accuracy with o objects in training
  # and f unique features.
  accuracy = numpy.zeros((max(objectRange)+1, max(featureRange) + 1))
  totals = numpy.zeros((max(objectRange)+1, max(featureRange) + 1))
  for r in results:
    if r["numFeatures"] in featureRange and r["numObjects"] in objectRange:
      accuracy[r["numObjects"], r["numFeatures"]] += r["sequenceAccuracyPct"]
      totals[r["numObjects"], r["numFeatures"]] += 1

  for o in objectRange:
    for f in featureRange:
      accuracy[o, f] = 100.0 * accuracy[o, f] / totals[o, f]

  ########################################################################
  #
  # Create the plot.
  plt.figure()
  plotPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "plots", "accuracy_during_sensorimotor_inference.pdf")

  # Plot each curve
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']

  for i in range(len(featureRange)):
    f = featureRange[i]
    legendList.append('Sequence layer, feature pool size: {}'.format(f))
    plt.plot(objectRange, accuracy[objectRange, f], color=colorList[i])

  plt.plot(objectRange, [100] * len(objectRange),
           color=colorList[len(featureRange)])
  legendList.append('Sensorimotor layer')

  # format
  plt.legend(legendList, bbox_to_anchor=(0., 0.6, 1., .102), loc="right", prop={'size':10})
  plt.xlabel("Number of objects")
  plt.ylim(-10.0, 110.0)
  plt.ylabel(yaxis)
  plt.title(title)

    # save
  plt.savefig(plotPath)
  plt.close()


def plotAccuracyDuringSequenceInference(results, locationRange, featureRange,
                             seqRange, title="", yaxis=""):
  """
  Plot accuracy vs number of locations
  """

  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # accuracy[f,l] = how long it took it to converge with f unique features
  # and l locations on average.
  accuracy = numpy.zeros((max(featureRange)+1, max(locationRange) + 1))
  totals = numpy.zeros((max(featureRange)+1, max(locationRange) + 1))
  for r in results:
    if r["numFeatures"] in featureRange and r["numSequences"] in seqRange:
      accuracy[r["numFeatures"], r["numLocations"]] += r["sensorimotorAccuracyPct"]
      totals[r["numFeatures"], r["numLocations"]] += 1

  for f in featureRange:
    for l in locationRange:
      accuracy[f, l] = 100.0*accuracy[f, l] / totals[f, l]

  ########################################################################
  #
  # Create the plot.
  plt.figure()
  plotPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "plots", "accuracy_during_sequence_inference.pdf")

  # Plot each curve
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']

  for i in range(len(featureRange)):
    f = featureRange[i]
    legendList.append('Sensorimotor layer, feature pool size: {}'.format(f))
    plt.plot(locationRange, accuracy[f,locationRange],
             color=colorList[i])

  plt.plot(locationRange, [100] * len(locationRange),
           color=colorList[len(featureRange)])
  legendList.append('Temporal sequence layer')

  # format
  plt.legend(legendList, bbox_to_anchor=(0., 0.65, 1., .102), loc="right", prop={'size':10})
  plt.xlabel("Size of location pool")
  # plt.xticks(range(0,max(locationRange)+1,10))
  # plt.yticks(range(0,int(accuracy.max())+2,10))
  plt.ylim(-10.0, 110.0)
  plt.ylabel(yaxis)
  plt.title(title)

    # save
  plt.savefig(plotPath)
  plt.close()




if __name__ == "__main__":

  dirName = os.path.dirname(os.path.realpath(__file__))

  parser = OptionParser("python %prog [-h]\n\n"
          "Regenerate the plots for every figure, if the "
          "appropriate pkl file exists.")
  options, args = parser.parse_args(sys.argv[1:])

  # Generate images similar to those used in the first plot for the section
  # "Simulations with Pure Temporal Sequences"
  resultsFig4A = os.path.join(dirName, "pure_sequences_example.pkl")
  if os.path.exists(resultsFig4A):
    with open(resultsFig4A, "rb") as f:
      results = cPickle.load(f)

    for trialNum, stat in enumerate(results["statistics"]):
      plotOneInferenceRun(
        stat,
        itemType="a single sequence",
        fields=[
          ("L4 PredictedActive", "Predicted active cells in sensorimotor layer"),
          ("TM NextPredicted", "Predicted cells in temporal sequence layer"),
          ("TM PredictedActive",
           "Predicted active cells in temporal sequence layer"),
        ],
        basename="pure_sequences",
        trialNumber=trialNum,
        plotDir=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "detailed_plots")
      )
    print "Plots for Fig 4A generated in 'detailed_plots'"

  # Generate the second plot for the section "Simulations with Pure
  # Temporal Sequences"
  resultsFig4B = os.path.join(dirName, "sequence_batch_results.pkl")
  if os.path.exists(resultsFig4B):
    featureRange = [5, 10, 100]
    seqRange = [50]
    locationRange = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                     1000, 1100, 1200, 1300, 1400, 1500, 1600]
    # Analyze results
    with open(resultsFig4B, "rb") as f:
      results = cPickle.load(f)

    plotAccuracyDuringSequenceInference(
      results, locationRange, featureRange, seqRange,
      title="Relative performance of layers while inferring temporal sequences",
      yaxis="Accuracy (%)")

    print "Plots for Fig 4B generated in 'plots'"

  # Generate images similar to the first plot for the section "Simulations with
  # Sensorimotor Sequences"
  resultsFig5A = os.path.join(dirName, "sensorimotor_sequence_example.pkl")
  if os.path.exists(resultsFig5A):
    with open(resultsFig5A, "rb") as f:
      results = cPickle.load(f)

    for trialNum, stat in enumerate(results["statistics"]):
      plotOneInferenceRun(
        stat,
        itemType="a single object",
        fields=[
          ("L4 PredictedActive", "Predicted active cells in sensorimotor layer"),
          ("TM NextPredicted", "Predicted cells in temporal sequence layer"),
          ("TM PredictedActive",
           "Predicted active cells in temporal sequence layer"),
        ],
        basename="sensorimotor_sequences",
        trialNumber=trialNum,
        ymax=50,
        plotDir=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "detailed_plots")
      )

    print "Plots for Fig 5A generated in 'detailed_plots'"


  # Generate the second plot for the section "Simulations with Sensorimotor
  # Sequences"
  resultsFig5B = os.path.join(dirName, "sensorimotor_batch_results.pkl")
  if os.path.exists(resultsFig5B):
    # These ranges must equal or be a subset of the actual ranges that were run
    featureRange = [5, 10, 50]
    objectRange = [2, 5, 10, 20, 30, 40, 50, 70]

    # Analyze results
    with open(resultsFig5B, "rb") as f:
      results = cPickle.load(f)

    plotAccuracyDuringSensorimotorInference(
      results, featureRange, objectRange,
      title="Relative performance of layers during sensorimotor inference",
      yaxis="Accuracy (%)")

    print "Plots for Fig 5B generated in 'plots'"


  # Generate a plot similar to one in the section "Simulations with Combined
  # Sequences".  Note that the dashed vertical lines and labels were added in
  # manually.
  resultsFig6 = os.path.join(dirName, "combined_results.pkl")
  if os.path.exists(resultsFig6):
    with open(resultsFig6, "rb") as f:
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

    print "Plots for Fig 6 generated in 'plots'"
