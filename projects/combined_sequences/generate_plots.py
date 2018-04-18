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
from collections import defaultdict

import numpy

import matplotlib as mpl
import traceback

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


def plotAccuracyDuringSensorimotorInference(resultsFig5B, title="", yaxis=""):
  """
  Plot accuracy vs number of features
  """
  # Read out results and get the ranges we want.
  with open(resultsFig5B, "rb") as f:
    results = cPickle.load(f)

  objectRange = []
  featureRange = []
  for r in results:
    if r["numObjects"] not in objectRange: objectRange.append(r["numObjects"])
    if r["numFeatures"] not in featureRange: featureRange.append(r["numFeatures"])
  objectRange.sort()
  featureRange.sort()
  print "objectRange=",objectRange
  print "featureRange=",featureRange

  ########################################################################
  #
  # Accumulate the TM accuracies for each condition in a list and compute mean
  # and stdeviations
  # For L2 we average across all feature ranges
  accuracies = defaultdict(list)
  l2Accuracies = defaultdict(list)
  for r in results:
    accuracies[(r["numObjects"], r["numFeatures"])].append(r["objectCorrectSparsityTM"])
    l2Accuracies[r["numObjects"]].append(r["objectAccuracyL2"])

  # meanAccuracy[o,f] = accuracy of TM with o objects and f unique features.
  meanAccuracy = numpy.zeros((max(objectRange)+1, max(featureRange) + 1))
  stdev = numpy.zeros((max(objectRange)+1, max(featureRange) + 1))
  meanL2Accuracy = numpy.zeros(max(objectRange)+1)
  stdevL2 = numpy.zeros(max(objectRange)+1)
  for o in objectRange:
    for f in featureRange:
      a = numpy.array(accuracies[(o, f)])
      meanAccuracy[o, f] = 100.0*a.mean()
      stdev[o, f] = 100.0*a.std()

    # Accuracies for L2
    a = numpy.array(l2Accuracies[o])
    meanL2Accuracy[o] = 100.0*a.mean()
    stdevL2[o] = 100.0*a.std()


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
    plt.errorbar(objectRange, meanAccuracy[objectRange, f],
                 yerr=stdev[objectRange, f],
                 color=colorList[i])

  plt.errorbar(objectRange, meanL2Accuracy[objectRange],
               yerr=stdevL2[objectRange],
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


def plotAccuracyDuringDecrementChange(results, title="", yaxis=""):
  """
  Plot accuracy vs decrement value
  """

  decrementRange = []
  featureRange = []
  for r in results:
    if r["basalPredictedSegmentDecrement"] not in decrementRange:
      decrementRange.append(r["basalPredictedSegmentDecrement"])
    if r["numFeatures"] not in featureRange:
      featureRange.append(r["numFeatures"])
  decrementRange.sort()
  featureRange.sort()
  print decrementRange
  print featureRange

  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # accuracy[o,f] = accuracy with o objects in training
  # and f unique features.
  accuracy = numpy.zeros((len(featureRange), len(decrementRange)))
  TMAccuracy = numpy.zeros((len(featureRange), len(decrementRange)))
  totals = numpy.zeros((len(featureRange), len(decrementRange)))
  for r in results:
    dec = r["basalPredictedSegmentDecrement"]
    nf = r["numFeatures"]
    accuracy[featureRange.index(nf), decrementRange.index(dec)] += r["objectAccuracyL2"]
    TMAccuracy[featureRange.index(nf), decrementRange.index(dec)] += r["sequenceCorrectClassificationsTM"]
    totals[featureRange.index(nf), decrementRange.index(dec)] += 1

  for i,f in enumerate(featureRange):
    print i, f, accuracy[i] / totals[i]
    print i, f, TMAccuracy[i] / totals[i]
    print

  # ########################################################################
  # #
  # # Create the plot.
  # plt.figure()
  # plotPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
  #                         "plots", "accuracy_during_sensorimotor_inference.pdf")
  #
  # # Plot each curve
  # legendList = []
  # colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
  #
  # for i in range(len(featureRange)):
  #   f = featureRange[i]
  #   legendList.append('Sequence layer, feature pool size: {}'.format(f))
  #   plt.plot(objectRange, accuracy[objectRange, f], color=colorList[i])
  #
  # plt.plot(objectRange, [100] * len(objectRange),
  #          color=colorList[len(featureRange)])
  # legendList.append('Sensorimotor layer')
  #
  # # format
  # plt.legend(legendList, bbox_to_anchor=(0., 0.6, 1., .102), loc="right", prop={'size':10})
  # plt.xlabel("Number of objects")
  # plt.ylim(-10.0, 110.0)
  # plt.ylabel(yaxis)
  # plt.title(title)
  #
  # # save
  # plt.savefig(plotPath)
  # plt.close()


def plotAccuracyAndMCsDuringDecrementChange(results, title="", yaxis=""):
  """
  Plot accuracy vs decrement value
  """

  decrementRange = []
  mcRange = []
  for r in results:
    if r["basalPredictedSegmentDecrement"] not in decrementRange:
      decrementRange.append(r["basalPredictedSegmentDecrement"])
    if r["inputSize"] not in mcRange:
      mcRange.append(r["inputSize"])
  decrementRange.sort()
  mcRange.sort()
  print decrementRange
  print mcRange

  ########################################################################
  #
  # Accumulate all the results per column in a convergence array.
  #
  # accuracy[o,f] = accuracy with o objects in training
  # and f unique features.
  accuracy = numpy.zeros((len(mcRange), len(decrementRange)))
  TMAccuracy = numpy.zeros((len(mcRange), len(decrementRange)))
  totals = numpy.zeros((len(mcRange), len(decrementRange)))
  for r in results:
    dec = r["basalPredictedSegmentDecrement"]
    nf = r["inputSize"]
    accuracy[mcRange.index(nf), decrementRange.index(dec)] += r["objectAccuracyL2"]
    TMAccuracy[mcRange.index(nf), decrementRange.index(dec)] += r["sequenceCorrectClassificationsTM"]
    totals[mcRange.index(nf), decrementRange.index(dec)] += 1

  for i,f in enumerate(mcRange):
    print i, f, accuracy[i] / totals[i]
    print i, f, TMAccuracy[i] / totals[i]
    print i, f, totals[i]
    print

  # ########################################################################
  # #
  # # Create the plot.
  # plt.figure()
  # plotPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
  #                         "plots", "accuracy_during_sensorimotor_inference.pdf")
  #
  # # Plot each curve
  # legendList = []
  # colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
  #
  # for i in range(len(featureRange)):
  #   f = featureRange[i]
  #   legendList.append('Sequence layer, feature pool size: {}'.format(f))
  #   plt.plot(objectRange, accuracy[objectRange, f], color=colorList[i])
  #
  # plt.plot(objectRange, [100] * len(objectRange),
  #          color=colorList[len(featureRange)])
  # legendList.append('Sensorimotor layer')
  #
  # # format
  # plt.legend(legendList, bbox_to_anchor=(0., 0.6, 1., .102), loc="right", prop={'size':10})
  # plt.xlabel("Number of objects")
  # plt.ylim(-10.0, 110.0)
  # plt.ylabel(yaxis)
  # plt.title(title)
  #
  # # save
  # plt.savefig(plotPath)
  # plt.close()


def plotAccuracyDuringSequenceInference(dirName, title="", yaxis=""):
  """
  Plot accuracy vs number of locations
  """
  # Read in results file
  with open(os.path.join(dirName,
            "sequence_batch_high_dec_normal_features.pkl"), "rb") as f:
    results = cPickle.load(f)

  locationRange = []
  featureRange = []
  for r in results:
    if r["numLocations"] not in locationRange: locationRange.append(r["numLocations"])
    if r["numFeatures"] not in featureRange: featureRange.append(r["numFeatures"])
    locationRange.sort()
  featureRange.sort()
  if 10 in featureRange: featureRange.remove(10)
  print "locationRange=",locationRange
  print "featureRange=",featureRange

  ########################################################################
  #
  # Accumulate the L2 accuracies for each condition in a list and compute mean
  # and stdeviations
  # For TM we average across all feature ranges
  L2Accuracies = defaultdict(list)
  TMAccuracies = defaultdict(list)
  for r in results:
    if r["numFeatures"] in featureRange:
      L2Accuracies[(r["numLocations"], r["numFeatures"])].append(r["sequenceAccuracyL2"])
      TMAccuracies[r["numLocations"]].append(r["sequenceCorrectSparsityTM"])

  # meanAccuracy[o,f] = accuracy of TM with o objects and f unique features.
  meanL2Accuracy = numpy.zeros((max(locationRange)+1, max(featureRange) + 1))
  stdevL2 = numpy.zeros((max(locationRange)+1, max(featureRange) + 1))
  meanTMAccuracy = numpy.zeros(max(locationRange)+1)
  stdevTM = numpy.zeros(max(locationRange)+1)
  for o in locationRange:
    for f in featureRange:
      a = numpy.array(L2Accuracies[(o, f)])
      meanL2Accuracy[o, f] = 100.0*a.mean()
      stdevL2[o, f] = 100.0*a.std()

    # Accuracies for TM
    a = numpy.array(TMAccuracies[o])
    meanTMAccuracy[o] = 100.0*a.mean()
    stdevTM[o] = 100.0*a.std()

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
    plt.errorbar(locationRange, meanL2Accuracy[locationRange, f],
             yerr=stdevL2[locationRange, f],
             color=colorList[i])

  plt.errorbar(locationRange, meanTMAccuracy[locationRange],
               yerr=stdevTM[locationRange],
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


def plotAccuracyVsSequencesDuringSequenceInference(dirName, title="", yaxis=""):

  # Read in results file
  with open(os.path.join(dirName, "sequences_range_2048_mcs.pkl"), "rb") as f:
    results = cPickle.load(f)

  sequenceRange = []
  featureRange = []
  for r in results:
    if r["numSequences"] not in sequenceRange: sequenceRange.append(r["numSequences"])
    if r["numFeatures"] not in featureRange: featureRange.append(r["numFeatures"])
    sequenceRange.sort()
  featureRange.sort()
  if 10 in featureRange: featureRange.remove(10)
  print "numSequences=",sequenceRange
  print "featureRange=",featureRange

  ########################################################################
  #
  # Accumulate the L2 accuracies for each condition in a list and compute mean
  # and stdeviations
  # For TM we average across all feature ranges
  L2Accuracies = defaultdict(list)
  TMAccuracies = defaultdict(list)
  for r in results:
    if r["numFeatures"] in featureRange:
      L2Accuracies[(r["numSequences"], r["numFeatures"])].append(r["sequenceAccuracyL2"])
      TMAccuracies[r["numSequences"]].append(r["sequenceCorrectSparsityTM"])

  # meanAccuracy[o,f] = accuracy of TM with o objects and f unique features.
  meanL2Accuracy = numpy.zeros((max(sequenceRange)+1, max(featureRange) + 1))
  stdevL2 = numpy.zeros((max(sequenceRange)+1, max(featureRange) + 1))
  meanTMAccuracy = numpy.zeros(max(sequenceRange)+1)
  stdevTM = numpy.zeros(max(sequenceRange)+1)
  for o in sequenceRange:
    for f in featureRange:
      a = numpy.array(L2Accuracies[(o, f)])
      meanL2Accuracy[o, f] = 100.0*a.mean()
      stdevL2[o, f] = 100.0*a.std()

    # Accuracies for TM
    a = numpy.array(TMAccuracies[o])
    meanTMAccuracy[o] = 100.0*a.mean()
    stdevTM[o] = 100.0*a.std()

  ########################################################################
  #
  # Create the plot.
  plt.figure()
  plotPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "plots", "accuracy_vs_sequences_2048_mcs.pdf")

  # Plot each curve
  legendList = []
  colorList = ['r', 'b', 'g', 'm', 'c', 'k', 'y']

  for i in range(len(featureRange)):
    f = featureRange[i]
    legendList.append('Sensorimotor layer, feature pool size: {}'.format(f))
    plt.errorbar(sequenceRange, meanL2Accuracy[sequenceRange, f],
             yerr=stdevL2[sequenceRange, f],
             color=colorList[i])

  plt.errorbar(sequenceRange, meanTMAccuracy[sequenceRange],
               yerr=stdevTM[sequenceRange],
               color=colorList[len(featureRange)])
  legendList.append('Temporal sequence layer')

  # format
  plt.legend(legendList, bbox_to_anchor=(0., 0.65, 1., .102),
             loc="right", prop={'size':10})
  plt.xlabel("Number of sequences")
  # plt.xticks(range(0,max(locationRange)+1,10))
  # plt.yticks(range(0,int(accuracy.max())+2,10))
  plt.ylim(-10.0, 110.0)
  plt.ylabel(yaxis)
  plt.title(title)

    # save
  plt.savefig(plotPath)
  plt.close()


def gen4(dirName):
  """Plots 4A and 4B"""
  # Generate images similar to those used in the first plot for the section
  # "Simulations with Pure Temporal Sequences"
  try:
    resultsFig4A = os.path.join(dirName, "pure_sequences_example.pkl")
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
  except Exception, e:
    print "\nCould not generate plots for Fig 4A: "
    traceback.print_exc()
    print

  # Generate the second plot for the section "Simulations with Pure
  # Temporal Sequences"
  try:
    plotAccuracyDuringSequenceInference(
      dirName,
      title="Relative performance of layers while inferring temporal sequences",
      yaxis="Accuracy (%)")

    print "Plots for Fig 4B generated in 'plots'"
  except Exception, e:
    print "\nCould not generate plots for Fig 4B: "
    traceback.print_exc()
    print

  # Generate the accuracy vs number of sequences
  try:
    plotAccuracyVsSequencesDuringSequenceInference(
      dirName,
      title="Relative performance of layers while inferring temporal sequences",
      yaxis="Accuracy (%)")

    print "Plots for Fig 4C generated in 'plots'"
  except Exception, e:
    print "\nCould not generate plots for Fig 4C: "
    traceback.print_exc()
    print


def gen5(dirName):
  # Generate images similar to the first plot for the section "Simulations with
  # Sensorimotor Sequences"
  try:
    resultsFig5A = os.path.join(dirName, "sensorimotor_sequence_example.pkl")
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
  except Exception, e:
    print "\nCould not generate plots for Fig 5A: "
    traceback.print_exc()
    print


  # Generate the second plot for the section "Simulations with Sensorimotor
  # Sequences"
  try:
    resultsFig5B = os.path.join(dirName, "sensorimotor_batch_results_more_objects.pkl")

    plotAccuracyDuringSensorimotorInference(
      resultsFig5B,
      title="Relative performance of layers during sensorimotor inference",
      yaxis="Accuracy (%)")

    print "Plots for Fig 5B generated in 'plots'"
  except Exception, e:
    print "\nCould not generate plots for Fig 5B: "
    traceback.print_exc()
    print


def gen6(dirName):
  # Generate a plot similar to one in the section "Simulations with Combined
  # Sequences".  Note that the dashed vertical lines and labels were added in
  # manually.
  try:
    resultsFig6 = os.path.join(dirName, "combined_results.pkl")
    # resultsFig6 = os.path.join(dirName, "superimposed_sequence_results.pkl")
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
  except Exception, e:
    print "\nCould not generate plots for Fig 6: "
    traceback.print_exc()
    print


if __name__ == "__main__":

  dirName = os.path.dirname(os.path.realpath(__file__))

  parser = OptionParser("python %prog [-h]\n\n"
          "Regenerate the plots for every figure, if the "
          "appropriate pkl file exists.")
  options, args = parser.parse_args(sys.argv[1:])


  gen4(dirName)
  # gen5(dirName)
  # gen6(dirName)

  # Generate performance as a function of decrements
  # try:
  #   for fn in [
  #     # "superimposed_more_increments_500_features.pkl",
  #     "superimposed_pool_increments_varying_features.pkl",
  #     "superimposed_more_increments_1000_features.pkl",
  #     "superimposed_more_increments_varying_features.pkl",
  #     "superimposed_more_increments_50_features.pkl",
  #     "superimposed_smaller_mcs.pkl",
  #   ]:
  #     # resultsFile = os.path.join(dirName, "superimposed_pool_increments_stripped.pkl")
  #     resultsFile = os.path.join(dirName, fn)
  #     print "\n\nFile: ",fn
  #
  #     # Analyze results
  #     with open(resultsFile, "rb") as f:
  #       results = cPickle.load(f)
  #
  #     plotAccuracyDuringDecrementChange(results)
  #
  #     # print "Plots for decrements generated in 'plots'"
  # except Exception, e:
  #   print "\nCould not generate plots for decrements: "
  #   traceback.print_exc()
  #   print


  # Generate performance as a function of minicolumns
  # try:
  #   for fn in [
  #     "superimposed_range_of_mcs.pkl",
  #   ]:
  #     resultsFile = os.path.join(dirName, fn)
  #     print "\n\nFile: ",fn
  #
  #     # Analyze results
  #     with open(resultsFile, "rb") as f:
  #       results = cPickle.load(f)
  #
  #       plotAccuracyAndMCsDuringDecrementChange(results)
  #
  #     # print "Plots for decrements generated in 'plots'"
  # except Exception, e:
  #   print "\nCould not generate plots for decrements: "
  #   traceback.print_exc()
  #   print
