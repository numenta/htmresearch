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

from __future__ import print_function

import pprint
import numpy as np
from tabulate import tabulate

from htmresearch.frameworks.pytorch.mnist_sparse_experiment import \
  MNISTSparseExperiment

def analyzeParameters(expName, suite):
  """
  Analyze the impact of each list parameter in this experiment
  """
  print("\n================",expName,"=====================")
  try:
    expParams = suite.get_params(expName)
    pprint.pprint(expParams)

    for p in ["boost_strength", "k", "learning_rate", "weight_sparsity",
              "k_inference_factor", "boost_strength_factor",
              "c1_out_channels", "c1_k", "learning_rate_factor",
              "batches_in_epoch",
              ]:
      if p in expParams and type(expParams[p]) == list:
        print("\n",p)
        for v1 in expParams[p]:
          # Retrieve the last totalCorrect from each experiment
          # Print them sorted from best to worst
          values, params = suite.get_values_fix_params(
            expName, 0, "testerror", "last", **{p:v1})
          v = np.array(values)
          try:
            print("Average/min/max for", p, v1, "=", v.mean(), v.min(), v.max())
            # sortedIndices = v.argsort()
            # for i in sortedIndices[::-1]:
            #   print(v[i],params[i]["name"])
          except:
            print("Can't compute stats for",p)

  except:
    print("Couldn't load experiment",expName)


def summarizeResults(expName, suite):
  """
  Summarize the totalCorrect value from the last iteration for each experiment
  in the directory tree.
  """
  print("\n================",expName,"=====================")

  try:
    # Retrieve the last totalCorrect from each experiment
    # Print them sorted from best to worst
    values, params = suite.get_values_fix_params(
      expName, 0, "totalCorrect", "last")
    v = np.array(values)
    sortedIndices = v.argsort()
    for i in sortedIndices[::-1]:
      print(v[i], params[i]["name"])

    print()
  except:
    print("Couldn't analyze experiment",expName)

  try:
    # Retrieve the last totalCorrect from each experiment
    # Print them sorted from best to worst
    values, params = suite.get_values_fix_params(
      expName, 0, "testerror", "last")
    v = np.array(values)
    sortedIndices = v.argsort()
    for i in sortedIndices[::-1]:
      print(v[i], params[i]["name"])

    print()
  except:
    print("Couldn't analyze experiment",expName)


def lastNoiseCurve(expPath, suite, iteration="last"):
  """
  Print the noise errors from the last iteration of this experiment. Return
  the accuracies in a list for later plotting.
  """
  plotValues = []
  noiseValues = ["0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3",
                            "0.35", "0.4", "0.45", "0.5"]
  print("\nNOISE CURVE =====",expPath,"====== ITERATION:",iteration,"=========")
  try:
    result = suite.get_value(expPath, 0, noiseValues, iteration)
    info = []
    for k in noiseValues:
      info.append([k,result[k]["testerror"]])
      plotValues.append(result[k]["testerror"])
    print(tabulate(info, headers=["noise","Test Error"], tablefmt="grid"))
    print("totalCorrect:", suite.get_value(expPath, 0, "totalCorrect", iteration))
  except:
    print("Couldn't load experiment",expPath)

  return plotValues


def learningCurve(expPath, suite):
  """
  Print the test and overall noise errors from each iteration of this experiment
  """
  print("\nLEARNING CURVE ================",expPath,"=====================")
  try:
    headers=["testerror","totalCorrect","elapsedTime","entropy"]
    result = suite.get_value(expPath, 0, headers, "all")
    info = []
    for i,v in enumerate(zip(result["testerror"],result["totalCorrect"],
                             result["elapsedTime"],result["entropy"])):
      info.append([i, v[0], v[1], int(v[2]), v[3]])
    headers.insert(0,"iteration")
    print(tabulate(info, headers=headers, tablefmt="grid"))
  except:
    print("Couldn't load experiment",expPath)


def bestScore(expPath, suite):
  """
  Given a single experiment, return the test, and total noise score from the
  iteration with maximum validation accuracy.
  """
  maxTestAccuracy = -1.0
  maxTotalAccuracy = -1.0
  maxIter = -1
  try:
    headers=["testerror", "elapsedTime", "totalCorrect"]
    result = suite.get_value(expPath, 0, headers, "all")
    for i,v in enumerate(zip(result["testerror"], result["elapsedTime"],
                             result["totalCorrect"])):
      if v[0] > maxTestAccuracy:
        maxTestAccuracy = v[0]
        if v[2] is not None:
          maxTotalAccuracy = v[2]
        maxIter = i

    return maxTestAccuracy, maxIter, maxTotalAccuracy
  except:
    print("Couldn't load experiment",expPath)
    return None, None, None


def getErrorBars(expPath, suite):
  """
  Go through each experiment in the path. Get the best scores for each
  experiment whose hyperparameters were tuned based on accuracy on validation
  set. Print out overall mean, and stdev for test accuracy, BG accuracy, and
  noise accuracy.
  """
  exps = suite.get_exps(expPath)
  testScores = np.zeros(len(exps))
  noiseScores = np.zeros(len(exps))
  for i,e in enumerate(exps):
    maxTestAccuracy, maxIter, maxTotalAccuracy = bestScore(e, suite)
    testScores[i] = maxTestAccuracy
    noiseScores[i] = maxTotalAccuracy

  print("")
  print("Experiment:", expPath, "Number of sub-experiments", len(exps))
  print("test score mean and standard deviation:", testScores.mean(), testScores.std())
  print("noise score mean and standard deviation:", noiseScores.mean(), noiseScores.std())



# To run it from htmresearch top level:
# python projects/sdr_paper/pytorch_experiments/analyze_experiment.py -c projects/sdr_paper/pytorch_experiments/experiments.cfg

if __name__ == '__main__':

  suite = MNISTSparseExperiment()

  # summarizeResults("./results", suite)
  #
  for expName in [
    # "./results/standardOneLayer",

    # Best sparse CNN net so far
    # "./results/bestSparseNetLuiz",
    # "./results/bestSparseNet",
    # "./results/bestSparseCNN",

    # This is the best sparse net (non CNN) so far, as of Jan 7.
    # "./results/bestSparseNet/k50.0n500.0",
    # "./results/DenseCNN1Seeds/seed47.0",

  ]:
    analyzeParameters(expName, suite)
    learningCurve(expName, suite)

  # These commands will print and plot the actual learning curve
  # sparseCNN2 = lastNoiseCurve("./results/twoLayerSparseCNNs/k100.0n500.0", suite)
  # sparseNN = lastNoiseCurve("./results/bestSparseNet/k50.0n500.0", suite)
  # denseNN = lastNoiseCurve("./results/standardOneLayer", suite)
  # denseCNN2 = lastNoiseCurve("./results/twoLayerDenseCNNs/k1000.0n1000.0", suite)


  # These commands will return the mean and stdev for experiments run on
  # different random seeds.

  # getErrorBars("results/BestDenseOneLayerSeeds", suite)
  # getErrorBars("results/BestDenseOneLayerSeeds2", suite)
  # getErrorBars("results/DenseCNN1Seeds", suite)
  # getErrorBars("results/twoLayerDenseCNNSeeds", suite)
  # getErrorBars("results/bestSparseNetSeeds", suite)
  # getErrorBars("results/bestSparseNet2Seeds", suite)
  # getErrorBars("results/bestSparseCNNOneLayerSeeds", suite)
  # getErrorBars("results/bestSparseCNNTwoLayerSeeds", suite)


  # getErrorBars("results/bestSparseNetSeeds", suite)
  # getErrorBars("results/bestSparseNet2Seeds", suite)
  # getErrorBars("results/bestSparseNet2SeedsLowerLR", suite)
  # getErrorBars("results/bestSparseNet2SeedsLowerLRNoMomentum", suite)

  # getErrorBars("results/sparseCNNFC2KSeeds", suite)
  # getErrorBars("results/sparseCNNFC2WSSeeds", suite)
  # getErrorBars("results/sparseCNNFC2Seeds", suite)
  # getErrorBars("results/denseCNN2SP3Seeds", suite)
