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
import os
import csv
import numpy as np
from tabulate import tabulate

from htmresearch.frameworks.pytorch.sparse_speech_experiment import \
  SparseSpeechExperiment


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
              "batches_in_epoch", "seed",
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
  print("Test error:")
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

  print("Validation error:")
  try:
    # Retrieve the last totalCorrect from each experiment
    # Print them sorted from best to worst
    values, params = suite.get_values_fix_params(
      expName, 0, "validationerror", "last")
    v = np.array(values)
    sortedIndices = v.argsort()
    for i in sortedIndices[::-1]:
      print(v[i], params[i]["name"])

    print()
  except:
    print("Couldn't analyze experiment",expName)


def lastNoiseCurve(expPath, suite, iteration="last"):
  """
  Print the noise errors from the last iteration of this experiment
  """
  noiseValues = ["0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3",
                            "0.35", "0.4", "0.45", "0.5"]
  print("\nNOISE CURVE =====",expPath,"====== ITERATION:",iteration,"=========")
  try:
    result = suite.get_value(expPath, 0, noiseValues, iteration)
    info = []
    for k in noiseValues:
      info.append([k,result[k]["testerror"]])
    print(tabulate(info, headers=["noise","Test Error"], tablefmt="grid"))
    print("totalCorrect:", suite.get_value(expPath, 0, "totalCorrect", iteration))
  except:
    print("Couldn't load experiment",expPath)


def learningCurve(expPath, suite):
  """
  Print the test, validation and other scores from each iteration of this
  experiment.  We select the test score that corresponds to the iteration with
  maximum validation accuracy.
  """
  print("\nLEARNING CURVE ================",expPath,"=====================")
  try:
    headers=["testResults","validation","bgResults","elapsedTime"]
    result = suite.get_value(expPath, 0, headers, "all")
    info = []
    maxValidationAccuracy = -1.0
    maxTestAccuracy = -1.0
    maxBGAccuracy = -1.0
    maxIter = -1
    for i,v in enumerate(zip(result["testResults"],result["validation"],
                             result["bgResults"], result["elapsedTime"])):
      info.append([i, v[0]["testerror"], v[1]["testerror"], v[2]["testerror"], int(v[3])])
      if v[1]["testerror"] > maxValidationAccuracy:
        maxValidationAccuracy = v[1]["testerror"]
        maxTestAccuracy = v[0]["testerror"]
        maxBGAccuracy = v[2]["testerror"]
        maxIter = i
    headers.insert(0,"iteration")
    print(tabulate(info, headers=headers, tablefmt="grid"))

    print("Max validation score =", maxValidationAccuracy, " at iteration", maxIter)
    print("Test score at that iteration =", maxTestAccuracy)
    print("BG score at that iteration =", maxBGAccuracy)
  except:
    print("Couldn't load experiment",expPath)


def bestScore(expPath, suite):
  """
  Given a single experiment, return the test, validation and other scores from
  the iteration with maximum validation accuracy.
  """
  maxValidationAccuracy = -1.0
  maxTestAccuracy = -1.0
  maxTotalAccuracy = -1.0
  maxBGAccuracy = -1.0
  maxIter = -1
  try:
    headers=["testResults", "validation", "bgResults", "elapsedTime", "totalCorrect"]
    result = suite.get_value(expPath, 0, headers, "all")
    for i,v in enumerate(zip(result["testResults"], result["validation"],
                             result["bgResults"], result["elapsedTime"],
                             result["totalCorrect"])):
      if v[1]["testerror"] > maxValidationAccuracy:
        maxValidationAccuracy = v[1]["testerror"]
        maxTestAccuracy = v[0]["testerror"]
        maxBGAccuracy = v[2]["testerror"]
        if v[4] is not None:
          maxTotalAccuracy = v[4]
        maxIter = i

    # print("Max validation score =", maxValidationAccuracy, " at iteration", maxIter)
    # print("Test score at that iteration =", maxTestAccuracy)
    # print("BG score at that iteration =", maxBGAccuracy)
    return maxTestAccuracy, maxValidationAccuracy, maxBGAccuracy, maxIter, maxTotalAccuracy
  except:
    print("Couldn't load experiment",expPath)
    return None, None, None, None, None


def findOptimalResults(expName, suite, outFile):
  """
  Go through every experiment in the specified folder. For each experiment, find
  the iteration with the best validation score, and return the metrics
  associated with that iteration.
  """
  writer = csv.writer(outFile)
  headers = ["testAccuracy", "bgAccuracy", "maxTotalAccuracy", "experiment path"]
  writer.writerow(headers)
  info = []
  print("\n================",expName,"=====================")
  try:
    # Retrieve the last totalCorrect from each experiment
    # Print them sorted from best to worst
    values, params = suite.get_values_fix_params(
      expName, 0, "testerror", "last")
    for p in params:
      expPath = p["name"]
      if not "results" in expPath:
        expPath = os.path.join("results", expPath)
      maxTestAccuracy, maxValidationAccuracy, maxBGAccuracy, maxIter, maxTotalAccuracy = bestScore(expPath, suite)
      row = [maxTestAccuracy, maxBGAccuracy, maxTotalAccuracy, expPath]
      info.append(row)
      writer.writerow(row)

    print(tabulate(info, headers=headers, tablefmt="grid"))
  except:
    print("Couldn't analyze experiment",expName)


def getErrorBars(expPath, suite):
  """
  Go through each experiment in the path. Get the best scores for each experiment
  based on accuracy on validation set. Print out overall mean, and stdev for
  test accuracy, BG accuracy, and noise accuracy.
  """
  exps = suite.get_exps(expPath)
  testScores = np.zeros(len(exps))
  noiseScores = np.zeros(len(exps))
  for i,e in enumerate(exps):
    maxTestAccuracy, maxValidationAccuracy, maxBGAccuracy, maxIter, maxTotalAccuracy = bestScore(
      e, suite)
    testScores[i] = maxTestAccuracy
    noiseScores[i] = maxTotalAccuracy
    print(e, maxTestAccuracy, maxTotalAccuracy)

  print("")
  print("Experiment:", expPath, "Number of sub-experiments", len(exps))
  print("test score mean and standard deviation:", testScores.mean(), testScores.std())
  print("noise score mean and standard deviation:", noiseScores.mean(), noiseScores.std())


if __name__ == '__main__':

  suite = SparseSpeechExperiment()

  # Find the test scores corresponding to the highest validation scores.
  # with open("out.csv", "wb") as f:
  #   findOptimalResults("./results", suite, f)

  # More details for some experiments
  # for expName in [
  #   # "./results/cnn13/learning_rate_factor0.80c1_out_channels64_64momentum0.90learning_rate0.010k1000n1000",
  #   # "./results/cnn15/c1_k2500_320c1_out_channels64_64momentum0.0k200n1000"
  #   "./old_results/cnn20/weight_sparsity0.40k100n1500",
  #
  # ]:
  #   analyzeParameters(expName, suite)
  #   learningCurve(expName, suite)


  # # Print details of the best ones so far
  #
  # lastNoiseCurve("./results/bestSparseCNN", suite)
  #
  # lastNoiseCurve("./results/cnn13/learning_rate0.020boost_strength1.40", suite)

  # lastNoiseCurve("./results/cnn9/weight_sparsity0.30c1_k400.0k50.0n150.0", suite, 5)

  # Error bars for 10 random seeds of each experiment. Used for Tables 1 and 2.

  # Dense CNN 2
  getErrorBars("./results/cnn22DenseSeedsNodropout", suite)
  # getErrorBars("./results/cnn22DenseSeedsWithDropout", suite)

  # Sparse CNN 2
  getErrorBars("./results/cnn23SparseSeedsMoreIters", suite)

  # Super sparse CNN 2
  # getErrorBars("./results/cnn28LargerWeight10pct", suite)

