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
              "k_inference_factor", "boost_strength_factor"]:
      if p in expParams and type(expParams[p]) == list:
        print(p)
        for v1 in expParams[p]:
          # Retrieve the last totalCorrect from each experiment
          # Print them sorted from best to worst
          values, params = suite.get_values_fix_params(
            expName, 0, "totalCorrect", "last", **{p:v1})
          v = np.array(values)
          print(v1,v)
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


def lastNoiseCurve(expPath, suite):
  """
  Print the noise errors from the last iteration of this experiment
  """
  noiseValues = ["0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3",
                            "0.35", "0.4", "0.45", "0.5"]
  print("\nNOISE CURVE ================",expPath,"=====================")
  try:
    result = suite.get_value(expPath, 0, noiseValues, "last")
    for k in noiseValues:
      print(k,result[k]["testerror"])
    pprint.pprint(result)
    print("totalCorrect:", suite.get_value(expPath, 0, "totalCorrect", "last"))
  except:
    print("Couldn't load experiment",expPath)


def learningCurve(expPath, suite):
  """
  Print the test and overall noise errors from each iteration of this experiment
  """
  print("\nLEARNING CURVE ================",expPath,"=====================")
  try:
    result = suite.get_value(expPath, 0, ["testerror","totalCorrect","elapsedTime"], "all")
    print("i testerror totalCorrect elapsedTime")
    for i,v in enumerate(zip(result["testerror"],result["totalCorrect"],result["elapsedTime"])):
      print(i,v[0],v[1],int(v[2]))
  except:
    print("Couldn't load experiment",expPath)


# To run it from htmresearch top level:
# python projects/sdr_paper/pytorch_experiments/analyze_experiment.py -c projects/sdr_paper/pytorch_experiments/experiments.cfg

if __name__ == '__main__':

  suite = MNISTSparseExperiment()

  # model = torch.load("results/experimentQuick/k10.0/model.pt")
  # model.eval()
  # print(model.l1.weight.data)

  # List of all experiments
  # experiments = suite.get_exps("./results")
  # pprint.pprint(experiments)

  summarizeResults("./results", suite)

  for expName in [
    "./results/standardOneLayer",
    "./results/experiment28",
  ]:
    analyzeParameters(expName, suite)


  # Print details of the best ones so far

  expPath = "./results/standardOneLayer"
  lastNoiseCurve(expPath, suite)
  learningCurve(expPath, suite)

  expPath = "./results/bestSparseNet/boost_strength1.0k50.0n500.0"
  lastNoiseCurve(expPath, suite)
  learningCurve(expPath, suite)

