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


def analyzeResults(expName, suite):
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


def printExperimentSpecifics(expPath, suite):
  # List the noise values from the best one
  noiseValues = ["0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3",
                            "0.35", "0.4", "0.45", "0.5"]
  print("\n================",expPath,"=====================")
  try:
    result = suite.get_value(expPath, 0, noiseValues, "last")
    for k in noiseValues:
      print(k,result[k]["testerror"])
    pprint.pprint(result)
    print("totalCorrect:", suite.get_value(expPath, 0, "totalCorrect", "last"))
  except:
    print("Couldn't load experiment",expPath)


# Need to run it from htmresearch top level:
# python projects/sdr_paper/pytorch_experiments/analyze_experiment.py -c projects/sdr_paper/pytorch_experiments/experiments.cfg

if __name__ == '__main__':

  suite = MNISTSparseExperiment()

  # model = torch.load("results/experimentQuick/k10.0/model.pt")
  # model.eval()
  # print(model.l1.weight.data)

  # List of all experiments
  # experiments = suite.get_exps("./results")
  # pprint.pprint(experiments)

  analyzeResults("./results", suite)

  for expName in [
    # "./results/experiment1", "./results/experiment2",
    # "./results/experiment3", "./results/experiment4",
    # "./results/experimentTemp", "./results/experiment7",
    # "./results/experiment8", "./results/experiment10",
    # "./results/experiment11",
    # "./results/experiment12",
    # "./results/experiment14",
    # "./results/experiment15",
    # "./results/experiment17",
    # "./results/experiment19",
    # "./results/experiment20",
    # "./results/experiment21",
    # "./results/experiment23",
    # "./results/experiment23Best",
    # "./results/experiment23Best2",
    # "./results/experiment24",
    "./results/standardOneLayer",
    "./results/experiment28",
    # "./results/experiment29",
  ]:
    analyzeParameters(expName, suite)


  # Print details of the best ones so far

  expPath = "./results/standardOneLayer"
  printExperimentSpecifics(expPath, suite)

  expPath = "./results/bestSparseNet/boost_strength1.0k50.0n500.0"
  printExperimentSpecifics(expPath, suite)

  # expPath = "./results/experiment1/learning_rate0.040boost_strength0.0k100.0momentum0.250"
  # printExperimentSpecifics(expPath, suite)
  #
  # expPath = "./results/experiment7/weight_sparsity0.350learning_rate0.040boost_strength1.50k200.0momentum0.250"
  # printExperimentSpecifics(expPath, suite)
  #
  # expPath = "./results/experiment10/weight_sparsity0.40learning_rate0.040n500.0boost_strength1.0k50.0momentum0.250"
  # printExperimentSpecifics(expPath, suite)

  # expPath = "./results/experiment23/k_inference_factor2.0boost_strength_factor0.90learning_rate0.040batch_size4.0n500.0boost_strength1.0k50.0"
  # printExperimentSpecifics(expPath, suite)
  #
  # expPath = "./results/experiment23Best/k_inference_factor1.50boost_strength_factor0.90learning_rate0.040batch_size4.0n500.0boost_strength1.0k50.0"
  # printExperimentSpecifics(expPath, suite)
  #
  # expPath = "./results/experiment24/k_inference_factor1.50boost_strength_factor0.850learning_rate0.040batch_size4.0n1000.0boost_strength1.50k50.0"
  # printExperimentSpecifics(expPath, suite)
