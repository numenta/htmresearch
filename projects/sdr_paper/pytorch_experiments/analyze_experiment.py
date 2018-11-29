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
import torch

from htmresearch.frameworks.pytorch.mnist_sparse_experiment import \
  MNISTSparseExperiment

def analyzeParameters(expName, suite):
  print("\n================",expName,"=====================")
  expParams = suite.get_params(expName)
  pprint.pprint(expParams)

  if type(expParams["boost_strength"]) == list:
    for boost in expParams["boost_strength"]:
      # Retrieve the last totalCorrect from each experiment with boost 0
      # Print them sorted from best to worst
      values, params = suite.get_values_fix_params(
        expName, 0, "totalCorrect", "last", boost_strength=boost)
      v = np.array(values)
      print("Average with boost", boost, "=", v.mean())
      # sortedIndices = v.argsort()
      # for i in sortedIndices[::-1]:
      #   print(v[i],params[i]["name"])

  if type(expParams["k"]) == list:
    for b in expParams["k"]:
      # Retrieve the last totalCorrect from each experiment
      # Print them sorted from best to worst
      values, params = suite.get_values_fix_params(
        expName, 0, "totalCorrect", "last", k=b)
      v = np.array(values)
      print("Average with k", b, "=", v.mean())
      # sortedIndices = v.argsort()
      # for i in sortedIndices[::-1]:
      #   print(v[i],params[i]["name"])

  if type(expParams["learning_rate"]) == list:
    for b in expParams["learning_rate"]:
      # Retrieve the last totalCorrect from each experiment
      values, params = suite.get_values_fix_params(
        expName, 0, "totalCorrect", "last", learning_rate=b)
      v = np.array(values)
      print("Average with learning_rate", b, "=", v.mean())
      # Print them sorted from best to worst
      # sortedIndices = v.argsort()
      # for i in sortedIndices[::-1]:
      #   print(v[i],params[i]["name"])

  if type(expParams["weight_sparsity"]) == list:
    for b in expParams["weight_sparsity"]:
      # Retrieve the last totalCorrect from each experiment
      values, params = suite.get_values_fix_params(
        expName, 0, "totalCorrect", "last", weight_sparsity=b)
      v = np.array(values)
      print("Average with weight_sparsity", b, "=", v.mean())
      # Print them sorted from best to worst
      # sortedIndices = v.argsort()
      # for i in sortedIndices[::-1]:
      #   print(v[i],params[i]["name"])


def analyzeResults(expName, suite):
  print("\n================",expName,"=====================")

  # Retrieve the last totalCorrect from each experiment
  # Print them sorted from best to worst
  values, params = suite.get_values_fix_params(
    expName, 0, "totalCorrect", "last")
  v = np.array(values)
  sortedIndices = v.argsort()
  for i in sortedIndices[::-1]:
    print(v[i], params[i]["name"])

  print()


def printExperimentSpecifics(expPath, suite):
  # List the noise values from the best one
  noiseValues = ["0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3",
                            "0.35", "0.4", "0.45", "0.5"]
  print("\n================",expPath,"=====================")
  result = suite.get_value(expPath, 0, noiseValues, "last")
  for k in noiseValues:
    print(k,result[k]["testerror"])
  pprint.pprint(result)
  print("totalCorrect:", suite.get_value(expPath, 0, "totalCorrect", "last"))


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

  for expName in ["./results/experiment1", "./results/experiment2",
                  # "./results/experiment3", "./results/experiment4",
                  # "./results/experimentTemp", "./results/experiment7",
                  # "./results/experiment8", "./results/experiment10",
                  "./results/experiment11"]:
    analyzeParameters(expName, suite)


  # Print details of the best ones so far
  # expPath = "./results/experiment1/learning_rate0.040boost_strength0.0k100.0momentum0.250"
  # printExperimentSpecifics(expPath, suite)
  #
  # expPath = "./results/experiment7/weight_sparsity0.350learning_rate0.040boost_strength1.50k200.0momentum0.250"
  # printExperimentSpecifics(expPath, suite)
  #
  # expPath = "./results/experiment10/weight_sparsity0.40learning_rate0.040n500.0boost_strength1.0k50.0momentum0.250"
  # printExperimentSpecifics(expPath, suite)
