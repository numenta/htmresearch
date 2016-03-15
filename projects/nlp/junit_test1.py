#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
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

helpStr = """
  Simple script to run unit test 1:

  The dataset is categories defined by base sentences (each with ten words).
  For each base sentence there are five new sentences each, with an additional
  word substitution that does not change the meaning of the sentence. For the
  test we use each of the sentences as a search term. A perfect result ranks the
  four similar sentences closest to the search.
"""

import argparse
import os

from htmresearch.support.junit_testing import (
  htmConfigs, nlpModelTypes, plotResults, printRankResults, setupExperiment,
  testModel,
)


# Dataset info
CATEGORY_SIZE = 6
NUMBER_OF_DOCS = 132


def runExperiment(args):
  """ Build a model, test it, print and return results metrics."""
  model, dataSet = setupExperiment(args)

  allRanks, avgRanks, avgStats = testModel(model,
                           dataSet,
                           categorySize=CATEGORY_SIZE,
                           verbosity=args.verbosity)
  printRankResults("JUnit1", avgRanks, avgStats)

  return allRanks, avgRanks, avgStats


def run(args):
  """ Method to handle scenarios for running a single model or all of them."""
  if args.modelName == "all":
    modelNames = nlpModelTypes
    runningAllModels = True
  else:
    modelNames = [args.modelName]
    runningAllModels = False

  allRanks = {}
  ranks = {}
  stats = {}
  for name in modelNames:
    # Setup args
    args.modelName = name
    args.modelDir = os.path.join(args.experimentDir, name)
    if runningAllModels and name == "htm":
      # Need to specify network config for htm models
      try:
        htmModelInfo = htmConfigs.pop()
      except KeyError:
        print "Not enough HTM configs, so skipping the HTM model."
        continue
      name = htmModelInfo[0]
      args.networkConfigPath = htmModelInfo[1]

    # Run the junit test, update metrics dicts
    ar, r, s = runExperiment(args)
    allRanks.update({name:ar})
    ranks.update({name:r})
    stats.update({name:s})

  if args.plot:
    plotResults(allRanks, ranks, maxRank=NUMBER_OF_DOCS,
                testName="JUnit Test 1")



if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description=helpStr
  )

  parser.add_argument("-c", "--networkConfigPath",
                      default="data/network_configs/sensor_knn.json",
                      help="Path to JSON specifying the network params.",
                      type=str)
  parser.add_argument("-m", "--modelName",
                      default="htm",
                      type=str,
                      help="Name of model class. If 'all', the models "
                           "specified in nlpModelTypes will run.")
  parser.add_argument("--experimentDir",
                      default="junit1_checkpoints",
                      help="Model(s) will be saved in this directory.")
  parser.add_argument("--retina",
                      default="en_associative_64_univ",
                      type=str,
                      help="Name of Cortical.io retina.")
  parser.add_argument("--apiKey",
                      default=None,
                      type=str,
                      help="Key for Cortical.io API. If not specified will "
                      "use the environment variable CORTICAL_API_KEY.")
  parser.add_argument("-v", "--verbosity",
                      default=1,
                      type=int,
                      help="verbosity 0 will print out experiment steps, "
                           "verbosity 1 will include train and test data.")
  parser.add_argument("--plot",
                      action="store_true",
                      default=False,
                      help="If true will generate plotly Plots.")
  args = parser.parse_args()

  # Default dataset for this unit test
  args.dataPath = "data/junit/unit_test_1.csv"

  run(args)
