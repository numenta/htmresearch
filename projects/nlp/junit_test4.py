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
  Simple script to run unit test 4.

  The dataset is categories defined by base sentences From each base sentence we
  create four short paragraphs that expand on that topic; paragraphs are 4-5
  sentences, don't contain the same words as in the topic, and try to avoid
  the same words as the other paragraphs. For the test we use the topic
  description as the search term. A perfect result ranks the category's four
  paragraphs as closest to the search.
"""

import argparse
import os

from htmresearch.support.junit_testing import (
  htmConfigs, nlpModelTypes, plotResults, printRankResults, setupExperiment,
  testModel,
)


# Dataset info
CATEGORY_SIZE = 5
NUMBER_OF_DOCS = 50



def runExperiment(args):
  """ Build a model and test it."""

  model, dataSet = setupExperiment(args)

  allRanks, avgRanks, avgStats = testModel(
    model,
    [d for d in dataSet if d[2]%100==0],
    categorySize=CATEGORY_SIZE,
    verbosity=args.verbosity
  )
  printRankResults("JUnit4", avgRanks, avgStats)

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
                testName="JUnit Test 4")



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
                      help="Name of model class. Options: [keywords,htm]")
  parser.add_argument("--experimentDir",
                      default="junit4_checkpoints",
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
  args.dataPath = "data/junit/unit_test_4.csv"

  run(args)
