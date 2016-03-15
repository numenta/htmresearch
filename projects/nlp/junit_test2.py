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
  Simple script to run unit test 2.

  The dataset is categories defined by base sentences (each with three
  words). For each base sentence there are four sentences created by
  successively adding one or more words. The longer sentences still contain the
  basic idea as in the base sentence. We run two variations of the same test on
  this data set: we use (a) the shortest sentences as search terms, and then (b)
  the longest. A perfect result ranks the four other sentences in the category
  as closest to the search term.
"""

import argparse
import os

from htmresearch.support.junit_testing import (
  htmConfigs, nlpModelTypes, plotResults, printRankResults, setupExperiment,
  testModel,
)


# Dataset info
CATEGORY_SIZE = 5
NUMBER_OF_DOCS = 80


def runExperiment(args, testIndex=0):
  """ Build a model and test it.
  @param testIndex (int) Specifies the doc of each group we use for inference.
  """
  model, dataSet = setupExperiment(args)

  allRanks, avgRanks, avgStats = testModel(
    model,
    [d for d in dataSet if d[2]%100==testIndex],
    categorySize=CATEGORY_SIZE,
    verbosity=args.verbosity
  )
  printRankResults("JUnit2", avgRanks, avgStats)

  return allRanks, avgRanks, avgStats


def run(args):
  """ Method to handle scenarios for running a single model or all of them."""
  if args.modelName == "all":
    modelNames = nlpModelTypes
    runningAllModels = True
  else:
    modelNames = [args.modelName]
    runningAllModels = False

  # Run both variations (a and b) of junit test 2
  test2Types = (("a", 0), ("b", 4))
  for testVariation, testIndex in test2Types:
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
          htmConfigs.add(htmModelInfo)  # TODO: replace this hack
        except KeyError:
          print "Not enough HTM configs, so skipping the HTM model."
          continue
        name = htmModelInfo[0]
        args.networkConfigPath = htmModelInfo[1]

      # Run the junit test, update metrics dicts
      ar, r, s = runExperiment(args, testIndex)
      allRanks.update({name:ar})
      ranks.update({name:r})
      stats.update({name:s})

    plotResults(
      allRanks, ranks, maxRank=NUMBER_OF_DOCS,
      testName="JUnit Test 2{}".format(testVariation))



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
                      default="junit2_checkpoints",
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
  args = parser.parse_args()

  # Default dataset for this unit test
  args.dataPath = "data/junit/unit_test_2.csv"

  run(args)
