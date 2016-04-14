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
Script to run the "simple queries" NLP models API test.

Example invocations:

python simple_query.py -m keywords --dataPath FILE --numLabels 3
python simple_query.py -m docfp --dataPath FILE -v 0
python simple_query.py -c ../data/network_configs/sensor_tm_knn.json \
  --dataPath FILE
  --retina en_associative_64_univ

"""
import argparse
import copy
import os

from textwrap import TextWrapper

from htmresearch.support.csv_helper import readDataAndReshuffle
from htmresearch.support.nlp_model_test_helpers import (
  executeModelLifecycle,
  htmConfigs,
  nlpModelAccuracies,
  nlpModelTypes,
  printSummary,
  testModel
)


wrapper = TextWrapper(width=80)



def queryModel(model, queryDocument, documentTextMap):
  """
  Assumes overlap distance metric.
  """

  print
  print "=================Querying model on a sample document================"
  print
  print "Query document:"
  print wrapper.fill(queryDocument)

  _, sortedIds, sortedDistances = model.inferDocument(
    queryDocument, returnDetailedResults=True, sortResults=True)

  print
  print "Here are some similar documents in order of similarity:"
  for i, docId in enumerate(sortedIds[:10]):
    print
    print "Document #{} ({} overlap):".format(docId, sortedDistances[i])
    print " ", wrapper.fill(documentTextMap[docId])

  print
  print "Here are some dissimilar documents in reverse order of similarity:"
  lastDocIndex = len(sortedIds)-1
  for i in xrange(lastDocIndex, lastDocIndex-10, -1):
    print
    print "Document #{} ({} overlap):".format(sortedIds[i], sortedDistances[i])
    print wrapper.fill(documentTextMap[sortedIds[i]])


def resultsCheck(modelName):
  print
  print "How are the query results?"

  try:
    expectation = nlpModelAccuracies["simple_queries"][modelName]
    print "We expect them to be", expectation
  except KeyError:
    print "No expectation for querying with {}.".format(modelName)


def run(args):
  """ Run the 'query' test.
  This method handles scenarios for running a single model or all of them.
  """
  (trainingData, labelRefs, _, documentTextMap) = readDataAndReshuffle(args)

  if args.modelName == "all":
    modelNames = nlpModelTypes
    runningAllModels = True
  else:
    modelNames = [args.modelName]
    runningAllModels = False

  accuracies = {}
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

    # Create a model, train it, save it, reload it
    _, model = executeModelLifecycle(args, trainingData, labelRefs)

    # Now query the model using some example HR complaints about managers
    queryModel(model,
               "Begin by treating the employees of the department with the "
               "respect they deserve. Halt the unfair practices "
               "that they are aware of doing. There is no compassion "
               "or loyalty to its senior employees",
               documentTextMap)

    queryModel(model,
               "My manager is really incompetent. He has no clue how to "
               "properly supervise his employees and keep them motivated.",
               documentTextMap)

    queryModel(model,
               "I wish I had a lot more vacation and much more flexibility "
               "in how I manage my own time. I should be able to choose "
               "when I come in as long as I manage to get all my tasks done.",
               documentTextMap)

    if args.verbosity > 0:
      # Print profile information
      print
      model.dumpProfile()

  resultsCheck(args.modelName)



if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description=helpStr
  )

  parser.add_argument("-c", "--networkConfigPath",
                      default="../data/network_configs/sensor_knn.json",
                      help="Path to JSON specifying the network params.",
                      type=str)
  parser.add_argument("-m", "--modelName",
                      default="htm",
                      type=str,
                      help="Name of model class. Options: [keywords,htm]")
  parser.add_argument("--retinaScaling",
                      default=1.0,
                      type=float,
                      help="Factor by which to scale the Cortical.io retina.")
  parser.add_argument("--retina",
                      default="en_associative_64_univ",
                      type=str,
                      help="Name of Cortical.io retina.")
  parser.add_argument("--apiKey",
                      default=None,
                      type=str,
                      help="Key for Cortical.io API. If not specified will "
                      "use the environment variable CORTICAL_API_KEY.")
  parser.add_argument("--numLabels",
                      default=10,
                      type=int,
                      help="Number of unique labels to train on.")
  parser.add_argument("--dataPath",
                      default="",
                      help="CSV file containing labeled dataset")
  parser.add_argument("--textPreprocess",
                      action="store_true",
                      default=False,
                      help="Whether or not to use text preprocessing.")
  parser.add_argument("--experimentDir",
                      default="simple_queries",
                      help="Models will be saved in this directory.")
  parser.add_argument("-v", "--verbosity",
                      default=1,
                      type=int,
                      help="verbosity 0 will print out experiment steps, "
                           "verbosity 1 will include train and test data.")
  args = parser.parse_args()

  if not os.path.isfile(args.dataPath):
    raise RuntimeError("Need a data file for this experiment!")

  model = run(args)
