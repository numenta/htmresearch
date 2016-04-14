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
Script to run the NLP classification models API tests.

To run the "hello classification" test, specify hello at the cmd line.

To run the "simple labels" test, specify the dataPath.

Example invocations:

python simple_labels.py -m keywords --hello
python simple_labels.py -m docfp --dataPath FILE --split None
python simple_labels.py --dataPath FILE \
  -m htm \
  -c ../data/network_configs/sensor_knn.json \
  --retina en_associative_64_univ

"""
import argparse
import copy
import os

from htmresearch.support.csv_helper import readDataAndReshuffle
from htmresearch.support.nlp_model_test_helpers import (
  assertResults,
  executeModelLifecycle,
  htmConfigs,
  nlpModelTypes,
  printSummary,
  testModel
)



def run(args):
  """ Run the classification test.
  This method handles scenarios for running a single model or all of them.
  Also tests serialization by checking the a model's results match before and
  after saving/loading.
  """
  if args.hello:
    args = _setupHelloTest(args)

  (dataset, labelRefs, documentCategoryMap, _) = readDataAndReshuffle(args)

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
    args.modelDir = os.path.join(args.experimentName, name)
    if runningAllModels and name == "htm":
      # Need to specify network config for htm models
      try:
        htmModelInfo = htmConfigs.pop()
      except KeyError:
        print "Not enough HTM configs, so skipping the HTM model."
        continue
      name = htmModelInfo[0]
      args.networkConfigPath = htmModelInfo[1]

    # Split data for train/test (We still test on the training data!)
    if args.split:
      split = int(len(dataset) * args.split)
      trainingData = dataset[:split]
    else:
      trainingData = dataset

    # Create a model, train it, save it, reload it
    _, model = executeModelLifecycle(args, trainingData, labelRefs)

    # Test the model
    accuracies.update({name:testModel(model,
                                      dataset,
                                      labelRefs,
                                      documentCategoryMap,
                                      args.verbosity)})

    if args.verbosity > 0:
      # Print profile information
      print
      model.dumpProfile()

  printSummary(args.experimentName, accuracies)

  if args.hello:
    assertResults("hello_classification", accuracies)



def _setupHelloTest(args):
  """ Setup args specific to the 'hello classification' test."""
  args.dataPath = "../data/etc/hello_classification.csv"
  args.experimentName = "hello_classification"
  args.numLabels = 2
  args.split = 0.80

  return args



if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description=helpStr
  )

  parser.add_argument("--dataPath",
                      default="",
                      help="CSV file containing labeled dataset. Required if "
                           "not runnning 'hello classification' test.")
  parser.add_argument("--hello",
                      default=False,
                      action="store_true",
                      help="Run the 'hello classification' test.")
  parser.add_argument("--split",
                      default=None,
                      help="Portion [0,1] of data for training (test on all).")
  parser.add_argument("-c", "--networkConfigPath",
                      default="../data/network_configs/sensor_knn.json",
                      help="Path to JSON specifying the network params.",
                      type=str)
  parser.add_argument("-m", "--modelName",
                      default="htm",
                      type=str,
                      help="Name of model class. If 'all', the models "
                           "specified in nlpModelTypes will run.")
  parser.add_argument("--retinaScaling",
                      default=1.0,
                      type=float,
                      help="Factor by which to scale the Cortical.io retina.")
  parser.add_argument("--retina",
                      default="en_associative",
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
  parser.add_argument("--textPreprocess",
                      action="store_true",
                      default=False,
                      help="Whether or not to use text preprocessing.")
  parser.add_argument("--experimentName",
                      default="simple_labels",
                      help="Models will be saved in this directory.")
  parser.add_argument("-v", "--verbosity",
                      default=1,
                      type=int,
                      help="verbosity 0 will print out experiment steps, "
                           "verbosity 1 will include train and test data.")
  args = parser.parse_args()

  if not (os.path.isfile(args.dataPath) or args.hello):
    raise RuntimeError("Need a data file for this experiment!")

  run(args)
