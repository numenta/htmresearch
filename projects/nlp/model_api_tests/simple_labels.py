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
Script to run the "simple" NLP models API test.

Example invocations:

python simple_labels.py -m keywords --dataPath FILE --numLabels 3
python simple_labels.py -m docfp --dataPath FILE -v 0
python simple_labels.py -c ../data/network_configs/sensor_knn.json \
  --dataPath FILE
  --retina en_associative_64_univ

"""
import argparse
import copy
import os

from htmresearch.support.csv_helper import readDataAndReshuffle
from htmresearch.support.nlp_model_test_helpers import (
  executeModelLifecycle,
  htmConfigs,
  nlpModelTypes,
  printSummary,
  testModel
)



def run(args):
  """ Run the 'simple' test.
  This method handles scenarios for running a single model or all of them.
  """
  (trainingData, labelRefs, documentCategoryMap, _) = readDataAndReshuffle(
    args, categoriesInOrderOfInterest=[8,9,10,5,6,11,13,0,1,2,3,4,7,12,14])

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

    # Test the model on the training data
    accuracies.update({name:testModel(model,
                                      trainingData,
                                      labelRefs,
                                      documentCategoryMap,
                                      args.verbosity)})

    if args.verbosity > 0:
      # Print profile information
      print
      model.dumpProfile()

  printSummary("simple", accuracies)



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
                      help="Name of model class. If 'all', the models "
                           "specified in nlpModelTypes will run.")
  parser.add_argument("--retinaScaling",
                      default=1.0,
                      type=float,
                      help="Factor by which to scale the Cortical.io retina.")
  parser.add_argument("--retina",
                      default="en_synonymous",
                      type=str,
                      help="Name of Cortical.io retina.")
  parser.add_argument("--apiKey",
                      default=None,
                      type=str,
                      help="Key for Cortical.io API. If not specified will "
                      "use the environment variable CORTICAL_API_KEY.")
  parser.add_argument("--dataPath",
                      default="",
                      help="CSV file containing labeled dataset")
  parser.add_argument("--numLabels",
                      default=10,
                      type=int,
                      help="Number of unique labels to train on.")
  parser.add_argument("--textPreprocess",
                      action="store_true",
                      default=False,
                      help="Whether or not to use text preprocessing.")
  parser.add_argument("--experimentDir",
                      default="simple_labels",
                      help="Models will be saved in this directory.")
  parser.add_argument("-v", "--verbosity",
                      default=1,
                      type=int,
                      help="verbosity 0 will print out experiment steps, "
                           "verbosity 1 will include train and test data.")
  args = parser.parse_args()

  if not os.path.isfile(args.dataPath):
    raise RuntimeError("Need a data file for this experiment!")

  run(args)
