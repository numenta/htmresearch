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
Simple script that explains how to run classification models.

Example invocations:

python hello_classification_model.py -m keywords
python hello_classification_model.py -m docfp -v 0
python hello_classification_model.py -m htm \
  -c ../data/network_configs/sensor_tm_knn.json \
  --retina en_associative_64_univ

"""
import argparse
import copy
import os

from htmresearch.support.nlp_model_test_helpers import (
  executeModelLifecycle,
  htmConfigs,
  nlpModelTypes,
  printSummary,
  testModel
)



# Training data we will feed the model. There are two categories here that can
# be discriminated using bag of words
trainingData = [
  ["fox eats carrots", [0], 0],
  ["fox eats broccoli", [0], 1],
  ["fox eats lettuce", [0], 2],
  ["fox eats peppers", [0], 3],
  ["carrots are healthy", [1], 4],
  ["broccoli is healthy", [1], 5],
  ["lettuce is healthy", [1], 6],
  ["peppers is healthy", [1], 7],
]
labelRefs = ["fox eats", "vegetables"]

# Test data will be a copy of training data plus two additional documents.
# The first is an incorrectly labeled version of a training sample.
# The second is semantically similar to one of thr training samples.
# Expected classification error using CIO encoder is 9 out of 10 = 90%
# Expected classification error using keywords encoder is 8 out of 10 = 80%
testData = copy.deepcopy(trainingData)
testData.append(["fox eats carrots", [1], 8])     # Should get this wrong
testData.append(["wolf consumes salad", [0], 9])  # CIO models should get this
documentCategoryMap = {data[2]: data[1] for data in testData}


def run(args):
  """ Run the 'hello' test.
  This method handles scenarios for running a single model or all of them.
  Also tests serialization by checking the a model's results match before and
  after saving/loading.
  """
  # Two categories for the data
  args.numLabels = 2

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
    model, newModel = executeModelLifecycle(args, trainingData, labelRefs)

    # Test the model
    accuracyPct = testModel(
      model, testData, labelRefs, documentCategoryMap, args.verbosity)
    accuracies.update({name:accuracyPct})

    # Validate serialization
    print
    print "Testing serialization for {}...".format(args.modelName)
    newAccuracyPct = testModel(
      newModel, testData, labelRefs, documentCategoryMap, args.verbosity)
    if accuracyPct == newAccuracyPct:
      print "Serialization validated."
    else:
      print ("Inconsistent results before ({}) and after ({}) saving/loading "
             "the model!".format(accuracyPct, newAccuracyPct))

  printSummary("hello", accuracies)



if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description=helpStr
  )

  parser.add_argument("-c", "--networkConfigPath",
                      default="../data/network_configs/sensor_simple_tp_knn.json",
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
  parser.add_argument("--experimentDir",
                      default="hello_classification",
                      help="Models will be saved in this directory.")
  parser.add_argument("-v", "--verbosity",
                      default=1,
                      type=int,
                      help="verbosity 0 will print out experiment steps, "
                           "verbosity 1 will include train and test data.")
  args = parser.parse_args()

  run(args)
