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
python hello_classification_model.py -m docfp
python hello_classification_model.py -m htm \
  -c data/network_configs/sensor_simple_tp_knn.json \
  --retina en_associative_64_univ
"""

import argparse

from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.frameworks.nlp.classify_htm import ClassificationModelHTM
from htmresearch.frameworks.nlp.model_factory import (
  createModel, getNetworkConfig)


# The dataset is 10 documents of three-word sentences, each classified
# into one of two categories. Bag of words (BOW) models should be able to
# discriminate them.
# The training set is the first 8 documents, and the full dataset will be used
# for test. Of two documents for test, the first is an incorrectly labeled
# version of a training sample, and the second is semantically similar to
# one of the training samples.
# Expected classification accuracy using Cio encoder is 9 out of 10 = 90%
# Expected classification accuracy using keywords encoder is 8 out of 10 = 80%
_DATASET = [
  [0, "fox eats carrots", [0]],
  [1, "fox eats broccoli", [0]],
  [2, "fox eats lettuce", [0]],
  [3, "fox eats peppers", [0]],
  [4, "carrots are healthy", [1]],
  [5, "broccoli is healthy", [1]],
  [6, "kale is healthy", [1]],
  [7, "peppers are healthy", [1]],
  [8, "fox eats carrots", [1]],    # Models should get this wrong
  [9, "wolf consumes salad", [0]]  # Cio models should get this correct
]



def instantiateModel(args):
  """
  Return an instance of the model we will use.
  """
  # Some values of K we know work well for this problem for specific model types
  kValues = { "keywords": 21, "docfp": 3 }

  # Create model after setting specific arguments required for this experiment
  args.networkConfig = getNetworkConfig(args.networkConfigPath)
  args.numLabels = 2
  args.k = kValues.get(args.modelName, 1)

  return createModel(**vars(args))



def trainModel(model, trainingData):
  """
  Train the given model on trainingData. Return the trained model instance.
  """
  # Do three passes if we're using a TM
  numPasses = 1
  if isinstance(model, ClassificationModelHTM) and (model.tmRegion is not None):
    numPasses = 3

  print
  print "==============Training the model, numPasses=",numPasses,"============="
  for i in xrange(numPasses):
    # Clear classifier - we only want it to store outputs from the last pass
    model.getClassifier().clear()
    for (docId, document, labels) in trainingData:
      print "{}. '{}' --> {}".format(docId, document, labels[0])
      model.trainDocument(document, labels, docId)

  return model



def testModel(model, testData):
  """
  Test the given model on testData, print out and return accuracy percentage.

  Accuracy is calculated as follows. Each token in a document votes for a single
  category; it's possible for a token to contribute no votes. The document is
  classified with the category that received the most votes. Note that it is
  possible for a document to receive no votes, in which case it is counted as a
  misclassification.
  """
  print
  print "==============Testing the model============="

  numCorrect = 0
  for (docId, document, labels) in testData:
    print "{}. '{}'".format(docId, document)
    categoryVotes, _, _ = model.inferDocument(document)

    if categoryVotes.sum() > 0:
      # We will count classification as correct if the best category is any
      # one of the categories associated with this document
      predicted = categoryVotes.argmax()
      if predicted in labels:
        numCorrect += 1
    else:
      # No classification possible for this doc
      predicted = None
    print "\tActual label:", labels[0]
    print "\tPredicted label:", predicted

  accuracyPct = numCorrect * 100.0 / len(testData)

  print
  print "Total correct =", numCorrect, "out of", len(testData), "documents"
  print "Accuracy =", accuracyPct, "%"

  return accuracyPct



def runExperiment(args):
  """
  Create model according to args, train on training data, save model,
  restore model, test on test data.
  """
  # Create model
  model = instantiateModel(args)

  # Train model on the first 80% of the dataset
  trainingSplit = int(len(_DATASET) * 0.80)
  model = trainModel(model, _DATASET[:trainingSplit])

  # Test model on the full dataset
  accuracyPct = testModel(model, _DATASET)

  # Validate serialization - testing after reloading should give same result
  model.save(args.modelDir)
  newModel = ClassificationModel.load(args.modelDir)
  print
  print "Testing serialization..."
  newAccuracyPct = testModel(newModel, _DATASET)
  if accuracyPct == newAccuracyPct:
    print "Serialization validated."
  else:
    print ("Inconsistent results before ({}) and after ({}) saving/loading "
           "the model!".format(accuracyPct, newAccuracyPct))



if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description=helpStr
  )

  parser.add_argument("-c", "--networkConfigPath",
                      default="data/network_configs/sensor_simple_tp_knn.json",
                      help="Path to JSON specifying the network params.",
                      type=str)
  parser.add_argument("-m", "--modelName",
                      default="htm",
                      type=str,
                      help="Name of model class. Options: [keywords,htm,docfp]")
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
  parser.add_argument("--modelDir",
                      default="MODELNAME.checkpoint",
                      help="Model will be saved in this directory.")

  args = parser.parse_args()

  # By default set checkpoint directory name based on model name
  if args.modelDir == "MODELNAME.checkpoint":
    args.modelDir = args.modelName + ".checkpoint"
    print "Save dir: ",args.modelDir

  runExperiment(args)
