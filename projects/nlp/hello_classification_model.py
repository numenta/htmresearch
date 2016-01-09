#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
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
python hello_classification_model.py -c data/network_configs/sensor_knn.json -m htm -v 2
python hello_classification_model.py -c data/network_configs/tp_knn.json -m htm

"""

import argparse
import simplejson
import numpy
import copy

from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.frameworks.nlp.classify_htm import ClassificationModelHTM
from htmresearch.frameworks.nlp.classify_document_fingerprint import (
  ClassificationModelDocumentFingerprint
)
from htmresearch.frameworks.nlp.classify_keywords import (
  ClassificationModelKeywords
)
from htmresearch.frameworks.nlp.classify_fingerprint import (
  ClassificationModelFingerprint
)

# Training data we will feed the model. There are two categories here that can
# be discriminated using bag of words
trainingData = [
  ["fox eats carrots", [0]],
  ["fox eats broccoli", [0]],
  ["fox eats lettuce", [0]],
  ["fox eats peppers", [0]],
  ["carrots are healthy", [1]],
  ["broccoli is healthy", [1]],
  ["lettuce is healthy", [1]],
  ["peppers is healthy", [1]],
]

# Test data will be a copy of training data plus two additional documents.
# The first is an incorrectly labeled version of a training sample.
# The second is semantically similar to one of thr training samples.
# Expected classification error using CIO encoder is 9 out of 10 = 90%
# Expected classification error using keywords encoder is 8 out of 10 = 80%
testData = copy.deepcopy(trainingData)
testData.append(["fox eats carrots", [1]])     # Should get this wrong
testData.append(["wolf consumes salad", [0]])  # CIO models should get this

def splitDocumentIntoTokens(document):
  """
  Given a document (set of words), return a list containing individual tokens
  to be fed into model.
  """
  return document.split()


def getNetworkConfig(networkConfigPath):
  """
  Given path to JSON model config file, return a dict.
  """
  try:
    with open(networkConfigPath, "rb") as fin:
      return simplejson.load(fin)
  except IOError as e:
    print "Could not find network configuration JSON at \'{}\'.".format(
      networkConfigPath)
    raise e


def createModel(args):
  """
  Return a classification model of the appropriate type. The model could be any
  supported subclass of ClassficationModel based on args.
  """
  if args.modelName == "htm":
    # Instantiate the HTM model
    model = ClassificationModelHTM(
      networkConfig=getNetworkConfig(args.networkConfigPath),
      inputFilePath=None,
      retina=args.retina,
      verbosity=args.verbosity,
      numLabels=2,
      prepData=False,
      modelDir="tempdir")

  elif args.modelName == "keywords":
    # Instantiate the keywords model
    model = ClassificationModelKeywords(
      verbosity=args.verbosity,
      numLabels=2,
      k=9,
      modelDir="tempdir")

  elif args.modelName == "docfp":
    # Instantiate the document fingerprint model
    model = ClassificationModelDocumentFingerprint(
      verbosity=args.verbosity,
      retina=args.retina,
      numLabels=2,
      k=3)

  elif args.modelName == "cioword":
    # Instantiate the Cio word fingerprint model
    model = ClassificationModelFingerprint(
      verbosity=args.verbosity,
      retina=args.retina,
      numLabels=2)

  else:
    raise RuntimeError("Unknown model type: " + args.modelName)

  return model


def trainModel(model, trainingData):
  """
  Train the given model on trainingData. Return the trained model instance.
  """

  print
  print "=======================Training model on sample text================"
  for docId, doc in enumerate(trainingData):
    document = doc[0]
    labels = doc[1]
    print
    print "Document=", document, "label=",doc[1], "id=",docId
    model.trainDocument(document, labels, docId)

  return model


def testModel(args, model, testData):
  """
  Test the given model on testData and print out accuracy.

  Accuracy is calculated as follows. Each token in a document votes for a single
  category. The document is classified with the category that received the most
  votes. Note that it is possible for a token and/or document to receive no
  votes, in which case it is counted as a misclassification.
  """
  print
  print "==========================Classifying sample text================"
  numCorrect = 0
  for docId, doc in enumerate(testData):
    document = doc[0]
    desiredLabels = doc[1]
    print
    print "Document=", document,", desired label: ",desiredLabels
    categoryVotes, _, _ = model.inferDocument(document)

    if categoryVotes.sum() > 0:
      print "Final classification for this doc:",categoryVotes.argmax()
      if categoryVotes.argmax() in desiredLabels:
        numCorrect += 1
    else:
      print "No classification possible for this doc"

  # Compute and print out percent accuracy
  print "Total correct =",numCorrect,"out of",len(testData),"documents"
  print "Accuracy =",(float(numCorrect*100.0)/len(testData)),"%"


def runExperiment(args, trainingData, testData):
  """
  Create model according to args, train on training data, save model,
  restore model, test on test data.
  """

  model = createModel(args)
  model = trainModel(model, trainingData)
  testModel(args, model, testData)

  # Test serialization - should give same result as above
  model.save(args.modelDir)
  newmodel = ClassificationModel.load(args.modelDir)
  print
  print "==========================Testing after de-serialization========"
  testModel(args, newmodel, testData)


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
                      help="Name of model class. Options: [keywords,htm,docfp]")
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
  parser.add_argument("--modelDir",
                      default="MODELNAME.checkpoint",
                      help="Model will be saved in this directory.")
  parser.add_argument("--textPreprocess",
                      action="store_true",
                      default=False,
                      help="Whether or not to use text preprocessing.")
  parser.add_argument("-v", "--verbosity",
                      default=1,
                      type=int,
                      help="verbosity 0 will print out experiment steps, "
                           "verbosity 1 will include results, and verbosity > "
                           "1 will print out preprocessed tokens and kNN "
                           "inference metrics.")
  args = parser.parse_args()

  # By default set checkpoint directory name based on model name
  if args.modelDir == "MODELNAME.checkpoint":
    args.modelDir = args.modelName + ".checkpoint"
    print "Save dir: ",args.modelDir

  runExperiment(args, trainingData, testData)
