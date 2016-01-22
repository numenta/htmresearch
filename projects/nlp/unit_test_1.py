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
Simple script to run unit test 1
"""

import argparse
import numpy
from textwrap import TextWrapper

from htmresearch.support.csv_helper import readDataAndReshuffle
from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.frameworks.nlp.model_factory import (
  createModel, getNetworkConfig)


wrapper = TextWrapper(width=100)

def instantiateModel(args):
  """
  Return an instance of the model we will use.
  """

  # Some values of K we know work well for this problem for specific model types
  kValues = { "keywords": 21 }

  # Create model after setting specific arguments required for this experiment
  args.networkConfig = getNetworkConfig(args.networkConfigPath)
  args.k = kValues.get(args.modelName, 1)
  model = createModel(**vars(args))

  return model


def trainModel(args, model, trainingData, labelRefs):
  """
  Train the given model on trainingData. Return the trained model instance.
  """
  print
  print "=======================Training model on sample text================"
  for i, doc in enumerate(trainingData):
    document = doc[0]
    labels = doc[1]
    docId = doc[2]
    print
    print "Document=", document, "label=",labelRefs[doc[1][0]], "id=",docId,
    model.trainDocument(document, labels, docId)

  return model


def testModel(args, model, testData, labelRefs, documentCategoryMap):
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
  for recordNum, doc in enumerate(testData):
    document = doc[0]
    desiredLabels = doc[1]
    if args.verbosity > 0:
      print
      print wrapper.fill(document)
      print "desired category index:",desiredLabels,
      print ", label: ",labelRefs[doc[1][0]]

    categoryVotes, _, _ = model.inferDocument(document)

    if categoryVotes.sum() > 0:
      # We will count classification as correct if the best category is any
      # one of the categories associated with this docId
      docId = doc[2]
      if categoryVotes.argmax() in documentCategoryMap[docId]:
        numCorrect += 1
      elif args.verbosity > 0:
        print "INCORRECT!!!"
        print "Final classification for this doc:",categoryVotes.argmax(),
        print "Label: ",labelRefs[categoryVotes.argmax()]
        print "Labels associated: ", documentCategoryMap[docId]
    else:
      print "No classification possible for this doc"

  # Compute and print out percent accuracy
  print
  print
  print "Total correct =",numCorrect,"out of",len(testData),"documents"
  print "Accuracy =",(float(numCorrect*100.0)/len(testData)),"%"


def runExperiment(args):
  """
  Create model according to args, train on training data, save model,
  restore model, test on test data.
  """

  (dataSet, labelRefs, documentCategoryMap,
   documentTextMap) = readDataAndReshuffle(args)

  # Train only with documents whose id's are divisible by 100
  trainingData = [x for i,x in enumerate(dataSet) if x[2]%100==0]
  testData = [x for i,x in enumerate(dataSet) if x[2]%100!=0]

  print "Num training",len(trainingData),"num testing",len(testData)

  # Create model
  model = instantiateModel(args)

  model = trainModel(args, model, trainingData, labelRefs)
  model.save(args.modelDir)
  newmodel = ClassificationModel.load(args.modelDir)
  testModel(args, newmodel, trainingData, labelRefs, documentCategoryMap)
  testModel(args, newmodel, testData, labelRefs, documentCategoryMap)

  return model


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
  parser.add_argument("--retinaScaling",
                      default=1.0,
                      type=float,
                      help="Factor by which to scale the Cortical.io retina.")
  parser.add_argument("--maxSparsity",
                      default=1.0,
                      type=float,
                      help="Maximum sparsity of CIO encodings.")
  parser.add_argument("--numLabels",
                      default=6,
                      type=int,
                      help="Number of unique labels to train on.")
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
  parser.add_argument("--dataPath",
                      default="data/junit/unit_test_1.csv",
                      help="CSV file containing labeled dataset")
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

  model = runExperiment(args)
