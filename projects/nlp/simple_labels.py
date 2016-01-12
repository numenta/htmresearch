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
Simple script to run a labeled dataset.

Example invocations:

python simple_labels.py -m keywords --dataPath FILE --numLabels 3
python simple_labels.py -m docfp --dataPath FILE
python simple_labels.py -c data/network_configs/sensor_knn.json -m htm -v 2 --dataPath FILE
python simple_labels.py -c data/network_configs/tp_knn.json -m htm --dataPath FILE

"""

import argparse
import numpy
from textwrap import TextWrapper

from htmresearch.support.csv_helper import readCSV, mapLabelRefs
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
      if args.verbosity > 0:
        print "Final classification for this doc:",categoryVotes.argmax(),
        print "Label: ",labelRefs[categoryVotes.argmax()]
        print "Labels associated: ", documentCategoryMap[docId]
      if categoryVotes.argmax() in documentCategoryMap[docId]:
        numCorrect += 1
    else:
      print "No classification possible for this doc"

  # Compute and print out percent accuracy
  print
  print
  print "Total correct =",numCorrect,"out of",len(testData),"documents"
  print "Accuracy =",(float(numCorrect*100.0)/len(testData)),"%"


def readData(args):
  """
  Read data file, print out some statistics, and return various data structures.

  Returns the tuple:
    (training dataset, test dataset, labelRefs, documentCategoryMap,
     documentTextMap)

  Return format:
      dataset = [
        ["fox eats carrots", [0], docId],
        ["fox eats peppers", [0], docId],
        ["carrots are healthy", [1], docId],
        ["peppers is healthy", [1], docId],
      ]

      labelRefs = [Category0Name, Category1Name, ...]

      documentCategoryMap = {
        docId: [categoryIndex0, categoryIndex1, ...],
        docId: [categoryIndex0, categoryIndex1, ...],
                :
      }

      documentTextMap = {
        docId: documentText,
        docId: documentText,
                :
      }

labelId to text map, and docId to categories

  """
  # Read data
  dataDict = readCSV(args.dataPath, 1)
  labelRefs, dataDict = mapLabelRefs(dataDict)
  categoriesInOrderOfInterest=[8,9,10,5,6,11,13,0,1,2,3,4,7,
                               12,14][0:args.numLabels]

  # Select data based on categories of interest. Shift category indices down
  # so we go from 0 to numLabels-1
  trainingData = []
  documentTextMap = {}
  counts = numpy.zeros(len(labelRefs))
  for document in dataDict.itervalues():
    try:
      docId = int(document[2])
    except:
      raise RuntimeError("docId "+str(docId)+" is not an integer")
    oldCategoryIndex = document[1][0]
    documentTextMap[docId] = document[0]
    if oldCategoryIndex in categoriesInOrderOfInterest:
      newIndex = categoriesInOrderOfInterest.index(oldCategoryIndex)
      trainingData.append([document[0], [newIndex], docId])
      counts[newIndex] += 1

  # For each document, figure out which categories it belongs to
  # Include the shifted category index
  documentCategoryMap = {}
  for doc in dataDict.iteritems():
    docId = int(doc[1][2])
    oldCategoryIndex = doc[1][1][0]
    if oldCategoryIndex in categoriesInOrderOfInterest:
      newIndex = categoriesInOrderOfInterest.index(oldCategoryIndex)
      v = documentCategoryMap.get(docId, [])
      v.append(newIndex)
      documentCategoryMap[docId] = v

  labelRefs = [labelRefs[i] for i in categoriesInOrderOfInterest]
  print "Total number of unique documents",len(documentCategoryMap)
  print "Category counts: ",counts
  print "Categories in training/test data:", labelRefs

  return (trainingData, trainingData, labelRefs, documentCategoryMap,
          documentTextMap)


def runExperiment(args):
  """
  Create model according to args, train on training data, save model,
  restore model, test on test data.
  """

  (trainingData, testData, labelRefs, documentCategoryMap,
   documentTextMap) = readData(args)

  # Create model
  model = instantiateModel(args)

  # model = createModel(args)
  model = trainModel(args, model, trainingData, labelRefs)
  model.save(args.modelDir)
  newmodel = ClassificationModel.load(args.modelDir)
  testModel(args, newmodel, testData, labelRefs, documentCategoryMap)

  # Print profile information
  print
  model.dumpProfile()

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
  parser.add_argument("--numLabels",
                      default=3,
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
                      default=None,
                      help="CSV file containing labeled dataset")
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

  model = runExperiment(args)
