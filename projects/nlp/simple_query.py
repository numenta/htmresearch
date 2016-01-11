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
Simple script to run a labeled dataset.

Example invocations:

python simple_labels.py -m keywords --dataPath FILE --numLabels 3
python simple_labels.py -m docfp --dataPath FILE
python simple_labels.py -c data/network_configs/sensor_knn.json -m htm -v 2 --dataPath FILE
python simple_labels.py -c data/network_configs/tp_knn.json -m htm --dataPath FILE

"""

import argparse
import numpy
import simplejson
from textwrap import TextWrapper

from htmresearch.encoders import EncoderTypes
from htmresearch.support.csv_helper import readCSV, mapLabelRefs
from htmresearch.frameworks.nlp.classify_htm import ClassificationModelHTM
from htmresearch.frameworks.nlp.classify_document_fingerprint import (
  ClassificationModelDocumentFingerprint
)
from htmresearch.frameworks.nlp.classify_fingerprint import (
  ClassificationModelFingerprint
)
from htmresearch.frameworks.nlp.classify_keywords import (
  ClassificationModelKeywords
)


wrapper = TextWrapper(width=100)


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
      retina=args.retina,
      verbosity=args.verbosity,
      numLabels=args.numLabels)

  elif args.modelName == "keywords":
    # Instantiate the keywords model
    model = ClassificationModelKeywords(
      verbosity=args.verbosity,
      numLabels=args.numLabels,
      k=21)

  elif args.modelName == "docfp":
    # Instantiate the document fingerprint model
    model = ClassificationModelDocumentFingerprint(
      verbosity=args.verbosity,
      retina=args.retina,
      numLabels=args.numLabels,
      k=1)

  elif args.modelName == "cioword":
    # Instantiate the Cio word fingerprint model
    model = ClassificationModelFingerprint(
      verbosity=args.verbosity,
      retina=args.retina,
      numLabels=args.numLabels,
      fingerprintType=EncoderTypes.word)

  else:
    raise RuntimeError("Unknown model type: " + args.modelName)

  return model


def trainModel(args, model, trainingData, labelRefs):
  """
  Train the given model on trainingData. Return the trained model instance.
  """
  print
  print "=======================Training model on sample text================"
  for recordNum, doc in enumerate(trainingData):
    document = doc[0]
    labels = doc[1]
    docId = doc[2]
    if args.verbosity > 0:
      print
      print "Document=", wrapper.fill(document)
      print "label=",labelRefs[labels[0]], "id=",docId
    model.trainDocument(document, labels, docId)

  return model


def queryModel(model, queryDocument, documentTextMap,
               labelRefs, documentCategoryMap):
  """
  Demonstrates how querying might be done with the new partitionId scheme. The
  code below assumes a document level classifier, so not appropriate for all
  model types. The implementation should be cleaned up and moved into the
  model, but this provides a basic idea.
  """

  print
  print "=================Querying model on a sample document================"
  print
  print "Query document:"
  print wrapper.fill(queryDocument)
  print

  categoryVotes, idList, distances = model.inferDocument(
                                                queryDocument,
                                                returnDetailedResults=True,
                                                sortResults=True)

  print "Here are some similar documents in order of similarity"
  for i, docId in enumerate(idList[0:10]):
    print distances[i], docId
    print "document=",wrapper.fill(documentTextMap[docId])
    print "Categories=",documentCategoryMap[str(docId)]
    print

  print "Here are some dissimilar documents in reverse order of similarity"
  lastDocIndex = len(idList)-1
  for i in range(lastDocIndex, lastDocIndex-10, -1):
    print distances[i], idList[i]
    print "document=",wrapper.fill(documentTextMap[idList[i]])
    print


def readData(args, categoriesInOrderOfInterest=None):
  """
  Read data file, print out some statistics, and return various data structures

  categoriesInOrderOfInterest (list) Optional list of integers representing
                                     the priority order of various categories

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
  if categoriesInOrderOfInterest is None:
      categoriesInOrderOfInterest = range(0,args.numLabels)
  else:
    categoriesInOrderOfInterest=categoriesInOrderOfInterest[0:args.numLabels]

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
    docId = doc[1][2]
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
   documentTextMap) = readData(args,
                         [8,9,10,5,6,11,13,0,1,2,3,4,7,12,14])

  model = createModel(args)
  model = trainModel(args, model, trainingData, labelRefs)

  print labelRefs

  # Now query the model using some example HR complaints about managers
  queryModel(model,
             "Begin by treating the employees of the department with the "
             "respect they deserve. Halt the unfair practices "
             "that they are aware of doing. There is no compassion "
             "or loyalty to its senior employees",
             documentTextMap, labelRefs, documentCategoryMap,
             )

  queryModel(model,
             "My manager is really incompetent. He has no clue how to "
             "properly supervise his employees and keep them motivated.",
             documentTextMap, labelRefs, documentCategoryMap,
             )

  queryModel(model,
             "I wish I had a lot more vacation and much more flexibility "
             "in how I manage my own time. I should be able to choose "
             "when I come in as long as I manage to get all my tasks done.",
             documentTextMap, labelRefs, documentCategoryMap,
             )

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
