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

import argparse
import numpy
import simplejson
from textwrap import TextWrapper

from htmresearch.support.csv_helper import readCSV, mapLabelRefs
from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.frameworks.nlp.classify_document_fingerprint import (
  ClassificationModelDocumentFingerprint
)


wrapper = TextWrapper(width=100)


def runExperiment(args):
  (trainingData, testData, labelRefs, documentCategoryMap,
   documentTextMap) = readData(args)
  
  model = ClassificationModelDocumentFingerprint(
    verbosity=args.verbosity,
    retina=args.retina,
    numLabels=args.numLabels,
    k=1)
  model = trainModel(args, model, trainingData, labelRefs)
  model.save(args.modelDir)
  newmodel = ClassificationModel.load(args.modelDir)
  
  # Print profile information
  print
  newmodel.dumpProfile()
  
  print dir(newmodel)
  print dir(newmodel.getClassifier())
  print newmodel.getClassifier()._numPatterns
  


def trainModel(args, model, trainingData, labelRefs):
  """
  Train the given model on trainingData. Return the trained model instance.
  """

  print
  print "=======================Training model on sample text================"
  for recordNum, doc in enumerate(trainingData):
    docTokens = model.tokenize(doc[0])
    lastToken = len(docTokens) - 1
    if args.verbosity > 0:
      print
      print "Document=", recordNum, "id=", doc[2]
      print wrapper.fill(doc[0])
      print "tokens=",docTokens
      print "label=",labelRefs[doc[1][0]],"index=",doc[1]
    for i, token in enumerate(docTokens):
      # print "Training data: ", token, id, doc[1]
      model.trainText(token, labels=doc[1],
                      sequenceId=doc[2], reset=int(i==lastToken))

  return model


def readData(args):
  """
  Read data file, print out some statistics, and return various data structures

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

  # Populate trainingData, documentTextMap
  trainingData = []
  documentTextMap = {}
  counts = numpy.zeros(len(labelRefs))
  for document in dataDict.itervalues():
    try:
      docId = int(document[2])
    except:
      raise RuntimeError("docId "+str(docId)+" is not an integer")
    documentTextMap[docId] = document[0]
    categoryIndex = document[1][0]
    trainingData.append([document[0], [categoryIndex], docId])
    counts[categoryIndex] += 1

  # For each document, figure out which categories it belongs to
  documentCategoryMap = {}
  for doc in dataDict.iteritems():
    docId = doc[1][2]
    categoryIndex = doc[1][1][0]
    v = documentCategoryMap.get(docId, [])
    v.append(categoryIndex)
    documentCategoryMap[docId] = v

  print "Total number of unique documents",len(documentCategoryMap)
  print "Category counts: ",counts
  print "Categories in training/test data:", labelRefs

  return (trainingData, trainingData, labelRefs, documentCategoryMap,
          documentTextMap)


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    # description=helpStr
  )

  parser.add_argument("-m", "--modelName",
                      default="docfp",
                      type=str,
                      help="Name of model class. Options: [docfp]")
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
  # parser.add_argument("--textPreprocess",
  #                     action="store_true",
  #                     default=False,
  #                     help="Whether or not to use text preprocessing.")
  parser.add_argument("-v", "--verbosity",
                      default=2,
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
