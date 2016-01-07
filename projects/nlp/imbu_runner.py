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
"""
Demo script for running Imbu with nupic.research NLP classification models.
"""

import argparse
import numpy
import os

from collections import OrderedDict

from htmresearch.encoders import EncoderTypes
from htmresearch.frameworks.nlp.classify_fingerprint import (
  ClassificationModelFingerprint)
from htmresearch.frameworks.nlp.classify_htm import ClassificationModelHTM
from htmresearch.frameworks.nlp.classify_keywords import (
  ClassificationModelKeywords)
from htmresearch.frameworks.nlp.classify_windows import (
  ClassificationModelWindows)
from htmresearch.support.csv_helper import readCSV
from htmresearch.support.text_preprocess import TextPreprocess

import simplejson as json


_MODEL_MAPPING = {
  "CioWordFingerprint": ClassificationModelFingerprint,
  "CioDocumentFingerprint": ClassificationModelFingerprint,
  "CioWindows": ClassificationModelWindows,
  "Keywords": ClassificationModelKeywords,
  "HTMNetwork": ClassificationModelHTM,
}
_MODEL_CLASSIFIER_METRIC = "pctOverlapOfInput"

root = os.path.dirname(os.path.realpath(__file__))
_NETWORK_JSON = os.path.join(root, "data/network_configs/imbu.json")



def loadJSON(jsonPath):
  try:
    with open(jsonPath, "rb") as fin:
      return json.load(fin)
  except IOError as e:
    print "Could not find JSON at '{}'.".format(jsonPath)
    raise e



def loadModel(modelPath):
  """Load a serialized model."""
  try:
    with open(modelPath, "rb") as f:
      model = pkl.load(f)
    print "Model loaded from '{}'.".format(modelPath)
    return model
  except IOError as e:
    print "Could not load model from '{}'.".format(modelPath)
    raise e



def _createModel(modelName, savePath, **htmArgs):
  """Return an instantiated model."""
  modelCls = _MODEL_MAPPING.get(modelName, None)

  if modelCls is None:
    raise ValueError("Could not instantiate model '{}'.".format(modelName))

  if modelName == "CioWordFingerprint":
    model = modelCls(
      fingerprintType=EncoderTypes.word,
      classifierMetric=_MODEL_CLASSIFIER_METRIC)

  elif modelName == "CioDocumentFingerprint":
    model =  modelCls(
      fingerprintType=EncoderTypes.document,
      classifierMetric=_MODEL_CLASSIFIER_METRIC)

  elif modelName == "HTMNetwork":
    model = modelCls(**htmArgs)

  else:
    model = modelCls(classifierMetric=_MODEL_CLASSIFIER_METRIC)

  model.verbosity = 0
  model.numLabels = 0
  if savePath:
    model.modelDir = savePath

  return model



def trainModel(model, trainingData):
  """
  Train the given model on trainingData.
  """
  TP = TextPreprocess()
  for text, _, uniqueID in trainingData.values():
    textTokens = TP.tokenize(text)
    lastToken = len(textTokens) - 1
    for i, token in enumerate(textTokens):
      # use the sequence's ID as the category label
      model.trainText(token,
                      [int(uniqueID)],
                      sequenceId=int(uniqueID),
                      reset=int(i==lastToken))



def run(args):

  print "Getting and prepping the data..."
  dataDict = readCSV(args.dataPath, numLabels=0)

  if args.loadPath:
    model = loadModel(args.loadPath)
  
  elif args.modelName == "HTMNetwork":
    networkConfig = loadJSON(_NETWORK_JSON)
    
    print "Creating the network model..."
    model = _createModel(
      modelName=args.modelName,
      savePath=args.savePath,
      networkConfig=networkConfig,
      inputFilePath=None,
      prepData=False,
      numLabels=0,
      retinaScaling=1.0)

  else:
    model = _createModel(modelName=args.modelName, savePath=args.savePath)

  print "Training the model (and encoding the data)..."
  trainModel(model, dataDict)


  if args.savePath:
    model.saveModel()

  # Query the model.
  printTemplate = "{0:<10}|{1:<30}"
  while 1<2:
    print "Now we query the model for samples (quit with 'q')..."
    input = raw_input("Enter a query: ")
    if input == "q": break
    sortedDistances = model.queryModel(input)
    print printTemplate.format("Sample ID", "Distance from query")
    for sID, dist in sortedDistances:
      print printTemplate.format(sID, dist)
  return



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("dataPath",
                      help="Path to data CSV; samples must be in column w/ "
                           "header 'Sample'; see readCSV() for more.")
  parser.add_argument("-m", "--modelName",
                      default="HTMNetwork",
                      type=str,
                      help="Name of model class. Also used for model results "
                           "directory and pickle checkpoint.")
  parser.add_argument("--loadPath",
                      default="",
                      type=str,
                      help="Path from which to load a serialized model.")
  parser.add_argument("--savePath",
                      default="",
                      type=str,
                      help="Path to save the serialized model.")
  parser.add_argument("--preprocess",
                      action="store_true",
                      default=False,
                      help="Whether or not to use text preprocessing.")
  args = parser.parse_args()
  
  run(args)
