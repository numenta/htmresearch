"""
Demo script for running Imbu with nupic.fluent models. In future versions, Imbu
will only interact with the ClassificationModel base class via a pickled model
file.
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
from htmresearch.support.csv_helper import readCSV

try:
  import simplejson as json
except ImportError:
  import json


_MODEL_MAPPING = {
  "CioWordFingerprint": ClassificationModelFingerprint,
  "CioDocumentFingerprint": ClassificationModelFingerprint,
  "Keywords": ClassificationModelKeywords,
  "HTMNetwork": ClassificationModelHTM,
}
_NETWORK_JSON = "/Users/alavin/nta/nupic.research/projects/nlp/data/network_configs/imbu.json"



def loadJSON(jsonPath):
  try:
    with open(jsonPath, "rb") as fin:
      return json.load(fin)
  except IOError as e:
    print "Could not find JSON at \'{}\'.".format(jsonPath)
    raise e


def loadModel(modelPath):
  """Load a serialized model."""
  try:
    with open(modelPath, "rb") as f:
      model = pkl.load(f)
    print "Model loaded from \'{}\'.".format(modelPath)
    return model
  except IOError as e:
    print "Could not load model from \'{}\'.".format(modelPath)
    raise e


def _createModel(modelName, savePath, **htmArgs):
  """Return an instantiated model."""
  modelCls = _MODEL_MAPPING.get(modelName, None)

  if modelCls is None:
    raise ValueError("Could not instantiate model \'{}\'.".format(modelName))

  # TODO: remove these if blocks and just use the else; either specify the Cio
  # FP type elsewhere, or split Word and Doc into separate classes.

  if modelName == "CioWordFingerprint":
    model = modelCls(fingerprintType=EncoderTypes.word)

  elif modelName == "CioDocumentFingerprint":
    model =  modelCls(fingerprintType=EncoderTypes.document)

  elif modelName == "HTMNetwork":
    model = modelCls(**htmArgs)

  else:
    model = modelCls()

  model.verbosity = 0
  model.numLabels = 0
  if savePath: model.modelDir = savePath

  return model


def run(args):

  if args.loadPath:
    model = loadModel(args.loadPath)
  elif args.modelName == "HTMNetwork":
    networkConfig = loadJSON(_NETWORK_JSON)
    
    print "Creating the network model..."
    model = _createModel(modelName=args.modelName, savePath=args.savePath,
      networkConfig=networkConfig, inputFilePath=args.dataPath, prepData=True,
      numLabels=0, stripCats=True)
    
    numRecords = sum(model.networkDataGen.getNumberOfTokens(model.networkDataPath))

    print "Training the model..."
    model.trainModel(iterations=numRecords)

  else:
    model = _createModel(modelName=args.modelName, savePath=args.savePath)

    dataDict = readCSV(args.dataPath, numLabels=0)

    print "Preparing and encoding the data..."
    samples = model.prepData(dataDict, args.preprocess)
    patterns = model.encodeSamples(samples)

    print "Training the model..."
    for i in xrange(len(samples)):
      model.trainModel(i)

  if args.savePath:
    model.saveModel()

  # Query the model. This is only for debugging; the Imbu app will query
  # directly from the saved model file.
  printTemplate = "{0:<10}|{1:<30}"
  while 1<2:
    print "Now we query the model for samples (quit with \'q\')..."
    input = raw_input("Enter a query: ")
    if input == "q": break
    sortedDistances = model.queryModel(input, args.preprocess)
    print printTemplate.format("Sample ID", "Distance from query")
    for sID, dist in sortedDistances:
      print printTemplate.format(sID, dist)
  return


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("dataPath",
                      help="Path to data CSV; samples must be in column w/ "
                           "header \'Sample\', see readCSV() for more.")
  parser.add_argument("-m", "--modelName",
                      default="CioWordFingerprint",
#                      default="CioDocumentFingerprint",
#                      default="Keywords",
#                      default="HTMNetwork",
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
