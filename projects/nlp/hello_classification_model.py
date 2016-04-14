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

The standard dataset is 10 documents of three-word sentences, each classified
into one of two categories. Bag of words (BOW) models should be able to
discriminate them.
Test data will be a copy of training data plus two additional documents. The
first is an incorrectly labeled version of a training sample. The second is
semantically similar to one of the training samples.
Expected classification accuracy using CIO encoder is 9 out of 10 = 90%
Expected classification accuracy using keywords encoder is 8 out of 10 = 80%

Example invocations:
python hello_classification_model.py -m keywords
python hello_classification_model.py -m docfp -v 0
python hello_classification_model.py -m htm \
  -c data/network_configs/sensor_simple_tp_knn.json \
  --retina en_associative_64_univ
"""

import argparse
import copy
from prettytable import PrettyTable
from textwrap import TextWrapper
from tqdm import tqdm

from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.frameworks.nlp.classify_htm import ClassificationModelHTM
from htmresearch.frameworks.nlp.model_factory import (
  createModel, getNetworkConfig)
from htmresearch.support.csv_helper import mapLabelRefs, readCSV


wrapper = TextWrapper(width=80)



def getData(dataPath):
  """
  Read in the dataset and return it as a list, where each item is a three tuple:
  (document string, labels array, ID string). Also return a list of the
  labels strings.
  """
  # Here numLabels specifies the number of category labels per document
  dataDict = readCSV(dataPath, numLabels=1)

  labelRefs, dataDict = mapLabelRefs(dataDict)

  dataset = [docInfo for docInfo in dataDict.values()]

  return dataset, labelRefs



def instantiateModel(args):
  """
  Return an instance of the model we will use.
  """
  # Some values of K we know work well for this problem for specific model types
  kValues = { "keywords": 21, "docfp": 3 }

  # Create model after setting specific arguments required for this experiment
  args.networkConfig = getNetworkConfig(args.networkConfigPath)
  args.k = kValues.get(args.modelName, 1)

  return createModel(**vars(args))



def trainModel(model, trainingData, labelRefs, verbosity=0):
  """
  Train the given model on trainingData. Return the trained model instance.
  """
  # Do three passes if we're using a TM
  numPasses = 1
  if isinstance(model, ClassificationModelHTM) and (model.tmRegion is not None):
    numPasses = 3

  modelName = repr(model).split()[0].split(".")[-1]
  print
  print "==============Training {} on sample text============".format(modelName)
  if verbosity > 0:
    printTemplate = PrettyTable(["ID", "Document", "Label"])
    printTemplate.align = "l"
    printTemplate.header_style = "upper"
  for _ in xrange(numPasses):
    for (document, labels, docId) in tqdm(trainingData):
      if verbosity > 0:
        docStr = unicode(document, errors="ignore")
        printTemplate.add_row([docId, wrapper.fill(docStr), labelRefs[labels[0]]])
      model.trainDocument(document, labels, docId)
    if verbosity > 0:
      print printTemplate

  return model



def testModel(model, testData, labelRefs, verbosity=0):
  """
  Test the given model on testData, print out and return accuracy percentage.

  Accuracy is calculated as follows. Each token in a document votes for a single
  category; it's possible for a token to contribute no votes. The document is
  classified with the category that received the most votes. Note that it is
  possible for a document to receive no votes, in which case it is counted as a
  misclassification.
  """
  modelName = repr(model).split()[0].split(".")[-1]
  print
  print "==============Testing {} on sample text=============".format(modelName)
  if verbosity > 0:
    print
    printTemplate = PrettyTable(
      ["ID", "Document", "Actual Label(s)", "Predicted Label"])
    printTemplate.align = "l"
    printTemplate.header_style = "upper"

  numCorrect = 0
  labelRefs.append("none")
  for (document, labels, docId) in tqdm(testData):

    categoryVotes, _, _ = model.inferDocument(document)

    if categoryVotes.sum() > 0:
      # We will count classification as correct if the best category is any
      # one of the categories associated with this docId
      predicted = categoryVotes.argmax()
      if predicted in labels:
        numCorrect += 1
    else:
      # No classification possible for this doc
      predicted = -1

    if verbosity > 0:
      docStr = unicode(document, errors="ignore")
      printTemplate.add_row(
        [docId,
         wrapper.fill(docStr),
         [labelRefs[l] for l in labels],
         labelRefs[predicted]]
      )

  accuracyPct = numCorrect * 100.0 / len(testData)

  if verbosity > 0:
    print printTemplate
  print
  print "Total correct =", numCorrect, "out of", len(testData), "documents"
  print "Accuracy =", accuracyPct, "%"

  return accuracyPct



def runExperiment(args):
  """
  Create model according to args, train on training data, save model,
  restore model, test on test data.
  """
  # Get data
  dataset, labelRefs = getData(args.dataPath)

  # Create model
  model = instantiateModel(args)

  # Train model on the first 80% of the dataset
  trainingSplit = int(len(dataset) * 0.80)
  model = trainModel(model, dataset[:trainingSplit], labelRefs, args.verbosity)

  # Test model
  accuracyPct = testModel(model, dataset, labelRefs, args.verbosity)

  # Validate serialization - testing after reloading should give same result as
  # above
  model.save(args.modelDir)
  newModel = ClassificationModel.load(args.modelDir)
  print
  print "Testing serialization for {}...".format(args.modelName)
  newAccuracyPct = testModel(newModel, dataset, labelRefs, args.verbosity)
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

  parser.add_argument("--dataPath",
                      default="data/etc/hello_classification.csv",
                      help="CSV file containing labeled dataset")
  parser.add_argument("--numLabels",
                      default=2,
                      type=int,
                      help="Number of unique labels to train on.")
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
  parser.add_argument("-v", "--verbosity",
                      default=1,
                      type=int,
                      help="verbosity 0 will print out experiment steps, "
                           "verbosity 1 will include train and test data.")
  args = parser.parse_args()

  # By default set checkpoint directory name based on model name
  if args.modelDir == "MODELNAME.checkpoint":
    args.modelDir = args.modelName + ".checkpoint"
    print "Save dir: ",args.modelDir

  runExperiment(args)
