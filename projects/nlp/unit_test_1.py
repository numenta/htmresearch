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
Simple script to run unit test 1:
  Train on the first sentence of each set, test on the rest.
"""

import argparse
import logging
import numpy
from textwrap import TextWrapper

from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.frameworks.nlp.model_factory import (
  createModel, getNetworkConfig)
from htmresearch.support.csv_helper import readDataAndReshuffle


wrapper = TextWrapper(width=85)


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


def _trainModel(args, model, trainingData, labelRefs):
  """
  Train the given model on trainingData. Return the trained model instance.
  """
  print
  print "======================Training model on sample text==================="
  if args.verbosity > 0:
    printTemplate = "{0:<85}|{1:<10}|{2:<5}"
    print printTemplate.format("Document", "Label", "ID")
  for (document, labels, docId) in trainingData:
    if args.verbosity > 0:
      print printTemplate.format(
        wrapper.fill(document), labelRefs[labels[0]], docId)
    model.trainDocument(document, labels, docId)

  return model


def _testModel(args, model, testData, labelRefs, documentCategoryMap):
  """
  Test the given model on testData, print out and return results metrics.

  For each data sample in testData the model infers the similarity to each other
  sample. From a list sorted most-to-least similar, we then get the ranks of the
  samples that share the same category as the inference sample. Ideally these
  ranks would be low. The returned metrics are the min, mean, and max ranks of
  the category samples.
  """
  print
  print "========================Testing on sample text========================"
  totalScore = 0
  for i, (document, labels, docId) in enumerate(testData):
    _, sortedIds, _ = model.inferDocument(
      document, returnDetailedResults=True, sortResults=True)

    # Compute the test metrics for this document
    expectedCategory = docId / 100
    ranks = numpy.array(
      [i for i, index in enumerate(sortedIds) if index/100 == expectedCategory])

    score = ranks.sum()

    if args.verbosity > 0:
      print
      print "Doc {}: {}".format(docId, wrapper.fill(document))
      print "Sum of ranks =", score
      print "Min, mean, max of ranks = {}, {}, {}".format(
        ranks.min(), ranks.mean(), ranks.max())

    totalScore += score

  print
  print
  print "Total score =", totalScore
  print "Avg. score per sample =", float(totalScore) / i

  return totalScore


def runExperiment(args):
  """
  Create model according to args, train on training data, save model,
  restore model, test on test data.
  """
  (dataSet, labelRefs, documentCategoryMap,
   documentTextMap) = readDataAndReshuffle(args)

  # Train only on the first document of each set
  trainingData = [x for x in dataSet if x[2]%100==0]
  testData = [x for x in dataSet if x[2]%100!=0]

  print "Num training",len(trainingData),"num testing",len(testData)

  # Create a model, train it, save it, reload it, test it
  model = instantiateModel(args)
  model = _trainModel(args, model, dataSet, labelRefs)
  model.save(args.modelDir)
  newmodel = ClassificationModel.load(args.modelDir)

  testScore = _testModel(
    args, newmodel, dataSet, labelRefs, documentCategoryMap)

  return model, testScore



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
                      help="Maximum sparsity of Cio encodings.")
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
