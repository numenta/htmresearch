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
  Methods to run unit tests.
"""

import itertools
import numpy
from prettytable import PrettyTable
from scipy.stats import skew

from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.frameworks.nlp.model_factory import (
  createModel, getNetworkConfig)
from htmresearch.support.csv_helper import readDataAndReshuffle


# Some values of K we know work well for this problem for specific model types
kValues = { "keywords": 21 }


def instantiateModel(args):
  """
  Return an instance of the model we will use.
  """
  # Create model after setting specific arguments required for this experiment
  args.networkConfig = getNetworkConfig(args.networkConfigPath)
  args.k = kValues.get(args.modelName, 1)
  model = createModel(**vars(args))

  return model


def trainModel(model, trainingData, labelRefs, verbosity=0):
  """
  Train the given model on trainingData. Return the trained model instance.
  """
  print
  print "======================Training model on sample text==================="
  if verbosity > 0:
    printTemplate = PrettyTable(["ID", "Document", "Label"])
    printTemplate.align = "l"
    printTemplate.header_style = "upper"
  for (document, labels, docId) in trainingData:
    if verbosity > 0:
      printTemplate.add_row([docId, document, labelRefs[labels[0]]])
    model.trainDocument(document, labels, docId)
  if verbosity > 0:
    print printTemplate

  return model


def setupExperiment(args):
  """
  Create model according to args, train on training data, save model,
  restore model.

  @return newModel (ClassificationModel) The restored NLP model.
  @return dataSet (list) Each item is a list representing a data sample, with
      the text string, list of label indices, and the sample ID.
  """
  dataSet, labelRefs, _, _ = readDataAndReshuffle(args)
  args.numLabels = len(labelRefs)

  # Create a model, train it, save it, reload it
  model = instantiateModel(args)
  model = trainModel(model, dataSet, labelRefs, args.verbosity)
  model.save(args.modelDir)
  newModel = ClassificationModel.load(args.modelDir)

  return newModel, dataSet


def testModel(model, testData, categorySize, verbosity=0):
  """
  Test the given model on testData, print out and return results metrics. The
  categorySize specifies the number of documents per category.

  For each data sample in testData the model infers the similarity to each other
  sample; distances are number of bits apart. We then calculate two types of
  results metrics: (i) true positives (TPs) and (ii) "ranks" statistics.

    i. From the sorted inference results, we get the top-ranked documents 1 to
    the categorySize. A true positive is defined as a top-ranked document that
    is in the same category as the test document. The max possible is thus the
    category size.

    ii. From the sorted inference results, we get the ranks of the
    samples that share the same category as the inference sample. Ideally these
    ranks would be low, and the test document itself would be at rank 0. The
    stats to describe these ranks are mean and skewness -- about 0 for normally
    distributed data, and a skewness value > 0 means that there is more weight
    in the left tail of the distribution. For example,
      [10, 11, 12, 13, 14, 15] --> mean=12.5, skew=0.0
      [0, 1, 2, 3, 4, 72] --> mean=13.7, skew=1.8

  @param categorySize (int) Number of documents per category; these unit tests
      use datasets with an exact number of docs in each category.

  @return allRanks (numpy array) Positions within the inference results list
      of the documents in the test document's category.
  """
  print
  print "========================Testing on sample text========================"
  print "A document passes the test if the ranks show its category's docs in "
  print "positions 0-{}.".format(categorySize-1)
  if verbosity > 0:
    print
    printTemplate = PrettyTable(["ID", "Document", "TP", "Ranks (Mean, Skew)"])
    printTemplate.align = "l"
    printTemplate.header_style = "upper"

  allRanks = []
  totalTPs = 0
  totalMeans = 0
  totalSkews = 0
  for (document, labels, docId) in testData:
    _, sortedIds, sortedDistances = model.inferDocument(
      document, returnDetailedResults=True, sortResults=True)

    # Compute TPs for this document
    expectedCategory = docId / 100
    truePositives = 0
    for i in xrange(categorySize):
      if sortedIds[i]/100 == expectedCategory:
        truePositives += 1
    totalTPs += truePositives

    # Compute the rank metrics for this document
    ranks = numpy.array(
      [i for i, index in enumerate(sortedIds) if index/100 == expectedCategory])
    allRanks.append(ranks)
    ranksMean = round(ranks.mean(), 2)
    ranksSkew = round(skew(ranks), 2)
    totalMeans += ranksMean
    totalSkews += ranksSkew

    if verbosity > 0:
      printTemplate.add_row(
        [docId, document, truePositives, (ranksMean, ranksSkew)])

  if verbosity > 0:
    print printTemplate
  lengthOfTest = float(len(testData))
  print
  print "Averages across all test documents:"
  print "TPs =", totalTPs/lengthOfTest
  print "Rank metrics (mean, skew) = ({}, {})".format(
    totalMeans/lengthOfTest, totalSkews/lengthOfTest)

  return allRanks


def printRankResults(testName, ranks):
  """ Print the ranking metric results."""
  ranksSum = sum(list(itertools.chain.from_iterable(ranks)))
  printTemplate = "{0:<32}|{1:<10}"
  print
  print
  print "Final rank sums for {} (lower is better):".format(testName)
  print printTemplate.format("Total", ranksSum)
  print printTemplate.format("Avg. per test sample",
                             float(ranksSum) / len(ranks))
