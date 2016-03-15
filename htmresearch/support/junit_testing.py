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
import os
import plotly.plotly as py

from plotly.graph_objs import Box, Figure, Histogram, Layout
from prettytable import PrettyTable
from scipy.stats import skew

from htmresearch.frameworks.nlp.classification_model import ClassificationModel
from htmresearch.frameworks.nlp.model_factory import (
  createModel, getNetworkConfig)
from htmresearch.support.csv_helper import readDataAndReshuffle


# There should be one "htm" model for each htm config entry.
nlpModelTypes = [
  "CioDocumentFingerprint",
  "CioWordFingerprint",
  "htm",
  "htm",
  "htm",
  "Keywords"]
htmConfigs = {
  ("HTM_sensor_knn", "data/network_configs/sensor_knn.json"),
  ("HTM_sensor_simple_tp_knn", "data/network_configs/sensor_simple_tp_knn.json"),
  ("HTM_sensor_tm_knn", "data/network_configs/sensor_tm_knn.json"),
}

# Some values of k we know work well.
kValues = { "Keywords": 21 }



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


def instantiateModel(args):
  """
  Set some specific arguments and return an instance of the model we will use.
  """
  args.networkConfig = getNetworkConfig(args.networkConfigPath)
  args.k = kValues.get(args.modelName, 1)
  return createModel(**vars(args))


def trainModel(model, trainingData, labelRefs, verbosity=0):
  """
  Train the given model on trainingData. Return the trained model instance.
  """
  modelName = repr(model).split()[0].split(".")[-1]
  print
  print "===================Training {} on sample text================".format(
    modelName)
  if verbosity > 0:
    printTemplate = PrettyTable(["ID", "Document", "Label"])
    printTemplate.align = "l"
    printTemplate.header_style = "upper"
  for (document, labels, docId) in trainingData:
    if verbosity > 0:
      docStr = unicode(document, errors='ignore')[0:100]
      printTemplate.add_row([docId, docStr, labelRefs[labels[0]]])
    model.trainDocument(document, labels, docId)
  if verbosity > 0:
    print printTemplate

  return model


def testModel(model, testData, categorySize, verbosity=0):
  """
  Test the given model on testData, print out and return results metrics.

  For each data sample in testData the model infers the similarity to each other
  sample; distances are number of bits apart. We then find the "ranks" of true
  positive (TP) documents -- those that are in the same category as the test
  document. Ideally these ranks will be low, and a perfect result would be ranks
  0-categorySize.

  The stats we use to describe these ranks are mean and skewness -- about 0 for
  normally distributed data, and a skewness value > 0 means that there is more
  weight in the left tail of the distribution. For example,
    [10, 11, 12, 13, 14, 15] --> mean=12.5, skew=0.0
    [0, 1, 2, 3, 4, 72] --> mean=13.7, skew=1.8

  @param categorySize (int) Number of documents per category; these unit tests
      use datasets with an exact number of docs in each category.

  @return (numpy array) Rank positions of TPs for all test instances.
  @return avgRanks (numpy array) Average rank positions of TPs -- length is the
      categorySize.
  @return avgStats (numpy array) Average stats of the TP ranks -- length is the
      categorySize.
  """
  modelName = repr(model).split()[0].split(".")[-1]
  print
  print "===================Testing {} on sample text==================".format(
    modelName)
  if verbosity > 0:
    print
    printTemplate = PrettyTable(["ID", "Document", "TP", "Ranks (Mean, Skew)"])
    printTemplate.align = "l"
    printTemplate.header_style = "upper"

  allRanks = []
  summedRanks = numpy.zeros(categorySize)
  totalTPs = 0
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

    if (verbosity >= 2) and (truePositives < categorySize):
      print "\nIncorrect inference result:"
      print "docId=",docId,"document=",document
      print "sortedIds=",sortedIds
      print "truePositives = ",truePositives

    # Compute the rank metrics for this document
    ranks = numpy.array(
      [i for i, index in enumerate(sortedIds) if index/100 == expectedCategory])
    allRanks.extend(ranks)
    summedRanks += ranks
    ranksMean = round(ranks.mean(), 2)
    ranksSkew = round(skew(ranks), 2)

    if verbosity > 0:
      docStr = unicode(document, errors='ignore')[0:100]
      printTemplate.add_row(
        [docId, docStr, truePositives, (ranksMean, ranksSkew)])

  lengthOfTest = float(len(testData))
  avgRanks = summedRanks/lengthOfTest
  avgStats = (round(avgRanks.mean(), 2), round(skew(avgRanks), 2))

  if verbosity > 0:
    print printTemplate
  print
  print "Averages across all test documents:"
  print "TPs =", totalTPs/lengthOfTest
  print "Rank metrics (mean, skew) = ({}, {})".format(avgStats[0], avgStats[1])

  return numpy.array(allRanks), avgRanks, avgStats


def printRankResults(testName, avgRanks, avgStats):
  """ Print the ranking metric results."""
  printTemplate = "{0:<32}|{1:<10}"
  print
  print
  print "Averaged rank metrics for {}:".format(testName)
  print printTemplate.format("Avg. ranks per doc", avgRanks)
  print printTemplate.format("Avg. mean and skew", avgStats)


def plotResults(ranksArrays, ranks, maxRank, testName="JUnit Test"):
  """ Plot a histogram of the ranks.

  @param ranksArrays (dict) Keys: model names. Values: List of TP ranks.
  @param ranks (dict) Keys: model names. Values: Averaged TP ranks.
  @param maxRank (int) Highest rank of TP possible.

  @return (str) Plot URLs.
  """
  py.sign_in(os.environ["PLOTLY_USERNAME"], os.environ["PLOTLY_API_KEY"])
  colors = ["rgba(93, 164, 214, 0.5)", "rgba(255, 144, 14, 0.5)",
            "rgba(44, 160, 101, 0.5)", "rgba(255, 65, 54, 0.5)",
            "rgba(207, 114, 255, 0.5)"]

  histogramTraces = []
  for i, (modelName, allRanks) in enumerate(ranksArrays.iteritems()):
    # Display distribution stats in legend
    mean = round(allRanks.mean(), 2)
    sk = round(skew(allRanks), 2)

    # Setup histogram for this model
    histogramTraces.append(Histogram(
      y=allRanks,
      name="{}: ({}, {})".format(modelName, mean, sk),
      autobiny=False,
      ybins=dict(
        start=0.0,
        end=maxRank,
        size=1.0,
      ),
      marker=dict(
        color=colors[i],
      ),
      opacity=0.7,
    ))

  histogramLayout = Layout(
    title="{} - Where are the True Positives?".format(testName),
    xaxis=dict(
      title="Count",
    ),
    yaxis=dict(
      title="Rank of TPs",
      range=[maxRank, 0],
    ),
    barmode="overlay",
    showlegend=True,
  )
  histogramFig = Figure(data=histogramTraces, layout=histogramLayout)
  histogramURL = py.plot(histogramFig)

  return histogramURL
