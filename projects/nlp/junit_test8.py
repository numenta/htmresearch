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
Simple script to run unit test 8.

Dataset: we start with base sentences each with ten unique words, where no two
words repeat within a sentence. For each base sentence we generate nine new
sentences that are each a permuted version of the original sentence.  The
generated sentences will be ordered according to how closely the order of the
words match the order in the original sentence. The most similar sentence will
be identical. The least similar sentence will have no two consecutive words in
the same order as the original sentence.

Methodology: for the unit test we train the system on each of the sentences. We
then query a temporal memory based network using just the base sentence as the
search term. A perfect result ranks the ten sentences generated from the search
term exactly according to how closely the order matches the base sentence.  A
perfect result would also not place any of the other sentences in the top 10.

"""

import argparse
from prettytable import PrettyTable

from htmresearch.support.junit_testing import setupExperiment


# Dataset info
CATEGORY_SIZE = 6
NUMBER_OF_DOCS = 108


def testModel(model, testData, categorySize=10, verbosity=0):
  """
  Test the given model on testData, print out and return results metrics.

  For each test document in testData the model infers a score based on the rank
  of the closest documents. A perfect result on this test would return a sorted
  list of documents where the top ones belong to the same category as the test,
  and where each document's index is equal to its rank. The score is the number
  of correctly returned documents where the rank equals the document index.

  @param categorySize (int) Number of documents per category; these unit tests
      use datasets with an exact number of docs in each category.
  """
  modelName = repr(model).split()[0].split(".")[-1]
  print
  print "===================Testing {} on sample text==================".format(
    modelName)
  print
  printTemplate = PrettyTable(["ID", "Document", "Score", "TPs"])
  printTemplate.align = "l"
  printTemplate.header_style = "upper"

  summedScore = 0
  for (document, labels, docId) in testData:
    _, sortedIds, sortedDistances = model.inferDocument(
      document, returnDetailedResults=True, sortResults=True)

    # Compute the number of documents whose rank exactly matches their index
    expectedCategory = docId / 100
    score = 0
    truePositives = 0
    for i in xrange(categorySize):
      if (i < len(sortedIds)) and sortedIds[i]/100 == expectedCategory:
        truePositives += 1
        if sortedIds[i]%100 == i:
          score += 1
    summedScore += score

    if verbosity >= 2:
      print "\nInference result:"
      print "docId=",docId,"document=",document
      print "sortedIds=",sortedIds[0:categorySize]
      print "score = ",score,"TP=",truePositives

    docStr = unicode(document, errors='ignore')[0:100]
    printTemplate.add_row( [docId, docStr, score, truePositives])

  lengthOfTest = float(len(testData))

  print printTemplate
  print
  print "Scores across all test documents:",
  print "  Total:", summedScore
  print "  Avg:", summedScore/float(lengthOfTest)



def runExperiment(args):
  """ Build a model and test it."""
  model, dataSet = setupExperiment(args)

  # Test model using just the base sentences as query sentences
  testModel(model, [d for d in dataSet if d[2]%100 == 0],
            verbosity=args.verbosity)
  return model



if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description=helpStr
  )

  parser.add_argument("-c", "--networkConfigPath",
          default="data/network_configs/sensor_16k_simple_tp_history2_knn.json",
          help="Path to JSON specifying the network params.",
          type=str)
  parser.add_argument("-m", "--modelName",
                      default="htm",
                      type=str,
                      help="Name of model class. Options: [keywords,htm]")
  parser.add_argument("--modelDir",
                      default="MODELNAME.checkpoint",
                      help="Model will be saved in this directory.")
  parser.add_argument("--retina",
                      default="en_synonymous",
                      type=str,
                      help="Name of Cortical.io retina.")
  parser.add_argument("--apiKey",
                      default=None,
                      type=str,
                      help="Key for Cortical.io API. If not specified will "
                      "use the environment variable CORTICAL_API_KEY.")
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

  # Default dataset for this unit test
  args.dataPath = "data/junit/unit_test_8.csv"

  model = runExperiment(args)
