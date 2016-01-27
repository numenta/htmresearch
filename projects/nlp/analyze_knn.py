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
Script to analyze the stored patterns in a knn classifier
"""

import argparse
import csv
import numpy
from textwrap import TextWrapper

from htmresearch.support.register_regions import registerAllResearchRegions
from htmresearch.support.csv_helper import readDataAndReshuffle
from htmresearch.frameworks.nlp.classification_model import ClassificationModel


wrapper = TextWrapper(width=100)

def analyzeModel(args, model, documentTextMap):
  """
  Test the given model on testData and print out accuracy.

  Accuracy is calculated as follows. Each token in a document votes for a single
  category. The document is classified with the category that received the most
  votes. Note that it is possible for a token and/or document to receive no
  votes, in which case it is counted as a misclassification.
  """
  knn = model.getClassifier()

  # It would be better if there were proper accessors for this information
  numPatterns = knn._numPatterns
  inputWidth = float(knn._Memory.nCols())
  sparsities = numpy.zeros(numPatterns)
  nonZeroBits = numpy.zeros(numPatterns)

  print "Number of patterns stored in KNN=",numPatterns
  print "Dimensionality of input patterns=",inputWidth
  for i in range(numPatterns):
    nonZeroBits[i] = len(knn.getPattern(i, sparseBinaryForm=True))
    sparsities[i] = (100.0*nonZeroBits[i]/inputWidth)

    # For debugging. You can print out specific documents like this:
    # words = len(documentTextMap[knn.getPartitionId(i)].split())
    # if sparsities[i] < 15 and words > 160:
    #   print wrapper.fill(documentTextMap[knn.getPartitionId(i)])
    #   print

  # Compute min/max/mean sparsities
  print "Min sparsity:     %5.3f"%(sparsities.min())
  print "Max sparsity:     %5.3f"%(sparsities.max())
  print "Average sparsity: %5.3f"%(sparsities.mean())
  print "Sparsity stdev:   %5.3f"%(sparsities.std())

  with open(args.outFile, "wb") as outFile:
    csvWriter = csv.writer(outFile)
    csvWriter.writerow(["words", "sparsity", "bytes"])
    sortedIndices = sparsities.argsort()
    for i in sortedIndices:
      bytes = len(documentTextMap[knn.getPartitionId(i)])
      words = len(documentTextMap[knn.getPartitionId(i)].split())
      csvWriter.writerow([words, sparsities[i], bytes])


def runExperiment(args):
  """
  Create model according to args, train on training data, save model,
  restore model, test on test data.
  """

  (trainingData, labelRefs, documentCategoryMap,
   documentTextMap) = readDataAndReshuffle(args,
                         [8,9,10,5,6,11,13,0,1,2,3,4,7,12,14])

  model = ClassificationModel.load(args.modelDir)

  analyzeModel(args, model, documentTextMap)

  return model


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description=helpStr
  )

  parser.add_argument("--numLabels",
                      default=13,
                      type=int,
                      help="Number of unique labels to train on.")
  parser.add_argument("-m", "--modelDir",
                      default="docfp.checkpoint",
                      help="Checkpoint directory where model is saved.")
  parser.add_argument("--dataPath",
                      default=None,
                      help="CSV file containing original labeled dataset.")
  parser.add_argument("--outFile",
                      default="temp.csv",
                      help="Some statistics will be output to this CSV file.")
  parser.add_argument("-v", "--verbosity",
                      default=1,
                      type=int,
                      help="verbosity level of printouts")
  args = parser.parse_args()

  model = runExperiment(args)
