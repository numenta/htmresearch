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
"""
Script to run "bucket" classification experiment.
"""

import argparse
import os
import pprint
import time

from htmresearch.frameworks.nlp.runner import Runner
from htmresearch.frameworks.nlp.htm_runner import HTMRunner



def checkInputs(args):
  """Function that displays a set of arguments and asks to proceed."""
  pprint.pprint(vars(args))
  userIn = raw_input("Proceed? (y/n): ")

  if userIn == 'y':
    return True

  if userIn == 'n':
    return False

  print "Incorrect input given\n"
  return checkInputs(args)


def run(args):
  start = time.time()

  root = os.path.dirname(os.path.realpath(__file__))
  resultsDir = os.path.join(root, args.resultsDir)

  if args.modelName == "HTMNetwork":
    runner = HTMRunner(dataPath=args.dataPath,
                       networkConfigPath=args.networkConfigPath,
                       resultsDir=resultsDir,
                       experimentName=args.experimentName,
                       experimentType=args.experimentType,
                       loadPath=args.loadPath,
                       modelName=args.modelName,
                       retinaScaling=args.retinaScaling,
                       retina=args.retina,
                       apiKey=args.apiKey,
                       numClasses=1,
                       plots=args.plots,
                       orderedSplit=args.orderedSplit,
                       verbosity=args.verbosity,
                       generateData=args.generateData,
                       classificationFile=args.classificationFile,
                       seed=args.seed)
    runner.initModel(0)
  else:
    runner = Runner(dataPath=args.dataPath,
                    resultsDir=resultsDir,
                    experimentName=args.experimentName,
                    experimentType=args.experimentType,
                    loadPath=args.loadPath,
                    modelName=args.modelName,
                    retinaScaling=args.retinaScaling,
                    retina=args.retina,
                    apiKey=args.apiKey,
                    numClasses=1,
                    plots=args.plots,
                    orderedSplit=args.orderedSplit,
                    verbosity=args.verbosity)
    runner.initModel(args.modelName)

  print "Reading in data and preprocessing."
  dataTime = time.time()
  runner.setupData(args.textPreprocess)
  print ("Data setup complete; elapsed time is {0:.2f} seconds.\nNow encoding "
         "the data".format(time.time() - dataTime))

  encodeTime = time.time()
  runner.encodeSamples()
  print ("Encoding complete; elapsed time is {0:.2f} seconds.\nNow running the "
         "experiment.".format(time.time() - encodeTime))

  runner.bucketData()
  runner.partitionIndices(args.seed)

  # runner.partitons is a list with an entry for each bucket; the list indices
  # correspond to the runner.buckets keys, which are ints that identify each
  # bucket (category) as mapped in runner.labelRefs.
  # Each entry is a 2-tuple where the items are training and testing indices,
  # respectively. The index values refer to the indices of the patterns listed in
  # each bucket.
  # In the below example, the bucket for category A is indexed at 0, and contains
  # 3 encoded data samples (i.e. pattern dicts). The partitions specify the first
  # pattern for training is at bucket index 1, which is the pattern w/ unique ID
  # 4, followed by the pattern at bucket index 0 (ID 13). The pattern at bucket
  # index 2 (ID 42) will then be used for testing.

  # labelRefs = ['category A', 'category B', ...]
  # buckets = {
  #   0: [
  #        {<pattern w/ ID 4>},
  #        {<pattern w/ ID 13>},
  #        {<pattern w/ ID 42>}
  #     ],
  #   1: [
  #     ...],
  #   ...
  # }
  # partitions = [([1, 0], [2]), (...), ...]


  # TODO: move the below code into Runner
  from collections import OrderedDict
  import numpy


  def highDimTest(queryPatterns):
    """
    Query the model for the input patterns, returning a dict w/ distance values
    for each query.
    """
    distances = {}
    for ID, pattern in queryPatterns.iteritems():
      # dist = runner.model.infer(pattern["pattern"])
      distances[ID] = runner.model.infer(pattern["pattern"])

    return distances


  def getMetricsHD(distances, alreadyTrained, testIDs):
    """
    @param distances        (dict)  Keys are IDs of queried samples, values are numpy.arrays of distances to KNN prototypes.
    @param distances      (numpy array)

    @param alreadyTrained   (list)  IDs corresponding to KNN prototypes (in distances).
    @return
    """
    rankIndices = numpy.argsort(distances)
    rankedIDs = [alreadyTrained[i] for i in rankIndices]
    testRanks = numpy.array([rankedIDs.index(ID) for ID in testIDs])  # TODO: faster way?

    return {
      "mean": testRanks.mean(),
      "lastTP": testRanks.max(),
      "firstTP": testRanks.min(),  # ideally this would be 0
      "numTop10": len([r for r in testRanks if r < 10]),
      "total": len(distances),
    }


  def lowDimTest(skipIDs):
    """
    Get distances for all patterns but those specified in skipIDs.

    return distances    (dict)    An entry for each data pattern not held out
        for training. Keys are the data samples' unique IDs, values are
        3-tuples of min, mean, max distance to trained on patterns.
    """
    # We use the unique IDs b/c the dataset contains duplicates.
    distances = OrderedDict()
    for i, p in enumerate(runner.patterns):
      if p["ID"] in skipIDs: continue
      dist = runner.model.infer(p["pattern"])
      distances[p["ID"]] = (dist.min(), dist.mean(), dist.max())

    return distances


  def getMetricsLD(distances, rankIDs):
    """
    Sort the distances from closest to farthest, returning the following metrics
    (that are functions of the ranks of the patterns specified by rankIDs):
      - mean rank: average rank of all rankIDs
      - rank of first mistake: where is the first non-rankID pattern?
      - rank of last TP: worst rank of the rankIDs
      - number of TPs in top 10: of the closest 10 patterns, how many are of
        rankIDs?
    """
    # uniqueIDs = distances.keys()

    # # Sort distances by minimums
    # orderedByMin = OrderedDict(sorted(distances.items(), key=lambda x: x[1]))
    # orderedByMean = OrderedDict(sorted(distances.items(), key=lambda x: x[2]))
    # orderedByMax = OrderedDict(sorted(distances.items(), key=lambda x: x[3]))
    # import pdb; pdb.set_trace()
    for i in xrange(3):
      # Calculate metrics for min(0), mean(10, and max(2) distances
      ordered = OrderedDict(sorted(distances.items(), key=lambda x: x[i+1]))
      import pdb; pdb.set_trace()
      # where are the rankIDs?
      # [rank for rank, ID in enumerate(ordered) if rank in rankIDs]
      ranks = numpy.array([ordered.keys().index(ID) for ID in rankIDs])
      mean = ranks.mean()
      lastTP = ranks.max()
      firstFP = ()
      numTop10 = len([r for r in ranks if r < 10])


    # An entry for each metric, where each entry is a 3-tuple representing the
    # min, mean, max distances.
    metricsDict = {
      "mean rank": (),
      "rank of first mistake": (),
      "rank of last TP": (),
      "number of TPs in top 10": (),
    }

    return metricsDict



  numTraining = 10  # TODO: make this an arg
  for idx, bucket in runner.buckets.iteritems():
    # train/test the model independently for each bucket
    runner.resetModel(idx)
    # Skip the data samples in the training set, rank those in the testing set.
    trainIndices = runner.partitions[idx][0]
    trainIDs = [p["ID"] for p in [runner.patterns[trainIdx] for trainIdx in trainIndices]]
    testIndices = runner.partitions[idx][1]
    testIDs = [p["ID"] for p in [runner.patterns[testIdx] for testIdx in testIndices]]

    if args.experimentType == "bucketsHighDim":
      # Train on all except for the bucket's training samples (trainIDs), and
      # query the model with the training samples (evaluating the ranks of the
      # testing samples).
      queryPatterns = {}
      alreadyTrained = []
      for i, pattern in enumerate(runner.patterns):
        if pattern["ID"] in trainIDs:
          # we use these patterns later when querying the model; duplicates get
          # overwritten b/c dict keys are the unique IDs
          queryPatterns[pattern["ID"]] = pattern
        elif pattern["ID"] in alreadyTrained:
          # samples appear multiple times, so don't repeat training
          continue
        else:
          runner.model.trainModel(i)
          alreadyTrained.append(pattern["ID"])
      assert(sorted(queryPatterns.keys()) == sorted(trainIDs))

      # Infer distances, one training sample per iteration, combining the
      # inferred distances by taking the minimum across iterations.
      ## --> mean
      queryDistances = highDimTest(queryPatterns)
      # summedDistances = numpy.zeros(len(alreadyTrained))  ##
      # meanDistances = []  ##
      currentBest = numpy.ones(len(alreadyTrained))
      bestDistances = []
      for n, (ID, dist) in enumerate(queryDistances.iteritems()):
        # summedDistances += dist  ##
        # meanDistances.append(summedDistances / (n+1.0))  ##
        currentBest = numpy.minimum(currentBest, dist)
        bestDistances.append(currentBest)

      metrics = []
      # for mD in meanDistances:  ##
      for bD in bestDistances:
        # metrics.append(getMetricsHD(mD, alreadyTrained, testIDs))  ##
        metrics.append(getMetricsHD(bD, alreadyTrained, testIDs))

      print "================="
      print "Results for bucket ", runner.labelRefs[bucket[0]["labels"]]
      pprint.pprint(metrics)

    elif args.experimentType == "bucketsLowDim":
      # Train on the bucket's training samples (one at a time), and query the
      # model for all other samples (evaluating the ranks of the testing samples).
      for n in xrange(numTraining):
        # each iteration we train the model on one more sample from the training partition
        bucketIdxForTraining = trainIndices[n]
        runner.model.trainModel(bucketIdxForTraining)
        import pdb; pdb.set_trace()
        distances = lowDimTest(trainIDs)
        getMetricsLD(distances, testIDs)

  import pdb; pdb.set_trace()  # TODO: plots, aggregate metrics across buckets



  # For each bucket we define the following:
  #   trainIDs: selection of numTraining (e.g. 10) IDs out of the bucket
  #   allIDs: the full set w/o trainIDs
  #   testIDs: the bucket w/o trainIDs

  if experimentType == "lowDim":
    # The model only trains on the samples in the current training partition, so
    # the resulting KNN space is low dimensional (i.e. 1-10).
    return

  if experimentType == "highDim":
    # The model trains on everything up front, populating the KNN space w/ all
    # of the samples (thus it's high dimensional).
    return




  runner.writeOutClassifications()

  resultCalcs = runner.calculateResults()
  runner.evaluateCumulativeResults(resultCalcs)

  print "Saving..."
  runner.saveModel()

  print "Experiment complete in {0:.2f} seconds.".format(time.time() - start)

  if args.validation:
    print "Validating experiment against expected classifications..."
    print runner.validateExperiment(args.validation)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("dataPath",
                      help="Path to data CSV or folder of CSVs.",
                      type=str)
  parser.add_argument("--networkConfigPath",
                      default="htm_network_config.json",
                      help="Path to JSON specifying the network params.",
                      type=str)
  parser.add_argument("--test",
                      default=None,
                      help="Path to data CSV to use for testing if provided. "
                           "Otherwise will test on \'dataPath\'.")
  parser.add_argument("-n", "--experimentName",
                      default="kfolds",
                      type=str,
                      help="Experiment name.")
  parser.add_argument("-e", "--experimentType",
                      default="k-folds",
                      type=str,
                      help="Either 'k-folds', 'incremental', or 'buckets'.")
  parser.add_argument("-m", "--modelName",
                      default="Keywords",
                      type=str,
                      help="Name of model class. Also used for model results "
                           "directory and pickle checkpoint.")
  parser.add_argument("--retinaScaling",
                      default=1.0,
                      type=float,
                      help="Factor by which to scale the Cortical.io retina.")
  parser.add_argument("--retina",
                      default="en_associative",
                      type=str,
                      help="Name of Cortical.io retina.")
  parser.add_argument("--apiKey",
                      default=None,
                      type=str,
                      help="Key for Cortical.io API.")
  parser.add_argument("--resultsDir",
                      default="results",
                      help="This will hold the experiment results.")
  parser.add_argument("--textPreprocess",
                      action="store_true",
                      default=False,
                      help="Whether or not to use text preprocessing.")
  parser.add_argument("--loadPath",
                      help="Path from which to load the serialized model.",
                      type=str,
                      default=None)
  parser.add_argument("--plots",
                      default=1,
                      type=int,
                      help="0 for no evaluation plots, 1 for classification "
                           "accuracy plots, 2 includes the confusion matrix.")
  parser.add_argument("--orderedSplit",
                      default=False,
                      action="store_true",
                      help="To split the train and test sets, False will split "
                           "the samples randomly, True will allocate the "
                           "first n samples to training with the remainder "
                           "for testing.")
  parser.add_argument("--seed",
                      default=42,
                      type=int,
                      help="Random seed, used in partitioning the data.")
  parser.add_argument("-v", "--verbosity",
                      default=1,
                      type=int,
                      help="verbosity 0 will print out experiment steps, "
                           "verbosity 1 will include results, and verbosity > "
                           "1 will print out preprocessed tokens and kNN "
                           "inference metrics.")
  parser.add_argument("--validation",
                      default="",
                      help="Path to file of expected classifications.")
  parser.add_argument("--skipConfirmation",
                      help="If specified will skip the user confirmation step.",
                      default=False,
                      action="store_true")
  parser.add_argument("--generateData",
                      default=False,
                      action="store_true",
                      help="Whether or not to generate network data files.")
  parser.add_argument("--votingMethod",
                      default="last",
                      choices=["last", "most"],
                      help="Method to use when picking final classifications.")
  parser.add_argument("--classificationFile",
                      default="",
                      help="JSON file mapping string labels to ids.")

  args = parser.parse_args()

  if args.skipConfirmation or checkInputs(args):
    run(args)
