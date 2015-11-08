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

from collections import defaultdict, OrderedDict
import numpy
import pprint

from htmresearch.frameworks.nlp.runner import Runner
from htmresearch.support.data_split import Buckets


class BucketRunner(Runner):
  """Runner methods specific to the buckets experiment."""

  def __init__(self,
               dataPath,
               resultsDir,
               experimentName,
               experimentType,
               modelName,
               retinaScaling=1.0,
               retina="en_associative",
               apiKey=None,
               loadPath=None,
               plots=0,
               orderedSplit=False,
               folds=None,
               trainSizes=None,
               verbosity=0):
    """
    """
    numClasses = 1
    super(BucketRunner, self).__init__(dataPath, resultsDir, experimentName,
                                    experimentType, modelName,
                                    retinaScaling, retina, apiKey,
                                    loadPath, numClasses, plots, orderedSplit,
                                    folds, trainSizes, verbosity)


  def bucketData(self):
    """
    Populate self.buckets with a dictionary of buckets, where each category
    (key) is a bucket of its corresponding data samples.
    The patterns in a bucket list are in the order they are originally read in;
    this may or may not match the samples' unique IDs.
    """
    self.buckets = defaultdict(list)
    for p in self.patterns:
      self.buckets[p["labels"][0]].append(p)


  def partitionIndices(self, seed=42):
    """
    partitions is a list with an entry for each bucket; the list indices
    correspond to the runner.buckets keys, which are ints that identify each
    bucket (category) as mapped in runner.labelRefs.
    Each entry is a 2-tuple where the items are training and testing indices,
    respectively. The index values refer to the indices of the patterns listed
    in each bucket.
    In the below example, the bucket for category A is indexed at 0, and
    contains 3 encoded data samples (i.e. pattern dicts). The partitions specify
    the first pattern for training is at bucket index 1, which is the pattern w/
    unique ID 4, followed by the pattern at bucket index 0 (ID 13). The pattern
    at bucket index 2 (ID 42) will then be used for testing.

    labelRefs = ['category A', 'category B', ...]
    buckets = {
      0: [
           {<pattern w/ ID 4>},
           {<pattern w/ ID 13>},
           {<pattern w/ ID 42>}
        ],
      1: [
        ...],
      ...
    }
    partitions = [([1, 0], [2]), (...), ...]
    """
    if not self.buckets:
        raise RuntimeError("You need to first bucket the data.")
    bucketSizes = [len(x) for x in self.buckets.values()]
    # Create one partition (train, test) for each bucket.
    self.partitions = Buckets().split(
      bucketSizes, numTraining=4, randomize=(not self.orderedSplit), seed=seed)


  def runExperiment(self):
    for idx, bucket in self.buckets.iteritems():
      # train/test the model independently for each bucket
      self.resetModel(idx)

      # Skip data samples in the training set, rank those in the testing set.
      trainIndices = self.partitions[idx][0]
      trainIDs = [p["ID"] for p in
        [self.patterns[trainIdx] for trainIdx in trainIndices]]
      testIndices = self.partitions[idx][1]
      testIDs = [p["ID"] for p in
        [self.patterns[testIdx] for testIdx in testIndices]]

      if self.experimentType == "bucketsHighDim":
        self.runHD(bucket, trainIDs, testIDs)
      elif self.experimentType == "bucketsLowDim":
        self.runLD(trainIndices, testIDs)


  def highDimTest(self, queryPatterns):
    """
    Query the model for the input patterns, returning a dict w/ distance values
    for each query.
    """
    distances = {}
    for ID, pattern in queryPatterns.iteritems():
      distances[ID] = self.model.infer(pattern["pattern"])

    return distances


  @staticmethod
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


  def runHD(self, bucket, trainIDs, testIDs):
    """
    Train on all except for the bucket's training samples (trainIDs), and
    query the model with the training samples (evaluating the ranks of the
    testing samples).

    The model trains on everything up front, populating the KNN space w/ all
    of the samples (thus it's high dimensional).
    """
    queryPatterns = {}
    alreadyTrained = []
    for i, pattern in enumerate(self.patterns):
      if pattern["ID"] in trainIDs:
        # we use these patterns later when querying the model; duplicates get
        # overwritten b/c dict keys are the unique IDs
        queryPatterns[pattern["ID"]] = pattern
      elif pattern["ID"] in alreadyTrained:
        # samples appear multiple times, so don't repeat training
        continue
      else:
        self.model.trainModel(i)
        alreadyTrained.append(pattern["ID"])
    assert(sorted(queryPatterns.keys()) == sorted(trainIDs))

    # Infer distances, one training sample per iteration, combining the
    # inferred distances by taking the minimum across iterations.
    ## --> mean
    queryDistances = self.highDimTest(queryPatterns)
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
      metrics.append(self.getMetricsHD(bD, alreadyTrained, testIDs))

    print "================="
    print "Results for bucket ", self.labelRefs[bucket[0]["labels"]]
    pprint.pprint(metrics)


  def runLD(self):
    """
    Train on the bucket's training samples (one at a time), and query the
    model for all other samples (evaluating the ranks of the testing samples).

    The model only trains on the samples in the current training partition, so
    the resulting KNN space is low dimensional (i.e. 1-10).
    """
    for n in xrange(numTraining):
      # each iteration we train the model on one more sample from the training partition
      bucketIdxForTraining = trainIndices[n]
      self.model.trainModel(bucketIdxForTraining)
      import pdb; pdb.set_trace()
      distances = lowDimTest(trainIDs)
      getMetricsLD(distances, testIDs)


  def evaluateResults(self):
    """
    Calculate evaluation metrics from the bucketing results.
    """


    return
