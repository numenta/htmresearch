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
import itertools
import numpy
import pprint

from collections import defaultdict, OrderedDict

from htmresearch.frameworks.nlp.runner import Runner
from htmresearch.support.data_split import Buckets


class BucketRunner(Runner):
  """Runner methods specific to the buckets experiment."""

  def __init__(self, combineMethod="min", *args, **kwargs):
    """
    @param combineMethod  (str) How to combine KNN distances from subsequent
                                inference iterations.
    """
    self.combineMethod = combineMethod
    self.buckets = None

    super(BucketRunner, self).__init__(*args, **kwargs)


  def bucketData(self):
    """
    Populate self.buckets with a dictionary of buckets, where each category
    (key) is a bucket of its corresponding data samples.

    The patterns in a bucket list are in the order they are originally read in;
    this may or may not match the samples' unique IDs.

    Buckets with insufficient size (fewer samples than numInference) are
    skipped over later.
    """
    self.buckets = defaultdict(list)
    for d in self.dataDict.values():
      bucketName = d[1][0]
      self.buckets[bucketName].append(d)


  def partitionIndices(self, seed=42, numInference=10):
    """
    partitions is a list with an entry for each bucket; the list indices
    correspond to the runner.buckets keys, which are ints that identify each
    bucket (category) as mapped in runner.labelRefs.
    Each entry is a 2-tuple where the items are testing and "ranking" indices,
    respectively. The index values refer to the indices of the patterns listed
    in each bucket.
    In the below example, the bucket for category A is indexed at 0, and
    contains three data samples. The partitions specify the first pattern for
    testing is at bucket index 1, which is the sample w/ unique ID 4, followed
    by the pattern at bucket index 0 (ID 13). The pattern at bucket index 2
    (ID 42) will then be used in the ranking step, where we evaluate test
    (i.e. inference) results.

    labelRefs = ['category A', 'category B', ...]
    buckets = {
      0: [
           {<data w/ ID 4>},
           {<data w/ ID 13>},
           {<data w/ ID 42>}
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

    # Create one partition (test, rank) for each bucket.
    self.partitions = Buckets().split(
      bucketSizes, numInference, randomize=(not self.orderedSplit), seed=seed)


  def train(self):
    """No training for non-HTM models, just populating the KNN later."""
    pass


  def runExperiment(self, numInference):
    """
    Populate the KNN space w/ all of the samples and experiment with each
    bucket, returning a dict with metrics for each bucket.
    """
    # The prototypeMap dict maps data samples' unique IDs to KNN prototype #s.
    prototypeMap = self.populateKNN()
    metrics = {}
    for idx, bucket in self.buckets.iteritems():
      if len(bucket) < numInference:
        print "Skipping bucket '{}' because it has too few samples.".format(
          self.labelRefs[idx])
        continue
      testIDs, rankIDs = self.prepBucket(idx)

      metrics[idx] = self.testBucket(idx, prototypeMap, testIDs, rankIDs)

    return metrics


  def populateKNN(self):
    """
    Populate the KNN space with every pattern. We track the KNN prototype number
    for each pattern ID so we know where they are in the KNN space.
    """
    prototypeMap = OrderedDict()
    prototypeNum = 0
    for i, pattern in enumerate(self.patterns):
      if prototypeMap.get(pattern["ID"]) is not None:
        # samples appear multiple times, so don't repeat training
        continue
      else:
        # train on pattern i, and keep track of it's index in KNN space
        self.model.trainModel(i)
        prototypeMap[pattern["ID"]] = prototypeNum
        prototypeNum += 1

    return prototypeMap


  def prepBucket(self, idx):
    """Test samples are for inferring, rank samples are for evaluation."""
    testIndices = self.partitions[idx][0]
    rankIndices = self.partitions[idx][1]

    testIDs = [d[2] for d in
      [self.dataDict[testIdx] for testIdx in testIndices]]
    rankIDs = [d[2] for d in
      [self.dataDict[rankIdx] for rankIdx in rankIndices]]

    return testIDs, rankIDs


  def testBucket(self, bucketNum, trainIDs, testIDs, rankIDs):
    """
    Use the testing samples to infer distances. The distances for subsequent
    iterations are combined according to self.combineMethod.

    @param bucketNum    (int)     Index of bucket to test.
    @param testIDs      (list)    Unique IDs of samples for inference.
    @param trainIDs     (dict)    Maps sample unique IDs to their prototype
                                  index in KNN space.
    @param rankIDs      (list)    IDs of the samples we want metrics on.
    """
    # Query the model for the input patterns, returning a dict w/ distance
    # values for each query.
    distances = OrderedDict()
    for i, pattern in enumerate(self.patterns):
      if pattern["ID"] in testIDs and pattern["ID"] not in distances.keys():
        distances[pattern["ID"]] = self.model.infer(pattern["pattern"])
    assert(sorted(distances.keys()) == sorted(testIDs))

    accumulatedDistances = self.setupDistances(distances, trainIDs)

    metrics = self.getMetrics(accumulatedDistances, trainIDs, rankIDs)

    if self.verbosity > 0:
      print "====="
      print "Total data samples in KNN space = ", len(trainIDs)
      print "Results for bucket ", self.labelRefs[self.buckets[bucketNum][0][1]]
      pprint.pprint(metrics)

    return metrics


  @staticmethod
  def setupDistances(distances, trainIDs, method="min"):
    """
    Combine the distance results of each iteration with the method specified.

    @param distances  (dict)  Keys are unique IDs of the inferred samples,
                              values are the distance arrays from KNN
                              inference results.
    @param trainIDs   (dict)  Map of unique IDs to sequence numbers as they are
                              used when populating KNN space.
    @param method     (str)   Method to combine distances arrays.
    """
    if method not in ("min", "mean"):
      raise ValueError(
        "Distance combination method must be one of 'min' or 'mean'.")

    if method == "mean":
      currentBest = numpy.zeros(len(alreadyTrained))
    elif method == "min":
      currentBest = numpy.ones(len(trainIDs))
    accumulatedDistances = []
    for ID, dist in distances.iteritems():

      if method == "mean":
        currentBest += dist
        currentBest = currentBest / (n+1.0)
      elif method == "min":
        currentBest = numpy.minimum(currentBest, dist)

      # In each iteration, exclude the queried samples.
      currentBest[trainIDs[ID]] = 1.0  # TODO: better way to get rid of these?

      accumulatedDistances.append(currentBest)

    return accumulatedDistances


  def getMetrics(self, accumulatedDistances, trainIDs, rankIDs):
    """
    @param accumulatedDistances (list)    Results of setupDistance, where each
                                    subsequent item is a numpy.array of
                                    combined distances to KNN prototypes.
    @param trainIDs       (dict)    IDs corresponding to KNN prototype indices.
    @param rankIDs        (list)    IDs of the samples we want metrics on.
    @return metrics       (list)    Dict of metrics for each iteration.
    """
    metrics = []
    # for mD in meanDistances:  ## mean
    for distances in accumulatedDistances:
      rankIndices = numpy.argsort(distances)
      rankPrototypes = [trainIDs[ID] for ID in rankIDs]
      # ranks = rankIndices[list(itertools.chain.from_iterable(rankPrototypes))]
      ranks = rankIndices[rankPrototypes]

      metrics.append({
        "mean": ranks.mean(),
        "lastTP": ranks.max(),
        "firstTP": ranks.min(),  # ideally this would be 0
        "numTop10": len([r for r in ranks if r < 10]),
        "totalRanked": len(rankIDs),
        }
      )

    return metrics


  def evaluateResults(self, metrics):
    """
    Calculate evaluation metrics from the bucketing results.

    @param metrics    (dict)    Keys are bucket numbers.
    """
    import pdb; pdb.set_trace()
    # TODO: plots, aggregate metrics across buckets


    if self.plots:  ## if nothing but plotting, move this to the run script
      print "plots..."


    return
