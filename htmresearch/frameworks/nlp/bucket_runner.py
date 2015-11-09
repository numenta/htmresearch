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
import itertools
import numpy
import pprint

from htmresearch.frameworks.nlp.runner import Runner
from htmresearch.support.data_split import Buckets


class BucketRunner(Runner):
  """Runner methods specific to the buckets experiment."""

  def __init__(self, numInference=10, *args, **kwargs):
    """
    @param numInference   (int)     Number of samples (per bucket) for inference
    """
    super(BucketRunner, self).__init__(numClasses=1, *args, **kwargs)


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

    # Create one partition (train, test) for each bucket.
    self.partitions = Buckets().split(
      bucketSizes, numInference, randomize=(not self.orderedSplit), seed=seed)


  def train(self):
    """No training for non-HTM models, just populating the KNN later."""
    pass


  def runExperiment(self, numInference):
    """
    The model trains on everything up front, populating the KNN space w/ all
    of the samples. Then we experiment with each bucket, returning a dict with
    metrics for each bucket.
    """
    # The trainIDs dict maps data samples' unique IDs to prototype #.
    trainIDs = self.populateKNN()
    metrics = {}
    for idx, bucket in self.buckets.iteritems():
      if len(bucket) < numInference:
        print "Skipping bucket '{}' because it has too few samples.".format(
          self.labelRefs[idx])
        continue
      testIDs, rankIDs = self.prepBucket(idx)
      import pdb; pdb.set_trace()
      metrics[idx] = self.testBucket(bucket, trainIDs, testIDs, rankIDs, idx)

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
        prototypeMap[pattern["ID"]] = [prototypeNum]  # in a list b/c some models (htm) classify every word
        prototypeNum += 1

    return prototypeMap


  def prepBucket(self, idx):
      # Test samples are for inferring, rank samples are for evaluation.
      testIndices = self.partitions[idx][0]
      # testIDs = [p["ID"] for p in
      #   [self.patterns[testIdx] for testIdx in testIndices]]
      rankIndices = self.partitions[idx][1]
      # rankIDs = [p["ID"] for p in
      #   [self.patterns[rankIdx] for rankIdx in rankIndices]]

      testIDs = [d[2] for d in
        [self.dataDict[testIdx] for testIdx in testIndices]]
      rankIDs = [d[2] for d in
        [self.dataDict[rankIdx] for rankIdx in rankIndices]]

      return testIDs, rankIDs


  def testBucket(self, bucket, trainIDs, testIDs, rankIDs, idx):
    """
    Use the testing samples to infer distances. The distances for subsequent
    iterations are combined according to self.combine.
    b
    y taking the min.


    @param testIDs      (list)    Unique IDs of samples for inference.
    @param trainIDs     (dict)    Maps sample unique IDs to their prototype
                                  index in KNN space.
    """
    # Query the model for the input patterns, returning a dict w/ distance
    # values for each query.
    distances = OrderedDict()
    for i, pattern in enumerate(self.patterns):
      if pattern["ID"] in testIDs and pattern["ID"] not in distances.keys():
        distances[pattern["ID"]] = self.model.infer(pattern["pattern"])
    assert(sorted(distances.keys()) == sorted(testIDs))

    # summedDistances = numpy.zeros(len(alreadyTrained))  ## mean
    # meanDistances = []  ## mean
    currentBest = numpy.ones(len(trainIDs.keys()))
    bestDistances = []
    no = []
    for ID, dist in distances.iteritems():
      # summedDistances += dist  ## mean
      # meanDistances.append(summedDistances / (n+1.0))  ## mean
      currentBest = numpy.minimum(currentBest, dist)

      # In each iteration, exclude the queried samples.
      for p in trainIDs[ID]:
        no.append(p)
      currentBest[no] = 1.0  # TODO: better way to get rid of these?

      bestDistances.append(currentBest)

    metrics = []
    # for mD in meanDistances:  ## mean
    for bD in bestDistances:
      # metrics.append(getMetricsHD(mD, alreadyTrained, rankIDs))  ## mean
      metrics.append(self.getMetrics(bD, trainIDs, rankIDs))

    if self.verbosity > 0:
      print "====="
      print "Total data samples in KNN space = ", len(trainIDs)
      print "Results for bucket ", self.labelRefs[bucket[0][1]]
      pprint.pprint(metrics)

    return metrics


  @staticmethod
  def getMetrics(distances, alreadyTrained, rankIDs):
    """
    @param distances      (list)    numpy.arrays of distances to KNN prototypes
    @param alreadyTrained (dict)    IDs corresponding to KNN prototype indices
    @param rankIDs        (list)    IDs of the samples we want metrics on
    @return
    """
    rankIndices = numpy.argsort(distances)
    # rankedIDs = [alreadyTrained[i] for i in rankIndices]
    # testRanks = numpy.array([rankedIDs.index(ID) for ID in testIDs])  # TODO: faster way?
    import pdb; pdb.set_trace()
    rankPrototypes = [alreadyTrained[ID] for ID in rankIDs]
    ranks = rankIndices[list(itertools.chain.from_iterable(rankPrototypes))]
    import pdb; pdb.set_trace()
    return {
      "mean": ranks.mean(),
      "lastTP": ranks.max(),
      "firstTP": ranks.min(),  # ideally this would be 0
      "numTop10": len([r for r in ranks if r < 10]),
      "totalRanked": len(rankIDs),
    }


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
