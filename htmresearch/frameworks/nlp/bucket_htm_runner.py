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
import numpy
import os
import pprint

from collections import OrderedDict

from htmresearch.frameworks.nlp.bucket_runner import BucketRunner
from htmresearch.frameworks.nlp.htm_runner import HTMRunner
from htmresearch.support.csv_helper import bucketCSVs
from htmresearch.support.data_split import Buckets
from htmresearch.support.network_text_data_generator import NetworkDataGenerator


class BucketHTMRunner(BucketRunner, HTMRunner):
  """Buckets experiment methods for HTM network models."""

  def __init__(self, trainingReps=1, *args, **kwargs):
    """
    Param for trainingReps specifies how many times the network runs through
    the data with all regions (but the classifier) learning; the classifier only
    learns on the final rep (i.e. only the final learned representations make up
    the KNN space). Add'l params go to constructors of super classes.
    """
    self.trainingReps = trainingReps

    self.bucketFiles = []
    self.classifiedSeqIds = []
    self.numTokens = []

    super(BucketHTMRunner, self).__init__(*args, **kwargs)


  def setupNetData(
    self, generateData=True, seed=42, preprocess=False, **kwargs):
    """
    Resulting network data files created:
      - One for each bucket
      - One for each training rep, where samples are not repeated in a given
      file. Each samples is given its own category (_category = _sequenceId).

    The classification json is saved when generating the final training file.
    """
    if generateData:
      ndg = NetworkDataGenerator()
      self.dataDict = ndg.split(
        filePath=self.dataPath, numLabels=1, textPreprocess=preprocess,
        **kwargs)

      filename, ext = os.path.splitext(self.dataPath)
      self.classificationFile = "{}_categories.json".format(filename)

      # Generate test data files: one network data file for each bucket.
      bucketFilePaths = bucketCSVs(self.dataPath)
      for bucketFile in bucketFilePaths:
        ndg.reset()
        ndg.split(
          filePath=bucketFile, numLabels=1, textPreprocess=preprocess, **kwargs)
        bucketFileName, ext = os.path.splitext(bucketFile)
        if not self.orderedSplit:
          # the sequences will be written to the file in random order
          ndg.randomizeData(seed)
        dataFile = "{}_network{}".format(bucketFileName, ext)
        ndg.saveData(dataFile, self.classificationFile)  # the classification file here gets (correctly) overwritten later
        self.bucketFiles.append(dataFile)

      # Generate training data file(s).
      self.trainingDicts = []
      uniqueDataDict = OrderedDict()
      included = []
      seqID = 0
      for dataEntry in self.dataDict.values():
        uniqueID = dataEntry[2]
        if uniqueID not in included:
          # skip over the samples that are repeated in multiple buckets
          uniqueDataDict[seqID] = dataEntry
          included.append(uniqueID)
          seqID += 1
      self.trainingDicts.append(uniqueDataDict)

      ndg.reset()
      ndg.split(
        dataDict=uniqueDataDict, numLabels=1, textPreprocess=preprocess,
        **kwargs)
      for rep in xrange(self.trainingReps):
        # use a different file for each training rep
        if not self.orderedSplit:
          ndg.randomizeData(seed)
        ndg.stripCategories()  # replace the categories w/ seqId
        dataFile = "{}_network_training_{}{}".format(filename, rep, ext)
        ndg.saveData(dataFile, self.classificationFile)
        self.dataFiles.append(dataFile)

      # TODO: maybe add a method (and arg) for removing all these data files

    else:
      # TODO (only if needed)
      raise NotImplementedError("Must generate data.")

    # labels references match the classification json
    self.mapLabelRefs()


  def partitionIndices(self, seed=42, numInference=10):
    """
    Sets self.partitions for the buckets' querying and ranking sets. The
    corresponding numbers of tokens for each sequence are stored in
    self.numTokens.

    The order of sequences is already specified by the network data files; if
    generated by the experiment, these are in order or randomized as specified
    by the orderedSplit arg.
    """
    super(BucketHTMRunner, self).partitionIndices(
      seed=seed, numInference=numInference)

    # Get the number of tokens in each bucket file so the network knows how many
    # iterations to run. The order of buckets in self.bucketFiles is not
    # necessarily the same
    ndg = NetworkDataGenerator()
    for dataFile in self.bucketFiles:
      self.numTokens.append(ndg.getNumberOfTokens(dataFile))


  def train(self):
    """
    Train the network regions on the entire dataset.
    There should be one datafile for each training rep in self.dataFiles, where
    every data sample (i.e. sequence) appears only once in each file.
    """
    # TODO: ignore patterns < minSparsity (= 0.9 * unionSparsity)
    if self.trainingReps != len(self.dataFiles):
      raise RuntimeError("Mismatch between the number of specified training "
        "reps and the number of data files (should be 1:1).")

    for dataFile in self.dataFiles:
      if self.verbosity > 0:
        print "Running all the data through the netwrok for training..."
      self.model.swapRecordStream(dataFile)
      numTokens = NetworkDataGenerator().getNumberOfTokens(dataFile)
      n = sum(numTokens)
      self.model.trainNetwork(n)

    # Populate the classifier space by running through the current data file;
    # learning (in other regions) is turned off by the model.
    if self.verbosity > 1:
      print "Populating the classifier with all of the sequences."
    self.classifiedSeqIds = self.model.classifyNetwork(n)


  def populateKNN(self):
    """
    The network already populated the KNN at the end of training, but we still
    need to map the sample IDs to their prototypes.

    Note: this subclass already handled duplicates when generating the network
    data files in setupNetData().
    """
    prototypeMap = OrderedDict()
    for i, seqId in enumerate(self.classifiedSeqIds):
      uniqueID = self.trainingDicts[0][seqId][2]
      # use a trainingDict here b/c in self.dataDict we can have multiple seqIds
      # point to the same uniqueID
      if uniqueID in prototypeMap:
        prototypeMap[uniqueID].append(i)
      else:
        prototypeMap[uniqueID] = [i]

    return prototypeMap


  def prepBucket(self, idx):
    queryIDs = [self.buckets[idx][i][2] for i in self.partitions[idx][0]]
    rankIDs = [self.buckets[idx][i][2] for i in self.partitions[idx][1]]
    return queryIDs, rankIDs


  def testBucket(self, bucketNum, prototypeMap, _, rankIDs):
    """
    Run the query samples for this bucket through the network, getting the
    inferred distances to each prototype.
    """
    protoIDs = OrderedDict()  # TODO: maybe move this into populateKN() (like BucketRunner)?
    for k, v in prototypeMap.iteritems():
      protoIDs[k] = self.classifiedSeqIds[v[0]]

    bucketFile = self.bucketFiles[bucketNum]
    distances = OrderedDict()
    for i, idx in enumerate(self.partitions[bucketNum][0]):
      # idx tells us which sample in this bucket we want to run, and the
      # corresponding numTokens value tells the network how many steps to run
      # in order to process the sequence.
      numTokens = self.numTokens[bucketNum][idx]
      sampleDist = self.model.inferNetwork(numTokens, fileRecord=bucketFile)
      inferID = self.buckets[bucketNum][idx][2]
      distances[inferID] = sampleDist[:len(protoIDs)]

    accumulatedDistances = self.setupDistances(distances, protoIDs)

    self.calcMetrics(accumulatedDistances, protoIDs, rankIDs)
