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

import cPickle as pkl
import numpy
import os

from collections import Counter, namedtuple

from htmresearch.frameworks.nlp.runner import Runner
from htmresearch.frameworks.nlp.classify_htm import ClassificationModelHTM
from htmresearch.support.data_split import KFolds
from htmresearch.support.network_text_data_generator import NetworkDataGenerator

try:
  import simplejson as json
except ImportError:
  import json



class HTMRunner(Runner):
  """
  Class to run the HTM NLP experiments with the specified data and evaluation
  metrics.
  """

  def __init__(self,
               dataPath,
               resultsDir,
               experimentName,
               experimentType,
               networkConfigPath=None,
               generateData=True,
               votingMethod="most",
               classificationFile="",
               seed=42,
               **kwargs):
    """
    @param networkConfigPath  (str)    Path to JSON specifying network params.
    @param generateData       (bool)   Whether or not we need to generate data.
    @param votingMethod       (str)    Classify with "last" token's score or
                                       "most" frequent of the sequence.
    @param classificationFile (str)    Path to JSON that maps labels to ids.

    See base class constructor for the other parameters.
    """
    if networkConfigPath is None:
      raise RuntimeError("Need to specify a network configuration JSON.")

    super(HTMRunner, self).__init__(
      dataPath, resultsDir, experimentName, experimentType, **kwargs)

    self.networkConfig = self._getNetworkConfig(networkConfigPath)
    self.model = None
    self.votingMethod = votingMethod
    self.dataFiles = []
    self.actualLabels = None

    if classificationFile == "" and not generateData:
      raise ValueError("Must give classificationFile if not generating data")
    self.classificationFile = classificationFile

    # Setup data now in order to init the network model. If you want to
    # specify data params, just call setupNetData() again later.
    self.setupNetData(generateData=generateData, seed=seed)


  @staticmethod
  def _getNetworkConfig(networkConfigPath):
    try:
      with open(networkConfigPath, "rb") as fin:
        return json.load(fin)
    except IOError as e:
      print "Could not find network configuration JSON at \'{}\'.".format(
        networkConfigPath)
      raise e


  def initModel(self, trial=0):
    """
    Load or instantiate the classification model. Assumes network data is
    already setup.
    """
    if self.loadPath:
      with open(self.loadPath, "rb") as f:
        self.model = pkl.load(f)
      # TODO: uncomment once we can save TPRegion; do we need this?
      # networkFile = self.model.network
      # self.model.network = Network(networkFile)
      print "Model loaded from \'{0}\'.".format(self.loadPath)
    else:
      print "Creating HTM classification model..."
      self.model = ClassificationModelHTM(self.networkConfig,
                                          self.dataFiles[trial],
                                          retinaScaling=self.retinaScaling,
                                          retina=self.retina,
                                          apiKey=self.apiKey,
                                          verbosity=self.verbosity,
                                          numLabels=self.numClasses,
                                          modelDir=self.modelDir,
                                          prepData=False)


  def setupNetData(self, generateData=False, seed=42, preprocess=False, **kwargs):
    """
    Generate the data in network API format if necessary. self.dataFiles is
    populated with the paths of network data files, one for each experiment
    iteration.

    Look at runner.py (setupData) and network_text_data_generator.py (split) for
    the parameters.
    """
    # TODO: logic here is confusing (a lot of if-statements), so maybe cleanup.
    if self.experimentType == "k-folds":
      splits = self.folds
    elif self.experimentType == "incremental":
      splits = len(self.trainSizes)

    if generateData:
      self.generateNetworkDataFiles(splits, seed, preprocess, **kwargs)
    else:
      # Use the input file for each trial; maintains the order of samples.
      self.dataFiles = [self.dataPath] * len(self.trainSizes)

    if self.numClasses > 0:
      # Setup labels data objects
      self.actualLabels = [self._getClassifications(i) for i in xrange(splits)]
      self.mapLabelRefs()


  def generateNetworkDataFiles(self, splits, seed, preprocess, **kwargs):
    # TODO: use model.prepData()?
    ndg = NetworkDataGenerator()
    self.dataDict = ndg.split(
      filePath=self.dataPath, numLabels=self.numClasses, textPreprocess=preprocess, **kwargs)

    filename, ext = os.path.splitext(self.dataPath)
    self.classificationFile = "{}_categories.json".format(filename)

    # Generate one data file for each experiment iteration.
    if self.experimentType == "k-folds" and not self.orderedSplit:
      # only randomize the data order once for k-folds cross validation
      ndg.randomizeData(seed)
    for i in xrange(splits):
      if self.experimentType != "k-folds" and not self.orderedSplit:
        ndg.randomizeData(seed)
        seed += 1
      # ext='.csv'
      dataFile = "{}_network_{}{}".format(filename, i, ext)
      ndg.saveData(dataFile, self.classificationFile)
      self.dataFiles.append(dataFile)

    if self.verbosity > 0:
      print "{} file(s) generated at {}".format(len(self.dataFiles),
        self.dataFiles)
      print "Classification JSON is at: {}".format(self.classificationFile)


  def _getClassifications(self, iteration):
    """
    Get the classifications for a particular iteration.
    @param iteration  (int)       Iteration of the experiment.
    @return           (list)      List of list of ids of classifications for a
                                  sample.
    """
    dataFile = self.dataFiles[iteration]
    classifications = NetworkDataGenerator.getClassifications(dataFile)
    return [[int(c) for c in classes.strip().split(" ")]
      for classes in classifications]


  def mapLabelRefs(self):
    """Get the mapping from label strings to the corresponding ints."""
    try:
      with open(self.classificationFile, "r") as f:
        labelToId = json.load(f)
    except IOError as e:
      print "Must have a valid classification JSON file"
      raise e

    # Convert the dict of strings -> ids to a list of strings ordered by id
    self.labelRefs = zip(*sorted(labelToId.iteritems(), key=lambda x: x[1]))[0]
    for recordNumber, data in self.dataDict.iteritems():
      self.dataDict[recordNumber] = (data[0], numpy.array(
        [self.labelRefs.index(label) for label in data[1]]), data[2])


  def resetModel(self, trial=0):
    """
    Load or instantiate the classification model; network API doesn't support
    resetting."""
    self.initModel(trial=trial)
    # TODO: change to same as Runner:
    #   self.model.resetModel()
    #   otherwise you're creating a new model instance twice each experiment


  def setupData(self, _):
    """Passthrough b/c network data generation was done upfront."""
    pass


  def encodeSamples(self):
    """Passthrough b/c the network encodes the samples."""
    pass


  def training(self, trial):
    """
    Train the network on all the tokens in the training set for a particular
    trial.
    @param trial      (int)       current trial number
    """
    if self.verbosity > 0:
      i = 0
      indices = []
      for numTokens in self.partitions[trial][0]:
        indices.append(i)
        i += numTokens
      print ("\tRunner selects to train on sequences starting at indices {}.".
            format(indices))

    for numTokens in self.partitions[trial][0]:
      self.model.trainModel(iterations=numTokens)


  def testing(self, trial, seed):
    """
    Test the network on the test set for a particular trial and store the
    results
    @param trial      (int)       trial count
    """
    if self.verbosity > 0:
      i = sum(self.partitions[trial][0])
      indices = []
      for numTokens in self.partitions[trial][1]:
        indices.append(i)
        i += numTokens
      print ("\tRunner selects to test on sequences starting at indices "
             "{}".format(indices))

    results = ([], [])
    testIndex = len(self.partitions[trial][0])
    for numTokens in self.partitions[trial][1]:
      predictions = []
      activations = []
      for _ in xrange(numTokens):
        import pdb; pdb.set_trace()
        predicted, active = self.model.testModel(seed)
        activations.append(active)
        predictions.append(predicted)
      winningPredictions = self._selectWinners(predictions, activations)

      # TODO: switch to standard (expected, actual) format
      results[0].append(winningPredictions)
      results[1].append(self.actualLabels[trial][testIndex])
      testIndex += 1

    # Prepare data for writeOutClassifications
    trainIdx = range(len(self.partitions[trial][0]))
    testIdx = range(len(self.partitions[trial][0]),
      len(self.partitions[trial][0]) + len(self.partitions[trial][1]))
    self.partitions[trial] = (trainIdx, testIdx)
    self.samples = NetworkDataGenerator.getSamples(self.dataFiles[trial])

    self.results.append(results)


  def _selectWinners(self, predictions, activations):
    """
    Selects the final classifications for the predictions.  Voting
    method=="last" means the predictions of the last sample are used. Voting
    method=="most" means the most frequent sample is used.
    @param predictions    (list)    List of list of possible classifications
    @return               (list)    List of winning classifications
    """
    if self.votingMethod == "last":
      return predictions[-1]
    elif self.votingMethod == "most":
      counter = Counter()
      for p in predictions:
        counter.update(p)
      return zip(*counter.most_common(self.numClasses))[0]
    elif self.votingmethod == "tree":
      import pdb; pdb.set_trace()
      # Calculate overlaps

      # Which are nodes?

      # Classify by the nodes


    else:
      raise ValueError("voting method must be either \'last\' or \'most\'")


  def partitionIndices(self, _):
    """
    Sets self.partitions for the number of tokens for each sample in the
    training and test sets.

    The order of sequences is already specified by the network data files; if
    generated by the experiment, these are in order or randomized as specified
    by the orderedSplit arg.
    """
    if self.experimentType == "k-folds":
      for fold in xrange(self.folds):
        dataFile = self.dataFiles[fold]
        numTokens = NetworkDataGenerator.getNumberOfTokens(dataFile)
        self.partitions = KFolds(self.folds).split(numTokens, randomize=False)
    else:
      for trial, split in enumerate(self.trainSizes):
        dataFile = self.dataFiles[trial]
        numTokens = NetworkDataGenerator.getNumberOfTokens(dataFile)
        self.partitions.append((numTokens[:split], numTokens[split:]))

  # TODO
  # This method is to partition data for which regions are learning, as in the
  # sequence classification experiments.
  def partitionLearning(self):
    """
    Find the number of partitions for the input data based on a specific
    networkConfig.

    @return partitions: (list of namedtuples) Region names and index at which the
      region is to begin learning. The final partition is reserved as a test set.
    """
    Partition = namedtuple("Partition", "partName index")

    # Add regions to partition list in order of learning.
    regionConfigs = ("spRegionConfig", "tmRegionConfig", "tpRegionConfig",
      "classifierRegionConfig")
    partitions = []

    return


  def writeOutClassifications(self):
    # TODO: implement this method after updating HTM network models and runner
    # per nupic.research #277
    pass
