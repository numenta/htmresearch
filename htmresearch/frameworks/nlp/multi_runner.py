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
import random

from htmresearch.frameworks.nlp.runner import Runner
from htmresearch.support.csv_helper import readCSV, readDir
from htmresearch.support.text_preprocess import TextPreprocess



class MultiRunner(Runner):
  """
  Class to run the baseline NLP experiments with the specified data, models,
  text processing, and evaluation metrics. This requires dataPath to be a directory
  """

  def __init__(self,
               dataPath,
               resultsDir,
               experimentName,
               load,
               modelName,
               modelModuleName,
               numClasses,
               plots,
               orderedSplit,
               trainSizes,
               verbosity,
               test=None):
    """
    @param dataPath         (str)     Path to raw data files for the experiment.
    @param resultsDir       (str)     Directory where for the results metrics.
    @param experimentName   (str)     Experiment name, used for saving results.
    @param load             (bool)    True if a serialized model is to be
                                      loaded.
    @param modelName        (str)     Name of nupic.fluent Model subclass.
    @param modeModuleName   (str)     Model module -- location of the subclass.
    @param numClasses       (int)     Number of classes (labels) per sample.
    @param plots            (int)     Specifies plotting of evaluation metrics.
    @param orderedSplit     (bool)    Indicates method for splitting train/test
                                      samples; False is random, True is ordered.
    @param trainSizes       (list)    Number of samples to use in training, per
                                      trial.
    @param verbosity        (int)     Greater value prints out more progress.
    @param test             (str)     Path to raw data file for testing or None

    """
    self.test = test

    self.testDict = None
    self.testSamples = None
    self.testPatterns = None

    super(MultiRunner, self).__init__(dataPath, resultsDir, experimentName, load,
                                      modelName, modelModuleName, numClasses, plots,
                                      orderedSplit, trainSizes, verbosity)

    
  def _mapLabelRefs(self):
    """Replace the label strings in self.dataDict with corresponding ints."""
    self.labelRefs = self.dataDict.keys()

    for category, samples in self.dataDict.iteritems():
      for idx, data in samples.iteritems():
        comment, labels = data
        self.dataDict[category][idx] = (comment, numpy.array(
            [self.labelRefs.index(label) for label in labels]))

    if self.testDict:
      for idx, data in self.testDict.iteritems():
        comment, labels = data
        self.testDict[idx] = (comment, numpy.array(
            [self.labelRefs.index(label) for label in labels]))


  def _preprocess(self, preprocess):
    """Tokenize the samples, with or without preprocessing."""
    texter = TextPreprocess()
    if preprocess:
      self.samples = {category: [(texter.tokenize(data[0],
                                                  ignoreCommon=100,
                                                  removeStrings=["identifier deleted]"],
                                                  correctSpell=True), data[1], idx)
                      for idx, data in samples.iteritems()]
                      for category, samples in self.dataDict.iteritems()}

      if self.testDict:
        self.testSamples = [(texter.tokenize(data[0],
                                            ignoreCommon=100,
                                            removeStrings=["identifier deleted]"],
                                            correctSpell=True), data[1], idx)
                            for idx, data in self.testDict.iteritems()]
    else:
      self.samples = {category: [(texter.tokenize(data[0]), data[1], idx)
                      for idx, data in samples.iteritems()]
                      for category, samples in self.dataDict.iteritems()}

      if self.testDict:
        self.testSamples = [(texter.tokenize(data[0]), data[1], idx)
                            for idx, data in self.testDict.iteritems()]


  def setupData(self, preprocess=False):
    """
    Get the data from a directory and preprocess if specified.
    One index in labelIdx implies the model will train on a single
    classification per sample.
    """
    self.dataDict = readDir(self.dataPath, self.numClasses, True)

    if self.test:
      self.testDict = readCSV(self.test, numLabels=self.numClasses)

    minCategorySize = min(map(len, self.dataDict.values()))
    if not (isinstance(self.trainSizes, list) or
        all([0 <= size <= minCategorySize for size in self.trainSizes])):
      raise ValueError("Invalid size(s) for training set.")

    self._mapLabelRefs()

    self._preprocess(preprocess)

    if self.verbosity > 1:
      for i, s in enumerate(self.samples): print i, s


  def encodeSamples(self):
    """
    Encode the text samples into bitmap patterns. The
    encoded patterns are stored in a dict along with their corresponding class
    labels.
    """
    self.patterns = {category: [{"pattern": self.model.encodePattern(sample),
                                "labels": labels,
                                "id": idx} for sample, labels, idx in samples]
                    for category, samples in self.samples.iteritems()}

    if self.testSamples:
      self.testPatterns = [{"pattern": self.model.encodePattern(sample),
                            "labels": labels,
                            "id": idx} for sample, labels, idx in self.testSamples]


  def training(self, trial):
    """
    Train the model one-by-one on each pattern specified in this trials
    partition of indices. Models' training methods require the sample and label
    to be in a list.
    """
    if self.verbosity > 0:
      print ("\tRunner selects to train on sample(s) {}".
        format(self.partitions[trial][0]))

    for labelRef, categoryIndices in enumerate(self.partitions[trial][0]):
      category = self.labelRefs[labelRef]
      for i in categoryIndices:
        self.model.trainModel([self.patterns[category][i]["pattern"]],
                              [self.patterns[category][i]["labels"]])


  def testing(self, trial):
    if self.verbosity > 0:
      print ("\tRunner selects to test on sample(s) {}".
        format(self.partitions[trial][1]))

    results = ([], [])
    if self.testPatterns:
      # Test the file that was provided
      for i in self.partitions[trial][1]:
        predicted = self.model.testModel(self.testPatterns[i]["pattern"])
        results[0].append(predicted)
        results[1].append(self.testPatterns[i]["labels"])
    else:
      flattenedPartition = []
      for labelRef, categoryIndices in enumerate(self.partitions[trial][1]):
        category = self.labelRefs[labelRef]
        for i in categoryIndices:
          predicted = self.model.testModel(self.patterns[category][i]["pattern"])
          results[0].append(predicted)
          results[1].append(self.patterns[category][i]["labels"])
        flattenedPartition += categoryIndices
      # The indices need to be flattened so classification_model can print them
      self.partitions[trial] = (self.partitions[trial][0], flattenedPartition)

    self.results.append(results)


  def partitionIndices(self, split, trial):
    """
    Returns train and test indices.

    TODO: use StandardSplit in data_split.py
    """
    trainIdxs = []
    testIdxs = []
    trainIdSet = set()
    for i, label in enumerate(self.labelRefs):
      length = len(self.samples[label])
      if self.orderedSplit:
        trainIdx = range(split)
        testIdx = range(split, length)
      else:
        # Randomly sampled, not repeated
        trainIdx = random.sample(xrange(length), split)
        testIdx = [i for i in xrange(length) if i not in trainIdx]
      trainIdxs.append(trainIdx)
      testIdxs.append(testIdx)

      trainIdSet.update([self.patterns[label][i]["id"] for i in trainIdx])

    if self.test:
      testIdxs = [i for i, testInstance in enumerate(self.testPatterns)
          if testInstance["id"] not in trainIdSet]

    return (trainIdxs, testIdxs)
