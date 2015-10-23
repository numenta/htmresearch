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
import itertools
import numpy
import os
import random

from collections import defaultdict

from htmresearch.encoders import EncoderTypes
from htmresearch.frameworks.nlp.classify_endpoint import (
  ClassificationModelEndpoint)
from htmresearch.frameworks.nlp.classify_fingerprint import (
  ClassificationModelFingerprint)
from htmresearch.frameworks.nlp.classify_keywords import (
  ClassificationModelKeywords)
from htmresearch.support.csv_helper import readCSV, writeFromDict
from htmresearch.support.data_split import KFolds


_MODEL_MAPPING = {
  "CioWordFingerprint": ClassificationModelFingerprint,
  "CioDocumentFingerprint": ClassificationModelFingerprint,
  "Keywords": ClassificationModelKeywords,
  "CioEndpoint": ClassificationModelEndpoint,
  }



class Runner(object):
  """
  Class to run the baseline NLP experiments with the specified data, models,
  text processing, and evaluation metrics.
  """

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
               numClasses=3,
               plots=0,
               orderedSplit=False,
               folds=None,
               trainSizes=None,
               verbosity=0):
    """
    @param dataPath         (str)     Path to raw data file for the experiment.
    @param resultsDir       (str)     Directory where for the results metrics.
    @param experimentName   (str)     Experiment name, used for saving results.
    @param experimentType   (str)     Either 'incremental' or 'k-folds'.
    @param modelName        (str)     Name of nlp model subclass.
    @param retinaScaling    (float)   For scaling dimensions of Cio encoders.
    @param retina           (str)     Name of Cio retina for encodings.
    @param apiKey           (str)     Key for Cio API.
    @param loadPath         (str)     Path to serialized model for loading.
    @param numClasses       (int)     Number of classes (labels) per sample.
    @param plots            (int)     Specifies plotting of evaluation metrics.
    @param orderedSplit     (bool)    Indicates method for splitting train/test
                                      samples; False is random, True is ordered.
    @param folds            (int)     For k-folds experiment, number of cross
                                      validation folds.
    @param trainSizes       (list)    For incremental experiment, number of 
                                      samples to use in training, per trial.
    @param verbosity        (int)     Greater value prints out more progress.
    """
    if experimentType not in ("incremental", "k-folds"):
      raise ValueError("Experiment type not recognized.")
    if (folds is None) and (trainSizes is None):
      raise ValueError("Runner needs to know how to split the data.")
    self.experimentType = experimentType
    self.folds = folds
    self.trainSizes = trainSizes
    self.dataPath = dataPath
    self.resultsDir = resultsDir
    self.experimentName = experimentName
    self.loadPath = loadPath
    self.modelName = modelName
    self.numClasses = numClasses
    self.plots = plots
    self.orderedSplit = orderedSplit
    self.retinaScaling = retinaScaling
    self.retina = retina
    self.apiKey = apiKey
#    self.trainSizes = trainSizes if trainSizes else []
    self.verbosity = verbosity

    self.modelDir = os.path.join(
      self.resultsDir, self.experimentName, self.modelName)
    if not os.path.exists(self.modelDir):
      os.makedirs(self.modelDir)

    if self.plots:
      from htmresearch.support.nlp_classification_plotting import PlotNLP
      self.plotter = PlotNLP()

    self.dataDict = None
    self.labels = None
    self.labelRefs = None
    self.partitions = []
    self.samples = {}
    self.patterns = None
    self.results = []
    self.model = None


  def initModel(self, modelName):
    """Load or instantiate the classification model."""
    if self.loadPath:
      self.model = self.loadModel()
    else:
      self.model = self._createModel(modelName)


  def _createModel(self, modelName):
    """Return an instantiated model."""
    modelCls = _MODEL_MAPPING.get(modelName, None)

    if modelCls is None:
      raise ValueError("Could not instantiate model \'{}\'.".format(modelName))

    # TODO: remove these if blocks and just use the else; either specify the Cio
    # FP type elsewhere, or split Word and Doc into separate classes.

    if modelName == "CioWordFingerprint":
      return modelCls(verbosity=self.verbosity,
                      numLabels=self.numClasses,
                      modelDir=self.modelDir,
                      fingerprintType=EncoderTypes.word,
                      retinaScaling=self.retinaScaling,
                      retina=self.retina,
                      apiKey=self.apiKey)

    elif modelName == "CioDocumentFingerprint":
      return modelCls(verbosity=self.verbosity,
                      numLabels=self.numClasses,
                      modelDir=self.modelDir,
                      fingerprintType=EncoderTypes.document,
                      retinaScaling=self.retinaScaling,
                      retina=self.retina,
                      apiKey=self.apiKey)

    else:
      return modelCls(verbosity=self.verbosity,
                      numLabels=self.numClasses,
                      modelDir=self.modelDir)


  def loadModel(self):
    """Load the serialized model."""
    try:
      with open(self.loadPath, "rb") as f:
        model = pkl.load(f)
      print "Model loaded from \'{}\'.".format(self.loadPath)
      return model
    except IOError as e:
      print "Could not load model from \'{}\'.".format(self.loadPath)
      raise e


  def resetModel(self, _):
    self.model.resetModel()


  def saveModel(self):
    self.model.saveModel()


  def _mapLabelRefs(self):
    """Replace the label strings in self.dataDict with corresponding ints."""
    self.labelRefs = [label for label in set(
      itertools.chain.from_iterable([x[1] for x in self.dataDict.values()]))]

    for uniqueID, data in self.dataDict.iteritems():
      self.dataDict[uniqueID] = (data[0], numpy.array(
        [self.labelRefs.index(label) for label in data[1]]))


  def setupData(self, preprocess=False):
    """
    Get the data from CSV and preprocess if specified. The call to readCSV()
    assumes a specific CSV format, detailed in its docstring.

    @param preprocess   (bool)    Whether or not to preprocess the data when
                                  reading in samples.
    """
    self.dataDict = readCSV(self.dataPath, numLabels=self.numClasses)

    if self.experimentType == "incremental":
      # stop now if the data won't work for the specified experiment
      if (not isinstance(self.trainSizes, list) or not
          all([0 <= size <= len(self.dataDict) for size in self.trainSizes])):
        raise ValueError("Invalid size(s) for training set(s).")

    self._mapLabelRefs()

    self.samples = self.model.prepData(self.dataDict, preprocess)

    self.encodeSamples()

    if self.verbosity > 1:
      for i, s in self.samples.iteritems():
        print i, s


  def encodeSamples(self):
    self.patterns = self.model.encodeSamples(self.samples)


  def runExperiment(self, seed=42):
    """Train and test the model for each trial specified by self.splitting."""
    self.partitionIndices(seed)

    for i, _ in enumerate(self.partitions):
      self.resetModel(i)
      if self.verbosity > 0:
        print "\tTraining and testing for run {}.".format(i)
      self._training(i)
      self._testing(i)


  def partitionIndices(self, seed=42):
    """
    Partitions list of two-tuples of train and test indices for each trial.
    """
    if self.experimentType == "k-folds":
      self.partitions = KFolds(self.folds).split(range(len(self.samples)), randomize=(not self.orderedSplit), seed=seed)
    else:
      # TODO: use StandardSplit in data_split.py
      length = len(self.samples)
      if self.orderedSplit:
        for split in self.trainSizes:
          trainIndices = range(split)
          testIndices = range(split, length)
          self.partitions.append((trainIndices, testIndices))
      else:
        # randomly sampled, not repeated
        random.seed(seed)
        for split in self.trainSizes:
          trainIndices = random.sample(xrange(length), split)
          testIndices = [i for i in xrange(length) if i not in trainIndices]
          self.partitions.append((trainIndices, testIndices))


  def _training(self, trial):
    """
    Train the model one-by-one on each pattern specified in this trials
    partition of indices. Models' training methods require the sample and label
    to be in a list.
    """
    if self.verbosity > 0:
      print ("\tRunner selects to train on sample(s) {}".format(
        self.partitions[trial][0]))

    for i in self.partitions[trial][0]:
      self.model.trainModel(i)


  def _testing(self, trial):
    if self.verbosity > 0:
      print ("\tRunner selects to test on sample(s) {}".format(
        self.partitions[trial][1]))

    results = ([], [])
    for i in self.partitions[trial][1]:
      predicted = self.model.testModel(i)
      results[0].append(predicted)
      results[1].append(self.patterns[i]["labels"])

    self.results.append(results)


  def writeOutClassifications(self):
    """Write the samples, actual, and predicted classes to a CSV."""
    headers = ("Tokenized sample", "Actual", "Predicted")
    
    if self.experimentType == "k-folds":
      splits = range(self.folds)
    else:
      splits = self.trainSizes
    
    for trial in splits:
      resultsDict = defaultdict(list)
      for i, sampleNum in enumerate(self.partitions[trial][1]):
        # Loop through the indices in the test set of this trial.
        sample = self.samples.values()[sampleNum][0]
        pred = sorted([self.labelRefs[j] for j in self.results[trial][0][i]])
        actual = sorted([self.labelRefs[j] for j in self.results[trial][1][i]])
        resultsDict[sampleNum] = (sampleNum, sample, actual, pred)

      resultsPath = os.path.join(self.model.modelDir,
                                 "results_trial" + str(trial) + ".csv")
      writeFromDict(resultsDict, headers, resultsPath)


  def calculateResults(self):
    """
    Calculate evaluation metrics from the result classifications.
    """
    # TODO: pass intended CM results to plotter.plotConfusionMatrix()
    resultCalcs = []
    for i, sampleNum in enumerate(self.partitions):
      if self.verbosity > 0:
        self.printTrialReport(i, sampleNum[1])
      resultCalcs.append(self.model.evaluateResults(
        self.results[i], self.labelRefs, sampleNum[i][1]))

    trainSizes = [len(x[0]) for x in self.partitions]
    self.printFinalReport(trainSizes, [r[0] for r in resultCalcs])

    if self.plots:
      trialAccuracies = self._calculateTrialAccuracies()
      classificationAccuracies = self._calculateClassificationAccuracies(
        trialAccuracies)

      self.plotter.plotCategoryAccuracies(trialAccuracies, self.trainSizes)
      self.plotter.plotCumulativeAccuracies(
        classificationAccuracies, self.trainSizes)

      if self.plots > 1:
        # Plot extra evaluation figures -- confusion matrix.
        self.plotter.plotConfusionMatrix(
          self.setupConfusionMatrices(resultCalcs))

    return resultCalcs


  def printTrialReport(self, trial, idx):
    """Print columns for sample #, actual label, and predicted label."""
    template = "{0:<10}|{1:<55}|{2:<55}"
    print "Classification results for the trial:"
    print template.format("#", "Actual", "Predicted")
    for i in xrange(len(self.results[trial][0])):
      if not any(self.results[trial][0][i]):
        # No predicted classes for this sample.
        print template.format(
          idx[i],
          [self.labelRefs[label] for label in self.results[trial][1][i]],
          "(none)")
      else:
        print template.format(
          idx[i],
          [self.labelRefs[label] for label in self.results[trial][1][i]],
          [self.labelRefs[label] for label in self.results[trial][0][i]])


  @staticmethod
  def printFinalReport(trainSizes, accuracies):
    """Prints result accuracies."""
    template = "{0:<20}|{1:<10}"
    print "---------- RESULTS ----------"
    print template.format("Size of training set", "Accuracy")
    for size, acc in itertools.izip(trainSizes, accuracies):
      print template.format(size, acc)


  def _calculateTrialAccuracies(self):
    """
    @return trialAccuracies     (defaultdict)   Items are defaultdicts, one for
        each size of the training set. Inner defaultdicts keys are
        categories, with numpy array values that contain one accuracy value for
        each trial.
    """
    # To handle multiple trials of the same size:
    # trialSize -> (category -> list of accuracies)
    trialAccuracies = defaultdict(lambda: defaultdict(lambda: numpy.ndarray(0)))
    for result, size in itertools.izip(self.results, self.trainSizes):
      accuracies = self.model.calculateClassificationResults(result)
      for label, acc in accuracies:
        category = self.labelRefs[label]
        accList = trialAccuracies[size][category]
        trialAccuracies[size][category] = numpy.append(accList, acc)

    return trialAccuracies


  def _calculateClassificationAccuracies(self, trialAccuracies):
    """
    @param trialAccuracies            (defaultdict)   Please see the description
        in self._calculateClassificationAccuracies().

    @return classificationAccuracies  (defaultdict)   Keys are classification
        categories, with multiple numpy arrays as values -- one for each size of
        training sets, with one accuracy value for each run of that training set
        size.
    """
    # Need the accuracies to be ordered for the plot
    trials = sorted(set(self.trainSizes))
    # category -> list of list of accuracies
    classificationAccuracies = defaultdict(list)
    for trial in trials:
      accuracies = trialAccuracies[trial]
      for label, acc in accuracies.iteritems():
        classificationAccuracies[label].append(acc)

    return classificationAccuracies


  def validateExperiment(self, expectationFilePath):
    """Returns accuracy of predicted labels against expected labels."""
    dataDict = readCSV(expectationFilePath, numLabels=self.numClasses)

    accuracies = numpy.zeros((len(self.results)))
    for i, trial in enumerate(self.results):
      for j, predictionList in enumerate(trial[0]):
        predictions = [self.labelRefs[p] for p in predictionList]
        if predictions == []:
          predictions = ["(none)"]
        expected = dataDict.items()[j+self.trainSizes[i]][1]

        accuracies[i] += (float(len(set(predictions) & set(expected[1])))
                          / len(expected[1]))

      accuracies[i] = accuracies[i] / len(trial[0])

    return accuracies


  def evaluateCumulativeResults(self, intermResults):
    """
    Cumulative statistics for the outputs of evaluateTrialResults().

    @param intermResults      (list)          List of returned results from
                                              evaluateTrialResults().
    @return                   (dict)          Returns a dictionary with entries
                                              for max, mean, and min accuracies,
                                              and the mean confusion matrix.
    """
    accuracy = []
    cm = numpy.zeros((intermResults[0][1].shape))

    # Find mean, max, and min values for the metrics.
    for result in intermResults:
      accuracy.append(result[0])
      cm = numpy.add(cm, result[1])

    results = {"max_accuracy":max(accuracy),
               "mean_accuracy":sum(accuracy)/float(len(accuracy)),
               "min_accuracy":min(accuracy),
               "total_cm":cm}

    if self.verbosity > 0:
      self._printCumulativeReport(results)

    return results


  @staticmethod
  def _printCumulativeReport(results):
    """
    Prints results as returned by evaluateFinalResults() after several trials.
    """
    print "max, mean, min accuracies = "
    print "{0:.3f}, {1:.3f}, {2:.3f}".format(
      results["max_accuracy"], results["mean_accuracy"],
      results["min_accuracy"])
    print "total confusion matrix =\n", results["total_cm"]
