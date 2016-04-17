#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
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
import numbers
import random

import numpy
from expsuite import PyExperimentSuite

from nupic.frameworks.opf.modelfactory import ModelFactory
# from nupic.algorithms.sdr_classifier import SDRClassifier

from htmresearch.algorithms.faulty_temporal_memory_shim import MonitoredFaultyTPShim
from htmresearch.support.sequence_prediction_dataset import ReberDataset
from htmresearch.support.sequence_prediction_dataset import SimpleDataset
from htmresearch.support.sequence_prediction_dataset import HighOrderDataset
from htmresearch.support.sequence_prediction_dataset import LongHighOrderDataset


NUM_SYMBOLS = 16
RANDOM_END = 50000

MODEL_PARAMS = {
  "model": "CLA",
  "version": 1,
  "predictAheadTime": None,
  "modelParams": {
    "inferenceType": "TemporalMultiStep",
    "sensorParams": {
      "verbosity" : 0,
      "encoders": {
        "element": {
          "fieldname": u"element",
          "name": u"element",
          "type": "SDRCategoryEncoder",
          "categoryList": range(max(RANDOM_END, NUM_SYMBOLS)),
          "n": 2048,
          "w": 41
        }
      },
      "sensorAutoReset" : None,
    },
      "spEnable": False,
      "spParams": {
        "spVerbosity" : 0,
        "globalInhibition": 1,
        "columnCount": 2048,
        "inputWidth": 0,
        "numActiveColumnsPerInhArea": 40,
        "seed": 1956,
        "columnDimensions": 0.5,
        "synPermConnected": 0.1,
        "synPermActiveInc": 0.1,
        "synPermInactiveDec": 0.01,
        "maxBoost": 0.0
    },
    "tpEnable" : True,
    "tpParams": {
      "verbosity": 0,
        "columnCount": 2048,
        "cellsPerColumn": 32,
        "inputWidth": 2048,
        "seed": 1960,
        "temporalImp": "monitored_tm_py",
        "newSynapseCount": 32,
        "maxSynapsesPerSegment": 128,
        "maxSegmentsPerCell": 128,
        "initialPerm": 0.21,
        "connectedPerm": 0.50,
        "permanenceInc": 0.1,
        "permanenceDec": 0.1,
        "predictedSegmentDecrement": 0.02,
        "globalDecay": 0.0,
        "maxAge": 0,
        "minThreshold": 15,
        "activationThreshold": 15,
        "outputType": "normal",
        "pamLength": 1,
      },
      "clParams": {
        "implementation": "cpp",
        "regionName" : "CLAClassifierRegion",
        "clVerbosity" : 0,
        "alpha": 0.0001,
        "steps": "1",
      },
      "trainSPNetOnlyIfRequested": False,
    },
}



def getEncoderMapping(model, numSymbols):
  encoder = model._getEncoder().encoders[0][1]
  mapping = dict()

  for i in range(numSymbols):
    mapping[i] = set(encoder.encode(i).nonzero()[0])

  return mapping



def classify(mapping, activeColumns, numPredictions):
  scores = [(len(encoding & activeColumns), i) for i, encoding in mapping.iteritems()]
  random.shuffle(scores)  # break ties randomly
  return [i for _, i in sorted(scores, reverse=True)[:numPredictions]]



class Suite(PyExperimentSuite):

  def reset(self, params, repetition):
    random.seed(params['seed'])

    if params['dataset'] == 'simple':
      self.dataset = SimpleDataset()
    elif params['dataset'] == 'reber':
      self.dataset = ReberDataset(maxLength=params['max_length'])
    elif params['dataset'] == 'high-order':
      self.dataset = HighOrderDataset(numPredictions=params['num_predictions'],
                                      seed=params['seed'])
      print "Sequence dataset: "
      print " Symbol Number {}".format(self.dataset.numSymbols)
      for seq in self.dataset.sequences:
        print seq

    elif params['dataset'] == 'high-order-long':
      self.dataset = LongHighOrderDataset(params['sequence_length'],
                                          seed=params['seed'])
      print "Sequence dataset: "
      print " Symbol Number {}".format(self.dataset.numSymbols)
      for seq in self.dataset.sequences:
        print seq
    else:
      raise Exception("Dataset not found")

    self.randomStart = self.dataset.numSymbols + 1
    self.randomEnd = self.randomStart + 5000

    MODEL_PARAMS['modelParams']['sensorParams']['encoders']['element']\
      ['categoryList'] = range(self.randomEnd)

    # if not os.path.exists(resultsDir):
    #   os.makedirs(resultsDir)
    # self.resultsFile = open(os.path.join(resultsDir, "0.log"), 'w')
    if params['verbosity'] > 0:
      print " initializing HTM model..."
    self.model = ModelFactory.create(MODEL_PARAMS)
    self.model.enableInference({"predictedField": "element"})
    # self.classifier = SDRClassifier(steps=[1], alpha=0.001)

    if params['kill_cell_percent'] > 0:
      # a hack to use faulty temporal memory instead
      self.model._getTPRegion().getSelf()._tfdr = MonitoredFaultyTPShim(
        numberOfCols=2048,
        cellsPerColumn=32,
        newSynapseCount=32,
        maxSynapsesPerSegment=128,
        maxSegmentsPerCell=128,
        initialPerm=0.21,
        connectedPerm=0.50,
        permanenceInc=0.10,
        permanenceDec=0.10,
        predictedSegmentDecrement=0.01,
        minThreshold=15,
        activationThreshold=15,
        seed=1960,
      )

    self.mapping = getEncoderMapping(self.model, self.dataset.numSymbols)

    self.numPredictedActiveCells = []
    self.numPredictedInactiveCells = []
    self.numUnpredictedActiveColumns = []

    self.currentSequence = []
    self.targetPrediction = []
    self.replenish_sequence(params, iteration=0)

    self.resets = []
    self.randoms = []
    self.verbosity = 1
    self.sequenceCounter = 0


  def replenish_sequence(self, params, iteration):
    if iteration > params['perturb_after']:
      print "PERTURBING"
      sequence, target = self.dataset.generateSequence(params['seed']+iteration,
                                                       perturbed=True)
    else:
      sequence, target = self.dataset.generateSequence(params['seed']+iteration)

    if (iteration > params['inject_noise_after'] and
        iteration < params['stop_inject_noise_after']):
      injectNoiseAt = random.randint(1, 3)
      sequence[injectNoiseAt] = random.randrange(self.randomStart, self.randomEnd)

      if params['verbosity'] > 0:
        print "injectNoise ", sequence[injectNoiseAt],  " at: ", injectNoiseAt

    # separate sequences with random elements
    if params['separate_sequences_with'] == 'random':
      random.seed(params['seed']+iteration)
      sequence.append(random.randrange(self.randomStart, self.randomEnd))
      target.append(None)

    if params['verbosity'] > 0:
      print "Add sequence to buffer"
      print "sequence: ", sequence
      print "target: ", target

    self.currentSequence += sequence
    self.targetPrediction += target


  def check_prediction(self, topPredictions, targets):
    if targets is None:
      correct = None
    else:
      if isinstance(targets, numbers.Number):
        # single target, multiple predictions
        correct = targets in topPredictions
      else:
        # multiple targets, multiple predictions
        correct = True
        for prediction in topPredictions:
           correct = correct and (prediction in targets)
    return correct


  def iterate(self, params, repetition, iteration):
    currentElement = self.currentSequence.pop(0)
    target = self.targetPrediction.pop(0)

    # whether there will be a reset signal after the current record
    resetFlag = (len(self.currentSequence) == 0 and
                 params['separate_sequences_with'] == 'reset')
    self.resets.append(resetFlag)

    # whether there will be a random symbol after the current record
    randomFlag = (len(self.currentSequence) == 1 and
                  params['separate_sequences_with'] == 'random')

    self.randoms.append(randomFlag)

    killCell = False
    if iteration == params['kill_cell_after'] and params['kill_cell_percent'] > 0:
      killCell = True
      tm = self.model._getTPRegion().getSelf()._tfdr
      tm.killCells(percent=params['kill_cell_percent'])
      self.model.disableLearning()

    result = self.model.run({"element": currentElement})
    tm = self.model._getTPRegion().getSelf()._tfdr

    tm.mmClearHistory()

    # Try use SDR classifier to classify active (not predicted) cells
    # The results is similar as classifying the predicted cells
    # classLabel = min(currentElement, self.dataset.numSymbols)
    # classification = {'bucketIdx': classLabel, 'actValue': classLabel}
    # result = self.classifier.compute(iteration, list(tm.activeCells),
    #                                  classification,
    #                                  learn=True, infer=True)
    # topPredictionsSDRClassifier = sorted(zip(result[1], result["actualValues"]),
    #                                      reverse=True)[0]
    # topPredictionsSDRClassifier = [topPredictionsSDRClassifier[1]]
    topPredictionsSDRClassifier = None

    # activeColumns = set([tm.columnForCell(cell) for cell in tm.activeCells])
    # print "active columns: "
    # print activeColumns
    # print "sdr mapping current: "
    # print self.mapping[element]
    # print "sdr mapping next: "
    # print self.mapping[target]
    # Use custom classifier (uses predicted cells to make predictions)
    predictiveColumns = set([tm.columnForCell(cell) for cell in tm.predictiveCells])
    topPredictions = classify(
      self.mapping, predictiveColumns, params['num_predictions'])

    # correct = self.check_prediction(topPredictions, target)
    truth = target
    if params['separate_sequences_with'] == 'random':
      if (self.randoms[-1] or
                len(self.randoms) >= 2 and self.randoms[-2]):
        truth = None

    correct = None if truth is None else (truth in topPredictions)

    data = {"iteration": iteration,
            "current": currentElement,
            "reset": resetFlag,
            "random": randomFlag,
            "train": True,
            "predictions": topPredictions,
            "predictionsSDR": topPredictionsSDRClassifier,
            "truth": target,
            "sequenceCounter": self.sequenceCounter}

    if params['verbosity'] > 0:
      print ("iteration: {0} \t"
             "current: {1} \t"
             "predictions: {2} \t"
             "predictions SDR: {3} \t"
             "truth: {4} \t"
             "correct: {5} \t"
             "predict column: {6}").format(
        iteration, currentElement, topPredictions, topPredictionsSDRClassifier,
        target, correct, len(predictiveColumns))

    if len(self.currentSequence) == 0:
      self.replenish_sequence(params, iteration)
      self.sequenceCounter += 1

    if self.resets[-1]:
      if params['verbosity'] > 0:
        print "Reset TM at iteration {}".format(iteration)
      tm.reset()

    return data


if __name__ == '__main__':
  suite = Suite()
  suite.start()

