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

import random
import numbers
import numpy

from scipy import reshape, dot, outer

from expsuite import PyExperimentSuite
from htmresearch.support.sequence_prediction_dataset import ReberDataset
from htmresearch.support.sequence_prediction_dataset import SimpleDataset
from htmresearch.support.sequence_prediction_dataset import HighOrderDataset

from htmresearch.algorithms.online_extreme_learning_machine import OSELM

def initializeELMnet(nDimInput, nDimOutput, numNeurons=10):
  # Build ELM network with nDim input units,
  # numNeurons hidden units (LSTM cells) and nDimOutput cells

  net = OSELM(nDimInput, nDimOutput,
              numHiddenNeurons=numNeurons, activationFunction='sig')
  return net



class Encoder(object):
  def __init__(self, num):
    self.num = num


  def encode(self, symbol):
    pass


  def randomSymbol(self):
    pass


  def classify(self, encoding, num=1):
    pass



class BasicEncoder(Encoder):
  def encode(self, symbol):
    encoding = numpy.zeros(self.num)
    encoding[symbol] = 1
    return encoding


  def randomSymbol(self, seed):
    random.seed(seed)
    return random.randrange(self.num)


  def classify(self, encoding, num=1):
    idx = numpy.argpartition(encoding, -num)[-num:]
    return idx[numpy.argsort(encoding[idx])][::-1].tolist()



class DistributedEncoder(Encoder):
  """
  TODO: We are using self.num to limit how many elements are non-random,
  where we should instead have a separate parameter that can be set.

  This can cause bugs if self.num is too small, and classifyWithRandom = False.

  MUST FIX IF WE WANT TO USE classifyWithRandom = False!
  """


  def __init__(self, num, maxValue=None, minValue=None,
               classifyWithRandom=None):
    super(DistributedEncoder, self).__init__(num)

    if maxValue is None or minValue is None:
      raise "maxValue and minValue are required"

    if classifyWithRandom is None:
      raise "classifyWithRandom is required"

    self.maxValue = maxValue
    self.minValue = minValue
    self.classifyWithRandom = classifyWithRandom
    self.seed = 42
    self.encodings = {}


  def encode(self, symbol):
    # numpy.random.seed(self.seed)
    if symbol in self.encodings:
      return self.encodings[symbol]
    encoding = (self.maxValue - self.minValue) * numpy.random.random(
      (1, self.num)) + self.minValue
    self.encodings[symbol] = encoding

    return encoding


  def randomSymbol(self, seed):
    random.seed(seed)
    return random.randrange(self.num, self.num+50000)


  @staticmethod
  def closest(node, nodes, num):
    nodes = numpy.array(nodes)
    dist_2 = numpy.sum((nodes - node) ** 2, axis=2)
    return dist_2.flatten().argsort()[:num]


  def classify(self, encoding, num=1):
    encodings = {k: v for (k, v) in self.encodings.iteritems() if
                 k <= self.num or self.classifyWithRandom}

    if len(encodings) == 0:
      return []

    idx = self.closest(encoding, encodings.values(), num)
    return [encodings.keys()[i] for i in idx]



class Suite(PyExperimentSuite):
  def reset(self, params, repetition):
    random.seed(params['seed'])

    if params['encoding'] == 'basic':
      self.encoder = BasicEncoder(params['encoding_num'])
    elif params['encoding'] == 'distributed':
      self.encoder = DistributedEncoder(params['encoding_num'],
                                        maxValue=params['encoding_max'],
                                        minValue=params['encoding_min'],
                                        classifyWithRandom=params[
                                          'classify_with_random'])
    else:
      raise Exception("Encoder not found")

    if params['dataset'] == 'simple':
      self.dataset = SimpleDataset()
    elif params['dataset'] == 'reber':
      self.dataset = ReberDataset(maxLength=params['max_length'])
    elif params['dataset'] == 'high-order':
      self.dataset = HighOrderDataset(numPredictions=params['num_predictions'],
                                      seed=params['seed'])
    else:
      raise Exception("Dataset not found")

    self.numLags = params['num_lags']

    self.history = []
    self.resets = []


    self.finishInitializeX = False
    self.randoms = []

    self.currentSequence = []
    self.targetPrediction = []
    self.replenishSequence(params, iteration=0)

    self.net = initializeELMnet(params['encoding_num'] * params['num_lags'],
                                params['encoding_num'],
                                numNeurons=params['num_cells'])
    self.sequenceCounter = 0


  def window(self, data, windowSize):
    start = max(0, len(data) - windowSize)
    return data[start:]


  def replenishSequence(self, params, iteration):
    if iteration > params['perturb_after']:
      sequence, target = self.dataset.generateSequence(params['seed']+iteration,
                                                       perturbed=True)
    else:
      sequence, target = self.dataset.generateSequence(params['seed']+iteration)

    if (iteration > params['inject_noise_after'] and
            iteration < params['stop_inject_noise_after']):
      injectNoiseAt = random.randint(1, 3)
      sequence[injectNoiseAt] = self.encoder.randomSymbol()

    if params['separate_sequences_with'] == 'random':
      sequence.append(self.encoder.randomSymbol(seed=params['seed']+iteration))
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
        correct = targets in topPredictions
      else:
        correct = True
        for prediction in topPredictions:
           correct = correct and (prediction in targets)
    return correct


  def killCells(self, killCellPercent):
    """
    kill a fraction of LSTM cells from the network
    """
    if killCellPercent <= 0:
      return

    numHiddenNeurons = self.net.numHiddenNeurons

    numDead = round(killCellPercent * numHiddenNeurons)
    zombiePermutation = numpy.random.permutation(numHiddenNeurons)
    deadCells = zombiePermutation[0:numDead]
    liveCells = zombiePermutation[numDead:]

    self.net.inputWeights = self.net.inputWeights[liveCells, :]
    self.net.bias = self.net.bias[:, liveCells]
    self.net.beta = self.net.beta[liveCells, :]
    self.net.M = self.net.M[liveCells, liveCells]
    self.net.numHiddenNeurons = numHiddenNeurons - numDead


  def iterate(self, params, repetition, iteration):
    currentElement = self.currentSequence.pop(0)
    target = self.targetPrediction.pop(0)

    # update buffered dataset
    self.history.append(currentElement)


    # whether there will be a reset signal after the current record
    resetFlag = (len(self.currentSequence) == 0 and
                 params['separate_sequences_with'] == 'reset')
    self.resets.append(resetFlag)

    # whether there will be a random symbol after the current record
    randomFlag = (len(self.currentSequence) == 1 and
                  params['separate_sequences_with'] == 'random')

    self.randoms.append(randomFlag)

    if len(self.currentSequence) == 0:
      self.replenishSequence(params, iteration)
      self.sequenceCounter += 1

    # # kill cells
    killCell = False
    if iteration == params['kill_cell_after']:
      killCell = True
      self.killCells(params['kill_cell_percent'])

    # reset compute counter
    if iteration < params['compute_after']:
      computeELM = False
    else:
      computeELM = True

    if computeELM:
      n = params['encoding_num']

      if self.finishInitializeX is False:
        # run initialization phase of OS-ELM
        NT = params['compute_after']
        features = numpy.zeros(shape=(NT, n*params['num_lags']))
        targets = numpy.zeros(shape=(NT, n))

        history = self.window(self.history, NT)

        for i in range(NT):
          targets[i, :] = self.encoder.encode(history[i])

        for lags in xrange(params['num_lags']):
          shiftTargets = numpy.roll(targets, lags, axis=0)
          shiftTargets[:lags, :] = 0
          features[:, lags*n:(lags+1)*n] = shiftTargets

        self.net.initializePhase(features[:, :], targets[:, :])
        if iteration > params['compute_after']:
          self.finishInitializeX = True
      else:
        # run sequential learning phase
        targets = numpy.zeros((1, params['encoding_num']))
        targets[0, :] = self.encoder.encode(self.history[-1])

        features = numpy.zeros((1, params['encoding_num'] * params['num_lags']))
        for lags in xrange(params['num_lags']):
          features[0, lags*n:(lags+1)*n] = self.encoder.encode(
            self.history[-1-(lags+1)])

      if iteration < params['stop_training_after']:
        self.net.train(features, targets)

      # run ELM on the latest data record
      currentFeatures = numpy.zeros((1, params['encoding_num'] * params['num_lags']))
      for lags in xrange(params['num_lags']):
        currentFeatures[0, lags*n:(lags+1)*n] = self.encoder.encode(
          self.history[-1-lags])

      output = self.net.predict(currentFeatures)

      predictions = self.encoder.classify(output[0],
                                          num=params['num_predictions'])

      correct = self.check_prediction(predictions, target)

      if params['verbosity'] > 0:
        print ("iteration: {0} \t"
               "current: {1} \t"
               "predictions: {2} \t"
               "truth: {3} \t"
               "correct: {4} \t").format(
          iteration, currentElement, predictions, target, correct)

      return {"current": currentElement,
              "random": self.randoms[-1],
              "predictions": predictions,
              "truth": target,
              "killCell": killCell,
              "sequenceCounter": self.sequenceCounter}



if __name__ == '__main__':
  suite = Suite()
  suite.start()
