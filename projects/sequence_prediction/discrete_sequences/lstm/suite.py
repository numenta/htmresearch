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
from pybrain.datasets import SequentialDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer
from htmresearch.support.sequence_prediction_dataset import ReberDataset
from htmresearch.support.sequence_prediction_dataset import SimpleDataset
from htmresearch.support.sequence_prediction_dataset import HighOrderDataset



class Encoder(object):
  def __init__(self, num):
    self.num = num


  def encode(self, symbol):
    pass


  def random(self):
    pass


  def classify(self, encoding, num=1):
    pass



class BasicEncoder(Encoder):
  def encode(self, symbol):
    encoding = numpy.zeros(self.num)
    encoding[symbol] = 1
    return encoding


  def randomSymbol(self):
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

    self.encodings = {}


  def encode(self, symbol):
    if symbol in self.encodings:
      return self.encodings[symbol]

    encoding = (self.maxValue - self.minValue) * numpy.random.random(
      (1, self.num)) + self.minValue
    self.encodings[symbol] = encoding

    return encoding


  def randomSymbol(self):
    return random.randrange(self.num, 10000000)


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
      self.dataset = HighOrderDataset(numPredictions=params['num_predictions'])
    else:
      raise Exception("Dataset not found")

    self.computeCounter = 0

    self.history = []
    self.resets = []
    self.randoms = []

    self.currentSequence = []
    self.targetPrediction = []
    self.replenishSequence(params, iteration=0)

    self.net = None
    self.sequenceCounter = 0

  def window(self, data, params):
    start = max(0, len(data) - params['learning_window'])
    return data[start:]


  def train(self, params):
    """
    Train LSTM network on buffered dataset history
    After training, run LSTM on history[:-1] to get the state correct
    :param params:
    :return:
    """
    n = params['encoding_num']
    net = buildNetwork(n, params['num_cells'], n,
                       hiddenclass=LSTMLayer,
                       bias=True,
                       outputbias=params['output_bias'],
                       recurrent=True)
    net.reset()

    # prepare training dataset
    ds = SequentialDataSet(n, n)
    trainer = RPropMinusTrainer(net,
                                dataset=ds,
                                verbose=params['verbosity'] > 0)

    history = self.window(self.history, params)
    resets = self.window(self.resets, params)

    for i in xrange(1, len(history)):
      if not resets[i - 1]:
        ds.addSample(self.encoder.encode(history[i - 1]),
                     self.encoder.encode(history[i]))
      if resets[i]:
        ds.newSequence()

    if len(history) > 1:
      trainer.trainEpochs(params['num_epochs'])
      net.reset()

    # run network on buffered dataset after training to get the state right
    for i in xrange(len(history) - 1):
      symbol = history[i]
      output = net.activate(self.encoder.encode(symbol))
      predictions = self.encoder.classify(output, num=params['num_predictions'])

      if resets[i]:
        net.reset()

    return net


  def killCells(self, killCellPercent):
    """
    kill a fraction of LSTM cells from the network
    :param killCellPercent:
    :return:
    """
    if killCellPercent <= 0:
      return

    inputLayer = self.net['in']
    lstmLayer = self.net['hidden0']

    numLSTMCell = lstmLayer.outdim
    numDead = round(killCellPercent * numLSTMCell)
    zombiePermutation = numpy.random.permutation(numLSTMCell)
    deadCells = zombiePermutation[0:numDead]

    # remove connections from input layer to dead LSTM cells
    connectionInputToHidden = self.net.connections[inputLayer][0]
    weightInputToHidden = reshape(connectionInputToHidden.params,
                                  (connectionInputToHidden.outdim,
                                   connectionInputToHidden.indim))

    for cell in deadCells:
      for dim in range(4):
        weightInputToHidden[dim * numLSTMCell + cell, :] *= 0

    newParams = reshape(weightInputToHidden,
                        (connectionInputToHidden.paramdim,))
    self.net.connections[inputLayer][0]._setParameters(
      newParams, connectionInputToHidden.owner)

    # remove dead connections within LSTM layer
    connectionHiddenToHidden = self.net.recurrentConns[0]
    weightHiddenToHidden = reshape(connectionHiddenToHidden.params,
                                   (connectionHiddenToHidden.outdim,
                                    connectionHiddenToHidden.indim))

    for cell in deadCells:
      weightHiddenToHidden[:, cell] *= 0

    newParams = reshape(weightHiddenToHidden,
                        (connectionHiddenToHidden.paramdim,))
    self.net.recurrentConns[0]._setParameters(
      newParams, connectionHiddenToHidden.owner)

    # remove connections from dead LSTM cell to output layer
    connectionHiddenToOutput = self.net.connections[lstmLayer][0]
    weightHiddenToOutput = reshape(connectionHiddenToOutput.params,
                                   (connectionHiddenToOutput.outdim,
                                    connectionHiddenToOutput.indim))
    for cell in deadCells:
      weightHiddenToOutput[:, cell] *= 0

    newParams = reshape(weightHiddenToOutput,
                        (connectionHiddenToOutput.paramdim,))
    self.net.connections[lstmLayer][0]._setParameters(
      newParams, connectionHiddenToOutput.owner)


  def replenishSequence(self, params, iteration):
    if iteration > params['perturb_after']:
      sequence, target = self.dataset.generateSequence(perturbed=True)
    else:
      sequence, target = self.dataset.generateSequence()

    if iteration > params['inject_noise_after']:
      injectNoiseAt = random.randint(1, 3)
      sequence[injectNoiseAt] = self.encoder.randomSymbol()

    if params['separate_sequences_with'] == 'random':
      sequence.append(self.encoder.randomSymbol())
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


  def iterate(self, params, repetition, iteration):
    element = self.currentSequence.pop(0)
    target = self.targetPrediction.pop(0)

    # update buffered dataset
    self.history.append(element)

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

    # kill cells
    killCell = False
    if iteration == params['kill_cell_after']:
      killCell = True
      self.killCells(params['kill_cell_percent'])

    # reset compute counter
    if iteration % params['compute_every'] == 0:
      self.computeCounter = params['compute_for']

    if self.computeCounter == 0 or iteration < params['compute_after']:
      computeLSTM = False
    else:
      computeLSTM = True

    if computeLSTM:
      self.computeCounter -= 1

      train = (not params['compute_test_mode'] or
               iteration % params['compute_every'] == 0)

      if train:
        if params['verbosity'] > 0:
          print "Training LSTM at iteration {}".format(iteration)

        self.net = self.train(params)

      # run LSTM on the latest data record

      output = self.net.activate(self.encoder.encode(element))
      predictions = self.encoder.classify(output, num=params['num_predictions'])

      correct = self.check_prediction(predictions, target)

      if params['verbosity'] > 0:
        print ("iteration: {0} \t"
               "current: {1} \t"
               "predictions: {2} \t"
               "truth: {3} \t"
               "correct: {4} \t").format(
          iteration, element, predictions, target, correct)

      if self.resets[-1]:
        if params['verbosity'] > 0:
          print "Reset LSTM at iteration {}".format(iteration)
        self.net.reset()

      return {"current": element,
              "reset": self.resets[-1],
              "random": self.randoms[-1],
              "train": train,
              "predictions": predictions,
              "truth": target,
              "killCell": killCell,
              "sequenceCounter": self.sequenceCounter}



if __name__ == '__main__':
  suite = Suite()
  suite.start()
