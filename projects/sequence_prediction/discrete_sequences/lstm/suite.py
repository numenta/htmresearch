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

from expsuite import PyExperimentSuite
import numpy
from pybrain.datasets import SequentialDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer

from htmresearch.support.reberGrammar import generateSequencesNumber



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

  def __init__(self, num, maxValue=None, minValue=None, classifyWithRandom=None):
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

    encoding = (self.maxValue - self.minValue) * numpy.random.random((1, self.num)) + self.minValue
    self.encodings[symbol] = encoding

    return encoding


  def randomSymbol(self):
    return random.randrange(self.num, 10000000)


  @staticmethod
  def closest(node, nodes, num):
    nodes = numpy.array(nodes)
    dist_2 = numpy.sum((nodes - node)**2, axis=2)
    return dist_2.flatten().argsort()[:num]


  def classify(self, encoding, num=1):
    encodings = {k:v for (k, v) in self.encodings.iteritems() if k <= self.num or self.classifyWithRandom}

    if len(encodings) == 0:
      return []

    idx = self.closest(encoding, encodings.values(), num)
    return [encodings.keys()[i] for i in idx]



class Dataset(object):

  def generateSequence(self):
    pass



class ReberDataset(Dataset):

  def __init__(self, maxLength=None):
    if maxLength is None:
      raise "maxLength not specified"

    self.maxLength = maxLength


  def generateSequence(self):
    return generateSequencesNumber(self.maxLength)[0]



class SimpleDataset(Dataset):

  def __init__(self):
    self.sequences = [
      [6, 8, 7, 4, 2, 3, 0],
      [2, 9, 7, 8, 5, 3, 4, 6],
    ]

  def generateSequence(self):
    return list(random.choice(self.sequences))



class HighOrderDataset(Dataset):

  def __init__(self, numPredictions=1):
    self.numPredictions = numPredictions


  def sequences(self, numPredictions, perturbed):
    if numPredictions == 1:
      if perturbed:
        return [
          [6, 8, 7, 4, 2, 3, 5],
          [1, 8, 7, 4, 2, 3, 0],
          [6, 3, 4, 2, 7, 8, 0],
          [1, 3, 4, 2, 7, 8, 5],
          [0, 9, 7, 8, 5, 3, 4, 6],
          [2, 9, 7, 8, 5, 3, 4, 1],
          [0, 4, 3, 5, 8, 7, 9, 1],
          [2, 4, 3, 5, 8, 7, 9, 6]
        ]
      else:
        return [
          [6, 8, 7, 4, 2, 3, 0],
          [1, 8, 7, 4, 2, 3, 5],
          [6, 3, 4, 2, 7, 8, 5],
          [1, 3, 4, 2, 7, 8, 0],
          [0, 9, 7, 8, 5, 3, 4, 1],
          [2, 9, 7, 8, 5, 3, 4, 6],
          [0, 4, 3, 5, 8, 7, 9, 6],
          [2, 4, 3, 5, 8, 7, 9, 1]
        ]

    elif numPredictions == 2:
      if perturbed:
        return [
          [4, 8, 3, 10, 9, 6, 0],
          [4, 8, 3, 10, 9, 6, 7],
          [5, 8, 3, 10, 9, 6, 1],
          [5, 8, 3, 10, 9, 6, 2],
          [4, 6, 9, 10, 3, 8, 2],
          [4, 6, 9, 10, 3, 8, 1],
          [5, 6, 9, 10, 3, 8, 7],
          [5, 6, 9, 10, 3, 8, 0],
          [4, 3, 8, 6, 1, 10, 11, 0],
          [4, 3, 8, 6, 1, 10, 11, 7],
          [5, 3, 8, 6, 1, 10, 11, 9],
          [5, 3, 8, 6, 1, 10, 11, 2],
          [4, 11, 10, 1, 6, 8, 3, 2],
          [4, 11, 10, 1, 6, 8, 3, 9],
          [5, 11, 10, 1, 6, 8, 3, 7],
          [5, 11, 10, 1, 6, 8, 3, 0]
        ]
      else:
        return [
          [4, 8, 3, 10, 9, 6, 1],
          [4, 8, 3, 10, 9, 6, 2],
          [5, 8, 3, 10, 9, 6, 0],
          [5, 8, 3, 10, 9, 6, 7],
          [4, 6, 9, 10, 3, 8, 7],
          [4, 6, 9, 10, 3, 8, 0],
          [5, 6, 9, 10, 3, 8, 2],
          [5, 6, 9, 10, 3, 8, 1],
          [4, 3, 8, 6, 1, 10, 11, 9],
          [4, 3, 8, 6, 1, 10, 11, 2],
          [5, 3, 8, 6, 1, 10, 11, 0],
          [5, 3, 8, 6, 1, 10, 11, 7],
          [4, 11, 10, 1, 6, 8, 3, 7],
          [4, 11, 10, 1, 6, 8, 3, 0],
          [5, 11, 10, 1, 6, 8, 3, 2],
          [5, 11, 10, 1, 6, 8, 3, 9]
        ]


  def generateSequence(self, perturbed=False):
    return list(random.choice(self.sequences(self.numPredictions, perturbed)))



class Suite(PyExperimentSuite):

  def reset(self, params, repetition):
    if params['encoding'] == 'basic':
      self.encoder = BasicEncoder(params['encoding_num'])
    elif params['encoding'] == 'distributed':
      self.encoder = DistributedEncoder(params['encoding_num'],
                                        maxValue=params['encoding_max'],
                                        minValue=params['encoding_min'],
                                        classifyWithRandom=params['classify_with_random'])
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
    self.currentSequence = self.dataset.generateSequence()

    self.net = None


  def window(self, data, params):
    start = max(0, len(data) - params['learning_window'])
    return data[start:]


  def train(self, params):
    n = params['encoding_num']
    net = buildNetwork(n, params['num_cells'], n,
                       hiddenclass=LSTMLayer,
                       bias=True,
                       outputbias=params['output_bias'],
                       recurrent=True)
    net.reset()

    ds = SequentialDataSet(n, n)
    trainer = RPropMinusTrainer(net, dataset=ds)

    history = self.window(self.history, params)
    resets = self.window(self.resets, params)

    for i in xrange(1, len(history)):
      if not resets[i-1]:
        ds.addSample(self.encoder.encode(history[i-1]),
                     self.encoder.encode(history[i]))
      if resets[i]:
        ds.newSequence()

    if len(history) > 1:
      trainer.trainEpochs(params['num_epochs'])
      net.reset()

    for i in xrange(len(history) - 1):
      symbol = history[i]
      output = net.activate(self.encoder.encode(symbol))
      predictions = self.encoder.classify(output, num=params['num_predictions'])

      if resets[i]:
        net.reset()

    return net


  def iterate(self, params, repetition, iteration):
    self.history.append(self.currentSequence.pop(0))

    resetFlag = (len(self.currentSequence) == 0 and
                 params['separate_sequences_with'] == 'reset')
    self.resets.append(resetFlag)

    randomFlag = (len(self.currentSequence) == 0 and
                  params['separate_sequences_with'] == 'random')
    self.randoms.append(randomFlag)

    if len(self.currentSequence) == 0:
      if randomFlag:
        self.currentSequence.append(self.encoder.randomSymbol())

      if iteration > params['perturb_after']:
        sequence = self.dataset.generateSequence(perturbed=True)
      else:
        sequence = self.dataset.generateSequence()

      self.currentSequence += sequence

    if iteration < params['compute_after']:
      return None

    if iteration % params['compute_every'] == 0:
      self.computeCounter = params['compute_for']

    if self.computeCounter == 0:
      return None
    else:
      self.computeCounter -= 1

    train = (not params['compute_test_mode'] or
             iteration % params['compute_every'] == 0)

    if train:
      self.net = self.train(params)

    history = self.window(self.history, params)
    resets = self.window(self.resets, params)

    if resets[-1]:
      self.net.reset()

    symbol = history[-1]
    output = self.net.activate(self.encoder.encode(symbol))
    predictions = self.encoder.classify(output, num=params['num_predictions'])

    truth = None if (self.resets[-1] or
                     self.randoms[-1] or
                     len(self.randoms) >= 2 and self.randoms[-2]) else self.currentSequence[0]

    return {"current": self.history[-1],
            "reset": self.resets[-1],
            "random": self.randoms[-1],
            "train": train,
            "predictions": predictions,
            "truth": truth}



if __name__ == '__main__':
  suite = Suite()
  suite.start()
