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

from reberGrammar.reberGrammar import generateSequencesNumber



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

  def __init__(self, *args, **kwargs):
    super(DistributedEncoder, self).__init__(*args, **kwargs)

    self.encodings = {}


  def encode(self, symbol):
    if symbol in self.encodings:
      return self.encodings[symbol]

    encoding = numpy.random.random((1, self.num))
    self.encodings[symbol] = encoding

    return encoding


  def randomSymbol(self):
    return random.randrange(self.num, 10000000)


  @staticmethod
  def closest(node, nodes):
    nodes = numpy.array(nodes)
    dist_2 = numpy.sum((nodes - node)**2, axis=2)
    return numpy.argmin(dist_2)


  def classify(self, encoding, num=1):
    # TODO: support num > 1
    idx = self.closest(encoding, self.encodings.values())
    return [self.encodings.keys()[idx]]



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
    if numPredictions == 1:
      self.sequences = [
        [6, 8, 7, 4, 2, 3, 0],
        [6, 3, 4, 2, 7, 8, 5],
        [1, 8, 7, 4, 2, 3, 5],
        [1, 3, 4, 2, 7, 8, 0],
        [0, 9, 7, 8, 5, 3, 4, 1],
        [0, 4, 3, 5, 8, 7, 9, 6],
        [2, 9, 7, 8, 5, 3, 4, 6],
        [2, 4, 3, 5, 8, 7, 9, 1]
      ]
    elif numPredictions == 2:
      self.sequences = [
        [4, 8, 3, 10, 9, 6, 1],
        [4, 6, 9, 10, 3, 8, 7],
        [4, 8, 3, 10, 9, 6, 2],
        [4, 6, 9, 10, 3, 8, 0],
        [5, 8, 3, 10, 9, 6, 0],
        [5, 6, 9, 10, 3, 8, 2],
        [5, 8, 3, 10, 9, 6, 7],
        [5, 6, 9, 10, 3, 8, 1],
        [4, 3, 8, 6, 1, 10, 11, 9],
        [4, 11, 10, 1, 6, 8, 3, 7],
        [4, 3, 8, 6, 1, 10, 11, 2],
        [4, 11, 10, 1, 6, 8, 3, 0],
        [5, 3, 8, 6, 1, 10, 11, 0],
        [5, 11, 10, 1, 6, 8, 3, 2],
        [5, 3, 8, 6, 1, 10, 11, 7],
        [5, 11, 10, 1, 6, 8, 3, 9]
      ]

  def generateSequence(self):
    return list(random.choice(self.sequences))



class Suite(PyExperimentSuite):

  def reset(self, params, repetition):
    if params['encoding'] == 'basic':
      self.encoder = BasicEncoder(params['encoding_num'])
    elif params['encoding'] == 'distributed':
      self.encoder = DistributedEncoder(params['encoding_num'])
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


  def train(self, params):
    n = params['encoding_num']
    net = buildNetwork(n, params['num_cells'], n,
                       hiddenclass=LSTMLayer, bias=True, outputbias=False, recurrent=True)
    net.reset()

    ds = SequentialDataSet(n, n)
    trainer = RPropMinusTrainer(net, dataset=ds)

    for i in xrange(1, len(self.history)):
      if not self.resets[i-1]:
        ds.addSample(self.encoder.encode(self.history[i-1]),
                     self.encoder.encode(self.history[i]))
      if self.resets[i]:
        ds.newSequence()

    if len(self.history) > 1:
      trainer.trainEpochs(params['num_epochs'])
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

      self.currentSequence += self.dataset.generateSequence()

    if iteration < params['compute_after']:
      return None

    if iteration % params['compute_every'] == 0:
      self.computeCounter = params['compute_for']

    if self.computeCounter == 0:
      return None
    else:
      self.computeCounter -= 1

    if (not params['compute_test_mode'] or
        iteration % params['compute_every'] == 0):
      self.net = self.train(params)

    predictions = None

    for i, symbol in enumerate(self.history):
      output = self.net.activate(self.encoder.encode(symbol))
      predictions = self.encoder.classify(output, num=params['num_predictions'])

      if self.resets[i]:
        self.net.reset()

    truth = None if (self.resets[-1] or
                     self.randoms[-1] or
                     len(self.randoms) >= 2 and self.randoms[-2]) else self.currentSequence[0]

    return {"iteration": iteration,
            "current": self.history[-1],
            "reset": self.resets[-1],
            "random": self.randoms[-1],
            "predictions": predictions,
            "truth": truth}



if __name__ == '__main__':
  suite = Suite()
  suite.start()
