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



class Encoder(object):

  def __init__(self, num):
    self.num = num


  def encode(self, symbol):
    pass


  def random(self):
    pass


  def classify(self, encoding):
    pass



class BasicEncoder(Encoder):

  def encode(self, symbol):
    encoding = numpy.zeros(self.num)
    encoding[symbol] = 1
    return encoding


  def random(self):
    return self.encode(random.randint(self.num))


  def classify(self, encoding):
    return numpy.argmax(encoding)



class Dataset(object):

  def generateSequence(self):
    pass



class ReberDataset(Dataset):

  def generateSequence(self):
    pass



class SimpleDataset(Dataset):

  def __init__(self):
    self.sequences = [
      [6, 8, 7, 4, 2, 3, 0],
      [2, 9, 7, 8, 5, 3, 4, 6],
    ]

  def generateSequence(self):
    return list(random.choice(self.sequences))



class Suite(PyExperimentSuite):

  def reset(self, params, repetition):
    if params['encoding'] == 'basic':
      self.encoder = BasicEncoder(params['encoding_num'])
    else:
      raise Exception("Encoder not found")

    if params['dataset'] == 'simple':
      self.dataset = SimpleDataset()
    else:
      raise Exception("Dataset not found")

    self.history = []
    self.predictions = []

    self.currentSequence = []


  def iterate(self, params, repetition, iteration):
    if len(self.currentSequence) == 0:
      self.currentSequence = self.dataset.generateSequence()
    self.history.append(self.currentSequence.pop(0))

    n = params['encoding_num']

    net = buildNetwork(n, params['num_cells'], n,
                       hiddenclass=LSTMLayer, bias=True, outputbias=False, recurrent=True)
    net.reset()

    ds = SequentialDataSet(n, n)
    trainer = RPropMinusTrainer(net, dataset=ds)

    for i in xrange(len(self.history) - 1):
      ds.addSample(self.encoder.encode(self.history[i]),
                   self.encoder.encode(self.history[i+1]))

    if len(self.history) > 1:
      trainer.trainEpochs(params['num_epochs'])
      net.reset()

    prediction = None

    for symbol in self.history:
      output = net.activate(self.encoder.encode(symbol))
      prediction = self.encoder.classify(output)

    self.predictions.append(prediction)

    return {"iteration": iteration,
            "history": self.history,
            "predictions": self.predictions}



if __name__ == '__main__':
  suite = Suite()
  suite.start()
