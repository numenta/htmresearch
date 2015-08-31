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

import operator
import random
import time

from matplotlib import pyplot as plt
import numpy as np

from pybrain.datasets import SequentialDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer

from predict import generateSequences
from plot import plotAccuracy



NUM_PREDICTIONS = 1



vectors = {}



def num2vec(num, nDim):
  if num in vectors:
    return vectors[num]

  sample = np.random.random((1,nDim))
  # if num < 100:
  #   vectors[num] = sample
  vectors[num] = sample
  return sample

def seq2vec(sequence, nDim):
  nSample = len(sequence)
  seq_vec = np.zeros((nSample, nDim))
  for i in xrange(nSample):
    seq_vec[i] = num2vec(sequence[i], nDim)
  return seq_vec

def closest_node(node, nodes):
  nodes = np.array(nodes)
  dist_2 = np.sum((nodes - node)**2, axis=2)
  return np.argmin(dist_2)

def classify(netActivation):
  idx = closest_node(netActivation, vectors.values())
  return vectors.keys()[idx]


def initializeLSTMnet(nDim, nLSTMcells=10):
  # Build LSTM network with nDim input units, nLSTMcells hidden units (LSTM cells) and nDim output cells
  net = buildNetwork(nDim, nLSTMcells, nDim,
                     hiddenclass=LSTMLayer, bias=True, outputbias=False, recurrent=True)
  return net



if __name__ == "__main__":
  sequences = generateSequences(NUM_PREDICTIONS)
  # nDim = max([len(sequence) for sequence in sequences]) + 2  # TODO: Why 2?
  nDim = 100

  from pylab import rcParams
  rcParams.update({'figure.autolayout': True})
  rcParams.update({'figure.facecolor': 'white'})
  rcParams.update({'ytick.labelsize': 8})

  # for i in xrange(len(sequences)):
  #   sequence = sequences[i]
  #   print sequence
  #   seq_vec = seq2vec(sequence, nDim)
  #   for j in xrange(len(sequence)-1):
  #     ds.addSample(seq_vec[j], seq_vec[j+1])
  #   ds.newSequence()

  rptPerSeqList = [1, 2, 5, 10, 20, 50, 100, 250, 500, 1000]
  accuracyList = []
  for rptNum in rptPerSeqList:
    # train LSTM
    # net = initializeLSTMnet(nDim, nLSTMcells=30)
    # net.reset()

    # trainer = RPropMinusTrainer(net)

    # for _ in xrange(rptNum):
    #   # Batch training mode
    #   # print "generate a dataset of sequences"
    #   ds = SequentialDataSet(nDim, nDim)
    #   trainer.setData(ds)
    #   import random
    #   random.shuffle(sequences)
    #   concat_sequences = []
    #   for sequence in sequences:
    #     concat_sequences += sequence
    #     concat_sequences.append(random.randrange(100, 1000000))
    #   # concat_sequences = sum(sequences, [])
    #   for j in xrange(len(concat_sequences) - 1):
    #     ds.addSample(num2vec(concat_sequences[j], nDim), num2vec(concat_sequences[j+1], nDim))

    #   trainer.train()
    net = initializeLSTMnet(nDim, nLSTMcells=50)
    net.reset()
    ds = SequentialDataSet(nDim, nDim)
    trainer = RPropMinusTrainer(net)
    trainer.setData(ds)
    for _ in xrange(1000):
      # Batch training mode
      # print "generate a dataset of sequences"
      import random
      random.shuffle(sequences)
      concat_sequences = []
      for sequence in sequences:
        concat_sequences += sequence
        concat_sequences.append(random.randrange(100, 1000000))
    for j in xrange(len(concat_sequences) - 1):
      ds.addSample(num2vec(concat_sequences[j], nDim), num2vec(concat_sequences[j+1], nDim))

    trainer.trainEpochs(rptNum)

    print
    print "test LSTM, repeats =", rptNum
    # test LSTM
    correct = []
    for i in xrange(len(sequences)):
      net.reset()
      sequence = sequences[i]
      sequence = sequence + [random.randrange(100, 1000000)]
      print sequence
      predictedInput = []
      for j in xrange(len(sequence)):
        sample = num2vec(sequence[j], nDim)
        netActivation = net.activate(sample)
        if j+1 < len(sequence) - 1:
          predictedInput.append(classify(netActivation))
          print " actual input: ", sequence[j+1], " predicted Input: ", predictedInput[j]

          correct.append(predictedInput[j] == sequence[j+1])

      # correct.append(predictedInput[-1] == sequence[-1])

    accuracyList.append(sum(correct)/float(len(correct)))
    print "Accuracy: ", accuracyList[-1]

  plt.semilogx(np.array(rptPerSeqList), np.array(accuracyList), '-*')
  plt.xlabel(' Repeat of entire batch')
  plt.ylabel(' Accuracy ')

  plt.show()

  # online mode (does not work well)
  # net = initializeLSTMnet(nDim, nLSTMcells=20)
  # accuracyList = []
  # for seq in xrange(5000):
  #   sequence = random.choice(sequences)
  #   print sequence
  #   seq_vec = seq2vec(sequence, nDim)
  #   ds = SequentialDataSet(nDim, nDim)
  #   for j in xrange(len(sequence)-1):
  #     ds.addSample(seq_vec[j], seq_vec[j+1])
  #
  #   # test LSTM
  #   net.reset()
  #   predictedInput = []
  #   for i in xrange(len(sequence)-1):
  #     sample = num2vec(sequence[i], nDim)
  #     netActivation = net.activate(sample)
  #     predictedInput.append(np.argmax(netActivation))
  #     print " predicted Input: ", predictedInput[i], " actual input: ", sequence[i+1]
  #
  #   accuracyList.append(predictedInput[-1] == sequence[-1])
  #
  #   # train LSTM
  #   net.reset()
  #   trainer = RPropMinusTrainer(net, dataset=ds)
  #   trainer.trainEpochs(1)
  #
  #   # test LSTM on the whole dataset
  #   # correct = []
  #   # for i in xrange(len(sequences)):
  #   #   sequence = sequences[i]
  #   #   print sequence
  #   #   net.reset()
  #   #   predictedInput = []
  #   #   for j in xrange(len(sequence)-1):
  #   #     sample = num2vec(sequence[j], nDim)
  #   #     netActivation = net.activate(sample)
  #   #     predictedInput.append(np.argmax(netActivation))
  #   #     print " actual input: ", sequence[j+1], " predicted Input: ", predictedInput[j]
  #   #
  #   #   correct.append(predictedInput[-1] == sequence[-1])
  #   # accuracyList.append(sum(correct)/float(len(correct)))
  #
  #   if seq % 100 == 0:
  #       rcParams.update({'figure.figsize': (12, 6)})
  #       plt.figure(1)
  #       plt.clf()
  #       plotAccuracy(accuracyList)
  #       plt.draw()

